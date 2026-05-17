"""
DA-V2 Prior-Guided UNet Refinement.

  1. Pre-computed zero-shot DA-V2 (vits) disparity prior D0  (loaded from disk)
  2. UNetRefinement takes [RGB, log(D0)] and predicts log-space correction delta
  3. D_final = D0 * exp(delta)   <->   log(D_final) = log(D0) + delta
  4. siRMSE loss on D_final (lambda=1, competition metric)

DA-V2 priors are precomputed by precompute_da2_priors.py and cached to disk.
"""

import argparse
import importlib.util
import logging
import os
import random
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

REPO        = '/work/courses/3dv/team24/Depth-Anything-V2'
METRIC_REPO = os.path.join(REPO, 'metric_depth')
sys.path.insert(0, REPO)
sys.path.insert(0, METRIC_REPO)


_spec = importlib.util.spec_from_file_location(
    'team_model', '/home/heelee/cil-visionavengers-depth/model.py')
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
UNetRefinement = _mod.UNetRefinement

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

MIN_DEPTH = 0.001
MAX_DEPTH = 80.0
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir',    default='/cluster/courses/cil/monocular-depth-estimation/train')
parser.add_argument('--priors-dir',  default='/home/heelee/checkpoints/da2_priors')
parser.add_argument('--img-size',    default=518,  type=int)
parser.add_argument('--epochs',      default=10,   type=int)
parser.add_argument('--bs',          default=8,    type=int)
parser.add_argument('--lr',          default=1e-4, type=float)
parser.add_argument('--save-path',   default='/home/heelee/checkpoints/cil_refinement')
parser.add_argument('--val-fraction', default=0.05, type=float)
parser.add_argument('--num-workers', default=2,    type=int)
parser.add_argument('--resume',      default=None)


class CILDepthWithPrior(Dataset):
    """
    Same train/val split as CILDepth. Images, depths, and DA-V2 priors are all
    resized to a fixed img_size x img_size — no random crop, so the cached
    prior always matches the training image exactly.
    """
    def __init__(self, data_dir, priors_dir, mode, img_size=518,
                 val_fraction=0.05, seed=42):
        self.img_size   = img_size
        self.priors_dir = priors_dir
        self.mode       = mode

        all_rgb = sorted(f for f in os.listdir(data_dir) if f.endswith('_rgb.png'))
        rng     = np.random.default_rng(seed)
        indices = np.arange(len(all_rgb))
        rng.shuffle(indices)
        n_val   = max(1, int(len(all_rgb) * val_fraction))
        chosen  = indices[n_val:] if mode == 'train' else indices[:n_val]

        self.samples = [
            (
                os.path.join(data_dir, all_rgb[i]),
                os.path.join(data_dir, all_rgb[i].replace('_rgb.png', '_depth.npy')),
                os.path.join(priors_dir, all_rgb[i].replace('_rgb.png', '_da2_prior.npy')),
            )
            for i in sorted(chosen)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, depth_path, prior_path = self.samples[idx]
        sz = self.img_size

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = cv2.resize(img, (sz, sz), interpolation=cv2.INTER_CUBIC)
        img = ((img - MEAN) / STD).transpose(2, 0, 1)  # CHW

        depth = np.load(depth_path).astype(np.float32)
        depth = cv2.resize(depth, (sz, sz), interpolation=cv2.INTER_NEAREST)

        prior = np.load(prior_path).astype(np.float32)  # sz x sz

        return {
            'image':      torch.from_numpy(img),
            'depth':      torch.from_numpy(depth),
            'valid_mask': torch.from_numpy(depth > 0),
            'prior':      torch.from_numpy(prior),
        }


def sirmse_loss(pred_disp, gt_depth, valid_mask):
    eps  = 1e-6
    mask = valid_mask & (gt_depth >= MIN_DEPTH) & (gt_depth <= MAX_DEPTH) & (pred_disp > 0)
    gt_disp = 1.0 / gt_depth[mask]
    d = torch.log(pred_disp[mask].clamp(min=eps)) - torch.log(gt_disp.clamp(min=eps))
    return torch.sqrt(d.pow(2).mean() - d.mean().pow(2) + eps)


def sirmse_eval(pred_disp, gt_depth, valid_mask):
    eps  = 1e-6
    mask = valid_mask & (gt_depth >= MIN_DEPTH) & (gt_depth <= MAX_DEPTH) & (pred_disp > 0)
    if mask.sum() < 10:
        return None
    pred_depth = (1.0 / pred_disp[mask].clamp(min=eps)).clamp(MIN_DEPTH, MAX_DEPTH)
    g = gt_depth[mask]
    d = torch.log(pred_depth.clamp(min=eps)) - torch.log(g.clamp(min=eps))
    return torch.sqrt(d.pow(2).mean() - d.mean().pow(2) + eps).item()


def fmt_dur(s):
    return f'{int(s//3600)}h {int((s%3600)//60)}m {int(s%60)}s'


def main():
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    writer = SummaryWriter(args.save_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainset = CILDepthWithPrior(args.data_dir, args.priors_dir, 'train',
                                  img_size=args.img_size, val_fraction=args.val_fraction)
    valset   = CILDepthWithPrior(args.data_dir, args.priors_dir, 'val',
                                  img_size=args.img_size, val_fraction=args.val_fraction)
    trainloader = DataLoader(trainset, batch_size=args.bs, shuffle=True,
                             pin_memory=True, num_workers=args.num_workers, drop_last=True)
    valloader   = DataLoader(valset,   batch_size=1, shuffle=False,
                             pin_memory=True, num_workers=args.num_workers)

    unet    = UNetRefinement().to(device)
    total_p = sum(p.numel() for p in unet.parameters())

    logger.info(f'Device        : {device}')
    logger.info(f'Priors dir    : {args.priors_dir}')
    logger.info(f'UNet params   : {total_p:,}')
    logger.info(f'Train samples : {len(trainset)}  |  Val samples: {len(valset)}')

    optimizer   = AdamW(unet.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    total_iters = args.epochs * len(trainloader)
    best_sirmse = float('inf')
    start_epoch = 0
    epoch_val_scores = []

    if args.resume:
        data = torch.load(args.resume, map_location='cpu')
        unet.load_state_dict(data['model'])
        optimizer.load_state_dict(data['optimizer'])
        start_epoch = data['epoch'] + 1
        best_sirmse = data['best_sirmse']
        if 'epoch_val_scores' in data:
            epoch_val_scores = data['epoch_val_scores']
        logger.info(f'Resumed from epoch {data["epoch"]}  best={best_sirmse:.4f}')

    train_start = time.time()
    epoch_times = []

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        unet.train()

        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()
            img        = sample['image'].to(device)
            depth      = sample['depth'].to(device)
            valid_mask = sample['valid_mask'].to(device)
            d_prior    = sample['prior'].to(device)

            if random.random() < 0.5:
                img, depth, valid_mask, d_prior = (
                    img.flip(-1), depth.flip(-1), valid_mask.flip(-1), d_prior.flip(-1))

            d_final = unet(img, d_prior)
            loss    = sirmse_loss(d_final, depth, valid_mask)
            loss.backward()
            optimizer.step()

            iters = epoch * len(trainloader) + i
            lr    = args.lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]['lr'] = lr
            writer.add_scalar('train/loss', loss.item(), iters)
            if i % 200 == 0:
                logger.info(f'Epoch {epoch}/{args.epochs}  iter {i}/{len(trainloader)}'
                            f'  lr={lr:.2e}  loss={loss.item():.4f}')

        epoch_elapsed = time.time() - epoch_start
        epoch_times.append(epoch_elapsed)
        logger.info(f'Epoch {epoch} done in {fmt_dur(epoch_elapsed)}')

        # Validation
        unet.eval()
        scores = []
        for sample in valloader:
            img        = sample['image'].to(device)
            depth      = sample['depth'].to(device)[0]
            valid_mask = sample['valid_mask'].to(device)[0]
            d_prior    = sample['prior'].to(device)
            with torch.no_grad():
                d_final = unet(img, d_prior)
                d_final = F.interpolate(d_final[:, None], depth.shape[-2:],
                                        mode='bilinear', align_corners=True)[0, 0]
            s = sirmse_eval(d_final, depth, valid_mask)
            if s is not None:
                scores.append(s)

        val_sirmse = float(np.mean(scores))
        epoch_val_scores.append((epoch, val_sirmse))
        logger.info(f'Epoch {epoch}  val siRMSE = {val_sirmse:.4f}  (best={best_sirmse:.4f})')
        writer.add_scalar('val/sirmse', val_sirmse, epoch)

        torch.save({'model': unet.state_dict(), 'optimizer': optimizer.state_dict(),
                    'epoch': epoch, 'best_sirmse': best_sirmse,
                    'epoch_val_scores': epoch_val_scores},
                   os.path.join(args.save_path, 'latest.pth'))
        if val_sirmse < best_sirmse:
            best_sirmse = val_sirmse
            torch.save(unet.state_dict(), os.path.join(args.save_path, 'best.pth'))
            logger.info(f'  -> new best: {best_sirmse:.4f}')

    total_elapsed = time.time() - train_start

    log_lines = [
        '=' * 70,
        f'Run completed : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '=' * 70,
        '',
        '[ What was implemented ]',
        '  Strategy : DA-V2 Prior + UNet Refinement (Option A)',
        '  Zero-shot DA-V2 (vits) disparity priors are precomputed at 518x518',
        '  and cached to disk. UNetRefinement([RGB, log(D0)]) predicts',
        '  log-space correction delta. D_final = D0 * exp(delta).',
        '  Output conv zero-initialized: identity correction at epoch 0.',
        '',
        '[ How it was implemented ]',
        f'  DA-V2 prior     : zero-shot vits (precomputed, frozen)',
        f'  Priors dir      : {args.priors_dir}',
        f'  UNet params     : {total_p:,}  (all trainable)',
        f'  Loss            : siRMSE (lambda=1, competition metric)',
        f'  Optimizer       : AdamW  lr={args.lr}  betas=(0.9,0.999)  wd=0.01',
        f'  LR schedule     : poly decay  (1 - iter/total)^0.9',
        f'  Batch size      : {args.bs}',
        f'  Epochs          : {args.epochs}',
        f'  Image size      : {args.img_size}x{args.img_size}',
        f'  Train samples   : {len(trainset)}  (val fraction={args.val_fraction})',
        f'  Val samples     : {len(valset)}',
        '',
        '[ Results ]',
        '  Epoch  val siRMSE',
    ]
    for ep, sc in epoch_val_scores:
        marker = '  <- best' if sc == best_sirmse else ''
        log_lines.append(f'    {ep:>5}  {sc:.4f}{marker}')
    log_lines += [
        '',
        f'  Best val siRMSE              : {best_sirmse:.4f}',
        f'  DA-V2 full decoder FT        : 0.4612',
        f'  DA-V2 refinenets+output FT   : 0.5135',
        f'  DA-V2 zero-shot (vits)       : 0.6029',
        '',
        '[ Training Time ]',
        f'  Total           : {fmt_dur(total_elapsed)}',
        f'  Per epoch (avg) : {fmt_dur(np.mean(epoch_times))}',
        f'  Per epoch (min) : {fmt_dur(min(epoch_times))}',
        f'  Per epoch (max) : {fmt_dur(max(epoch_times))}',
        '=' * 70,
    ]
    log_text = '\n'.join(log_lines)
    logger.info('\n' + log_text)
    log_path = os.path.join(args.save_path, 'run_log.txt')
    with open(log_path, 'w') as f:
        f.write(log_text + '\n')
    logger.info(f'Run log saved to {log_path}')


if __name__ == '__main__':
    main()
