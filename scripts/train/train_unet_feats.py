"""
DA-V2 Prior-Guided UNet Refinement with multi-scale feature injection.

Architecture:
  1. Precomputed DA2 features (f2, f3, f4) + disparity prior loaded from disk
  2. UNetRefinementWithFeats([RGB, log(D0)]) predicts log-space correction delta,
     guided by DA2 features injected at bottleneck / dec3 / dec2
  3. D_final = D0 * exp(delta)
  4. siRMSE loss

Run precompute_da2_features.py first to generate the .npz feature cache.
"""

import argparse
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

sys.path.insert(0, os.path.dirname(__file__))
from model_unet_feats import UNetRefinementWithFeats

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

MIN_DEPTH = 0.001
MAX_DEPTH = 80.0
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir',     default='/cluster/courses/cil/monocular-depth-estimation/train')
parser.add_argument('--feats-dir',    default='/work/scratch/heelee/da2_features')
parser.add_argument('--img-size',     default=518,  type=int)
parser.add_argument('--epochs',       default=10,   type=int)
parser.add_argument('--bs',           default=8,    type=int)
parser.add_argument('--lr',           default=1e-4, type=float)
parser.add_argument('--save-path',    default='/home/heelee/checkpoints/cil_unet_feats')
parser.add_argument('--val-fraction', default=0.05, type=float)
parser.add_argument('--num-workers',  default=2,    type=int)
parser.add_argument('--resume',       default=None)


class CILDepthWithFeats(Dataset):
    """
    RGB + GT depth는 data_dir에서,
    DA2 features (f2, f3, f4) + prior는 feats_dir의 .npz에서 로드.
    """
    def __init__(self, data_dir, feats_dir, mode, img_size=518,
                 val_fraction=0.05, seed=42):
        self.data_dir  = data_dir
        self.feats_dir = feats_dir
        self.img_size  = img_size

        all_rgb = sorted(f for f in os.listdir(data_dir) if f.endswith('_rgb.png'))
        rng     = np.random.default_rng(seed)
        indices = np.arange(len(all_rgb))
        rng.shuffle(indices)
        n_val  = max(1, int(len(all_rgb) * val_fraction))
        chosen = indices[n_val:] if mode == 'train' else indices[:n_val]

        self.samples = [
            (
                os.path.join(data_dir,  all_rgb[i]),
                os.path.join(data_dir,  all_rgb[i].replace('_rgb.png', '_depth.npy')),
                os.path.join(feats_dir, all_rgb[i].replace('_rgb.png', '_da2_feats.npz')),
            )
            for i in sorted(chosen)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, depth_path, feats_path = self.samples[idx]
        sz = self.img_size

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = cv2.resize(img, (sz, sz), interpolation=cv2.INTER_CUBIC)
        img = ((img - MEAN) / STD).transpose(2, 0, 1)

        depth = np.load(depth_path).astype(np.float32)
        depth = cv2.resize(depth, (sz, sz), interpolation=cv2.INTER_NEAREST)

        data  = np.load(feats_path)
        f2    = torch.from_numpy(data['f2'].astype(np.float32))    # [96,  74,  74]
        f3    = torch.from_numpy(data['f3'].astype(np.float32))    # [192, 37,  37]
        f4    = torch.from_numpy(data['f4'].astype(np.float32))    # [384, 19,  19]
        prior = torch.from_numpy(data['prior'].astype(np.float32)) # [518, 518]

        return {
            'image':      torch.from_numpy(img),
            'depth':      torch.from_numpy(depth),
            'valid_mask': torch.from_numpy(depth > 0),
            'prior':      prior,
            'f2':         f2,
            'f3':         f3,
            'f4':         f4,
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


def flip_sample(s):
    """Horizontal flip applied consistently to all spatial tensors."""
    return {k: s[k].flip(-1) for k in ('image', 'depth', 'valid_mask', 'prior', 'f2', 'f3', 'f4')}


def fmt_dur(sec):
    return f'{int(sec//3600)}h {int((sec%3600)//60)}m {int(sec%60)}s'


def main():
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    writer = SummaryWriter(args.save_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainset = CILDepthWithFeats(args.data_dir, args.feats_dir, 'train',
                                  img_size=args.img_size, val_fraction=args.val_fraction)
    valset   = CILDepthWithFeats(args.data_dir, args.feats_dir, 'val',
                                  img_size=args.img_size, val_fraction=args.val_fraction)
    trainloader = DataLoader(trainset, batch_size=args.bs, shuffle=True,
                             pin_memory=True, num_workers=args.num_workers, drop_last=True)
    valloader   = DataLoader(valset,   batch_size=1, shuffle=False,
                             pin_memory=True, num_workers=args.num_workers)

    model   = UNetRefinementWithFeats().to(device)
    total_p = sum(p.numel() for p in model.parameters())

    logger.info(f'Device        : {device}')
    logger.info(f'Feats dir     : {args.feats_dir}')
    logger.info(f'Model params  : {total_p:,}')
    logger.info(f'Train samples : {len(trainset)}  |  Val samples: {len(valset)}')

    optimizer   = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    total_iters = args.epochs * len(trainloader)
    best_sirmse = float('inf')
    start_epoch = 0
    epoch_val_scores = []

    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_sirmse = ckpt['best_sirmse']
        if 'epoch_val_scores' in ckpt:
            epoch_val_scores = ckpt['epoch_val_scores']
        logger.info(f'Resumed from epoch {ckpt["epoch"]}  best={best_sirmse:.4f}')

    train_start = time.time()
    epoch_times = []

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        model.train()

        for i, sample in enumerate(trainloader):
            if random.random() < 0.5:
                sample = flip_sample(sample)

            img        = sample['image'].to(device)
            depth      = sample['depth'].to(device)
            valid_mask = sample['valid_mask'].to(device)
            d_prior    = sample['prior'].to(device)
            f2         = sample['f2'].to(device)
            f3         = sample['f3'].to(device)
            f4         = sample['f4'].to(device)

            optimizer.zero_grad()
            d_final = model(img, d_prior, f2, f3, f4)
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
        model.eval()
        scores = []
        for sample in valloader:
            img        = sample['image'].to(device)
            depth      = sample['depth'].to(device)[0]
            valid_mask = sample['valid_mask'].to(device)[0]
            d_prior    = sample['prior'].to(device)
            f2         = sample['f2'].to(device)
            f3         = sample['f3'].to(device)
            f4         = sample['f4'].to(device)
            with torch.no_grad():
                d_final = model(img, d_prior, f2, f3, f4)
                d_final = F.interpolate(d_final[:, None], depth.shape[-2:],
                                        mode='bilinear', align_corners=True)[0, 0]
            s = sirmse_eval(d_final, depth, valid_mask)
            if s is not None:
                scores.append(s)

        val_sirmse = float(np.mean(scores))
        epoch_val_scores.append((epoch, val_sirmse))
        logger.info(f'Epoch {epoch}  val siRMSE = {val_sirmse:.4f}  (best={best_sirmse:.4f})')
        writer.add_scalar('val/sirmse', val_sirmse, epoch)

        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'epoch': epoch, 'best_sirmse': best_sirmse,
                    'epoch_val_scores': epoch_val_scores},
                   os.path.join(args.save_path, 'latest.pth'))
        if val_sirmse < best_sirmse:
            best_sirmse = val_sirmse
            torch.save(model.state_dict(), os.path.join(args.save_path, 'best.pth'))
            logger.info(f'  -> new best: {best_sirmse:.4f}')

    total_elapsed = time.time() - train_start

    log_lines = [
        '=' * 70,
        f'Run completed : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '=' * 70,
        '',
        '[ What was implemented ]',
        '  Strategy : DA-V2 Prior + UNet Refinement with multi-scale feature injection',
        '  DA2 features (f2,f3,f4) and prior are precomputed and cached to disk.',
        '  UNetRefinementWithFeats([RGB, log(D0)]) predicts log-space delta,',
        '  guided by DA2 features injected at bottleneck / dec3 / dec2.',
        '  D_final = D0 * exp(delta). Projection convs zero-initialized.',
        '',
        '[ How it was implemented ]',
        f'  Feats dir       : {args.feats_dir}',
        f'  Model params    : {total_p:,}',
        f'  Loss            : siRMSE',
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
        f'  DA-V2 prior + UNet           : 0.5576',
        f'  DA-V2 zero-shot (vits)       : 0.6029',
        '',
        '[ Training Time ]',
        f'  Total           : {fmt_dur(total_elapsed)}',
        f'  Per epoch (avg) : {fmt_dur(np.mean(epoch_times))}',
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
