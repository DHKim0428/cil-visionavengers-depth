"""
Partial fine-tune of Depth-Anything-V2 on the CIL monocular depth dataset.

Supported strategies (--strategy):
  decoder           — full DPT decoder (depth_head), backbone frozen
  refinenets_output — only refinenet{1-4} + output_conv{1,2}, rest frozen
"""

import argparse
import logging
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

REPO        = '/work/courses/3dv/team24/Depth-Anything-V2'
sys.path.insert(0, REPO)
sys.path.insert(0, '/home/heelee/cil-visionavengers-depth')

from depth_anything_v2.dpt import DepthAnythingV2
from dataset.data_loader import CILDepthDataset, rgb_names, split_names

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

MODEL_CFGS = {
    'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48,   96,  192,  384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96,  192,  384,  768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}
DEFAULT_CKPTS = {
    'vits': '/home/heelee/checkpoints/depth_anything_v2_vits.pth',
    'vitb': '/home/heelee/checkpoints/depth_anything_v2_vitb.pth',
}

# Layers to train per strategy (matched against full param name)
STRATEGY_LAYERS = {
    'decoder': ['depth_head'],
    'refinenets_output': [
        'depth_head.scratch.refinenet',
        'depth_head.scratch.output_conv',
    ],
}

STRATEGY_DESC = {
    'decoder': (
        'Full decoder fine-tune. The entire DPTHead (depth_head) is trained '
        'while the DINOv2 backbone (pretrained) is frozen.'
    ),
    'refinenets_output': (
        'Partial decoder fine-tune. Only the four FeatureFusionBlock (refinenet1-4) '
        'and the two output convolutions (output_conv1, output_conv2) are trained. '
        'The backbone, projection layers, resize layers, and channel-reduction '
        'convs (layer_rn) are all frozen.'
    ),
}

MIN_DEPTH = 0.001
MAX_DEPTH = 80.0

parser = argparse.ArgumentParser()
parser.add_argument('--encoder',         default='vits', choices=list(MODEL_CFGS))
parser.add_argument('--checkpoint',      default=None)
parser.add_argument('--strategy',        default='refinenets_output',
                    choices=list(STRATEGY_LAYERS),
                    help='Which parts of the model to train')
parser.add_argument('--data-dir',        default='/cluster/courses/cil/monocular-depth-estimation/train')
parser.add_argument('--img-size',        default=518, type=int)
parser.add_argument('--epochs',          default=10,  type=int)
parser.add_argument('--bs',              default=8,   type=int)
parser.add_argument('--lr',              default=1e-4, type=float)
parser.add_argument('--save-path',       default='/home/heelee/checkpoints/cil_refinenets_finetune')
parser.add_argument('--val-fraction',    default=0.05, type=float)
parser.add_argument('--resume',          default=None)


def sirmse_loss(pred, target, valid_mask):
    eps  = 1e-6
    mask = valid_mask & (target >= MIN_DEPTH) & (target <= MAX_DEPTH) & (pred > 0)
    gt_disp = 1.0 / target[mask]
    d = torch.log(pred[mask].clamp(min=eps)) - torch.log(gt_disp.clamp(min=eps))
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


def apply_strategy(model, strategy):
    """Freeze all params, then unfreeze those matching the strategy prefixes."""
    prefixes = STRATEGY_LAYERS[strategy]
    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(p) for p in prefixes)

    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    return trainable_names


def main():
    args  = parser.parse_args()
    ckpt  = args.checkpoint or DEFAULT_CKPTS[args.encoder]
    os.makedirs(args.save_path, exist_ok=True)
    writer = SummaryWriter(args.save_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_tag = f'da2_{args.encoder}'
    names = rgb_names(args.data_dir)
    train_names, val_names = split_names(names, args.val_fraction, seed=42)
    trainset = CILDepthDataset(args.data_dir, train_names, model=model_tag,
                               image_size=args.img_size, training=True)
    valset   = CILDepthDataset(args.data_dir, val_names,   model=model_tag,
                               image_size=args.img_size, training=False)

    trainloader = DataLoader(trainset, batch_size=args.bs, shuffle=True,
                             pin_memory=True, num_workers=4, drop_last=True)
    valloader   = DataLoader(valset,   batch_size=1, shuffle=False,
                             pin_memory=True, num_workers=4)

    model = DepthAnythingV2(**MODEL_CFGS[args.encoder])
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))

    trainable_names = apply_strategy(model, args.strategy)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in trainable_params)
    frozen_p    = total_p - trainable_p

    # Log setup
    logger.info(f'Device       : {device}')
    logger.info(f'Encoder      : {args.encoder}')
    logger.info(f'Strategy     : {args.strategy}')
    logger.info(f'Checkpoint   : {ckpt}')
    logger.info(f'Train samples: {len(trainset)}  |  Val samples: {len(valset)}')
    logger.info(f'Total params : {total_p:,}')
    logger.info(f'Frozen params: {frozen_p:,}')
    logger.info(f'Trainable    : {trainable_p:,}  ({100*trainable_p/total_p:.1f}% of total)')
    logger.info('Trainable layers:')
    # Group by top-level component
    seen = set()
    for n in trainable_names:
        key = '.'.join(n.split('.')[:4])
        if key not in seen:
            seen.add(key)
            p = sum(x.numel() for name, x in model.named_parameters() if name.startswith(key))
            logger.info(f'  {key:<55s}  {p:>8,} params')

    model = model.to(device)
    optimizer   = AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    total_iters = args.epochs * len(trainloader)
    best_sirmse = float('inf')
    start_epoch = 0
    epoch_val_scores = []
    epoch_times = []
    prior_elapsed = 0.0

    if args.resume:
        data = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(data['model'])
        optimizer.load_state_dict(data['optimizer'])
        start_epoch = data['epoch'] + 1
        best_sirmse = data['best_sirmse']
        if 'epoch_val_scores' in data:
            epoch_val_scores = data['epoch_val_scores']
        if 'epoch_times' in data:
            epoch_times = data['epoch_times']
        if 'prior_elapsed' in data:
            prior_elapsed = data['prior_elapsed']
        logger.info(f'Resumed from epoch {data["epoch"]}  best={best_sirmse:.4f}')

    train_start = time.time()
    epoch_times = []

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        model.train()

        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()
            img        = sample['image'].to(device)
            depth      = sample['depth'].to(device)
            valid_mask = sample['valid_mask'].to(device)

            if random.random() < 0.5:
                img, depth, valid_mask = img.flip(-1), depth.flip(-1), valid_mask.flip(-1)

            pred_disp = model(img)
            loss      = sirmse_loss(pred_disp, depth, valid_mask)
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
            with torch.no_grad():
                pred_disp = model(img)
                pred_disp = F.interpolate(pred_disp[:, None], depth.shape[-2:],
                                          mode='bilinear', align_corners=True)[0, 0]
            s = sirmse_eval(pred_disp, depth, valid_mask)
            if s is not None:
                scores.append(s)

        val_sirmse = float(np.mean(scores))
        epoch_val_scores.append((epoch, val_sirmse))
        logger.info(f'Epoch {epoch}  val siRMSE = {val_sirmse:.4f}  (best={best_sirmse:.4f})')
        writer.add_scalar('val/sirmse', val_sirmse, epoch)

        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'epoch': epoch, 'best_sirmse': best_sirmse,
                    'epoch_val_scores': epoch_val_scores,
                    'epoch_times': epoch_times,
                    'prior_elapsed': prior_elapsed + (time.time() - train_start)},
                   os.path.join(args.save_path, 'latest.pth'))
        if val_sirmse < best_sirmse:
            best_sirmse = val_sirmse
            torch.save(model.state_dict(), os.path.join(args.save_path, 'best.pth'))
            logger.info(f'  → new best: {best_sirmse:.4f}')

    total_elapsed = time.time() - train_start
    total_wall = prior_elapsed + total_elapsed

    # ── Detailed run log ──────────────────────────────────────────────────────
    log_lines = [
        '=' * 70,
        f'Run completed : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '=' * 70,
        '',
        '[ What was implemented ]',
        f'  Strategy : {args.strategy}',
        f'  {STRATEGY_DESC[args.strategy]}',
        '',
        '[ How it was implemented ]',
        f'  Model           : Depth-Anything-V2  encoder={args.encoder}',
        f'  Base checkpoint : {ckpt}',
        f'  Total params    : {total_p:,}',
        f'  Frozen params   : {frozen_p:,}',
        f'  Trainable params: {trainable_p:,}  ({100*trainable_p/total_p:.1f}% of total)',
        '  Trainable layers:',
    ]
    for n in seen:
        p = sum(x.numel() for name, x in model.named_parameters() if name.startswith(n))
        log_lines.append(f'    {n:<55s}  {p:>8,} params')
    log_lines += [
        '',
        f'  Loss            : siRMSE = sqrt( mean(d²) - mean(d)² )',
        f'                    d = log(pred_disp) - log(1/gt_depth)',
        f'                    valid mask: gt in [{MIN_DEPTH}, {MAX_DEPTH}]',
        f'  Optimizer       : AdamW  lr={args.lr}  betas=(0.9,0.999)  wd=0.01',
        f'  LR schedule     : poly decay  (1 - iter/total)^0.9',
        f'  Batch size      : {args.bs}',
        f'  Epochs          : {args.epochs}',
        f'  Image size      : {args.img_size}×{args.img_size}',
        f'  Train samples   : {len(trainset)}  (val fraction={args.val_fraction})',
        f'  Val samples     : {len(valset)}',
        '',
        '[ Results ]',
        '  Epoch  val siRMSE',
    ]
    for ep, sc in epoch_val_scores:
        marker = '  ← best' if sc == best_sirmse else ''
        log_lines.append(f'    {ep:>5}  {sc:.4f}{marker}')
    log_lines += [
        '',
        f'  Best val siRMSE : {best_sirmse:.4f}',
        f'  Zero-shot baseline (vits, 10% data): 0.6029',
        f'  Improvement over zero-shot         : {0.6029 - best_sirmse:+.4f}',
        '',
        '[ Training Time ]',
        f'  This run        : {fmt_dur(total_elapsed)}',
        f'  Total (all runs): {fmt_dur(total_wall)}',
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
