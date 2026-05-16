import argparse
import os
import sys
import random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm

sys.path.insert(0, '/home/dchileban/cil/Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2


DATA_DIR = '/cluster/courses/cil/monocular-depth-estimation/train'
CKPT_PATH = '/home/dchileban/cil/checkpoints/depth_anything_v2_vitb.pth'

MODEL_CFG = {
    'encoder': 'vitb',
    'features': 128,
    'out_channels': [96, 192, 384, 768]
}

MIN_DEPTH = 0.001
MAX_DEPTH = 80.0

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class CILDepthDataset(Dataset):
    def __init__(self, data_dir, filenames, input_size=518, augment=False):
        self.data_dir = data_dir
        self.filenames = filenames
        self.input_size = input_size
        self.augment = augment

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        rgb_name = self.filenames[idx]
        rgb_path = os.path.join(self.data_dir, rgb_name)
        depth_path = os.path.join(
            self.data_dir,
            rgb_name.replace('_rgb.png', '_depth.npy')
        )

        image = cv2.imread(rgb_path)
        if image is None:
            raise RuntimeError(f'Could not read image: {rgb_path}')

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = np.load(depth_path).astype(np.float32)

        image = cv2.resize(
            image,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR
        )

        depth = cv2.resize(
            depth,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_NEAREST
        )

        if self.augment:
            if random.random() < 0.5:
                image = np.ascontiguousarray(image[:, ::-1])
                depth = np.ascontiguousarray(depth[:, ::-1])

            if random.random() < 0.3:
                factor = random.uniform(0.8, 1.2)
                image = np.clip(
                    image.astype(np.float32) * factor,
                    0,
                    255
                ).astype(np.uint8)

        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image = image.transpose(2, 0, 1)

        valid = (depth >= MIN_DEPTH) & (depth <= MAX_DEPTH)

        return {
            'image': torch.from_numpy(image).float(),
            'depth': torch.from_numpy(depth).float(),
            'valid': torch.from_numpy(valid).bool(),
            'name': rgb_name,
        }


def si_rmse_loss_from_disparity(pred_disp, gt_depth, valid_mask):
    eps = 1e-6

    pred_disp = torch.clamp(pred_disp, min=eps)
    gt_depth = torch.clamp(gt_depth, min=MIN_DEPTH, max=MAX_DEPTH)

    log_pred_depth = -torch.log(pred_disp)
    log_gt_depth = torch.log(gt_depth)

    d = log_pred_depth - log_gt_depth
    valid_mask = valid_mask & torch.isfinite(d)

    losses = []

    for i in range(d.shape[0]):
        di = d[i][valid_mask[i]]

        if di.numel() < 10:
            continue

        loss_i = torch.mean(di ** 2) - torch.mean(di) ** 2
        loss_i = torch.sqrt(torch.clamp(loss_i, min=1e-8))
        losses.append(loss_i)

    if len(losses) == 0:
        return torch.tensor(0.0, device=pred_disp.device, requires_grad=True)

    return torch.stack(losses).mean()


def normalize_for_vis(x):
    x = x.astype(np.float32)
    lo = np.percentile(x, 2)
    hi = np.percentile(x, 98)
    x = (x - lo) / max(hi - lo, 1e-6)
    return np.clip(x, 0, 1)


def colorize_depth(depth):
    depth_norm = normalize_for_vis(depth)
    colored = cm.get_cmap('Spectral_r')(depth_norm)[..., :3]
    return colored.astype(np.float32)


@torch.no_grad()
def log_validation_images(writer, model, loader, device, epoch, max_images=4):
    model.eval()

    batch = next(iter(loader))

    image = batch['image'].to(device)
    depth = batch['depth'].to(device)
    valid = batch['valid'].to(device)

    pred_disp = model(image)

    if pred_disp.ndim == 4:
        pred_disp = pred_disp.squeeze(1)

    if pred_disp.shape[-2:] != depth.shape[-2:]:
        pred_disp = F.interpolate(
            pred_disp.unsqueeze(1),
            size=depth.shape[-2:],
            mode='bilinear',
            align_corners=False
        ).squeeze(1)

    pred_depth = 1.0 / torch.clamp(pred_disp, min=1e-6)
    pred_depth = torch.clamp(pred_depth, MIN_DEPTH, MAX_DEPTH)

    image = image.detach().cpu()
    depth = depth.detach().cpu()
    pred_depth = pred_depth.detach().cpu()
    valid = valid.detach().cpu()

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    rows = []
    n = min(max_images, image.shape[0])

    for i in range(n):
        rgb = image[i] * std + mean
        rgb = torch.clamp(rgb, 0, 1).numpy().transpose(1, 2, 0)

        gt_np = depth[i].numpy()
        pred_np = pred_depth[i].numpy()
        valid_np = valid[i].numpy()

        gt_vis = colorize_depth(gt_np)
        pred_vis = colorize_depth(pred_np)

        gt_vis[~valid_np] = 0.1

        row = np.concatenate([rgb, gt_vis, pred_vis], axis=1)
        rows.append(row)

    grid = np.concatenate(rows, axis=0)
    grid = torch.from_numpy(grid).permute(2, 0, 1)

    writer.add_image('val/RGB_GT_Pred', grid, epoch)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses = []

    for batch in tqdm(loader, desc='Validation', leave=False):
        image = batch['image'].to(device)
        depth = batch['depth'].to(device)
        valid = batch['valid'].to(device)

        pred = model(image)

        if pred.ndim == 4:
            pred = pred.squeeze(1)

        if pred.shape[-2:] != depth.shape[-2:]:
            pred = F.interpolate(
                pred.unsqueeze(1),
                size=depth.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(1)

        loss = si_rmse_loss_from_disparity(pred, depth, valid)
        losses.append(loss.item())

    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', type=str, default=DATA_DIR)
    parser.add_argument('--ckpt', type=str, default=CKPT_PATH)
    parser.add_argument('--output-dir', type=str, default='/home/dchileban/cil/logs')

    parser.add_argument('--log-dir', type=str, default='/home/dchileban/cil/logs')
    parser.add_argument('--log-images-every', type=int, default=1)
    parser.add_argument('--num-log-images', type=int, default=4)

    parser.add_argument('--input-size', type=int, default=392)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--accum-steps', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-6)
    parser.add_argument('--weight-decay', type=float, default=1e-4)

    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--freeze-encoder', action='store_true')
    parser.add_argument('--amp', action='store_true')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    writer = SummaryWriter(args.log_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    all_rgb = sorted(f for f in os.listdir(args.data_dir) if f.endswith('_rgb.png'))

    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(all_rgb))
    rng.shuffle(indices)

    val_size = int(len(all_rgb) * args.val_fraction)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_files = [all_rgb[i] for i in train_indices]
    val_files = [all_rgb[i] for i in val_indices]

    print(f'Train samples: {len(train_files)}')
    print(f'Val samples  : {len(val_files)}')

    train_dataset = CILDepthDataset(
        args.data_dir,
        train_files,
        input_size=args.input_size,
        augment=True
    )

    val_dataset = CILDepthDataset(
        args.data_dir,
        val_files,
        input_size=args.input_size,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = DepthAnythingV2(**MODEL_CFG)
    state_dict = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)

    if args.freeze_encoder:
        print('Freezing encoder parameters...')
        for name, param in model.named_parameters():
            if 'pretrained' in name or 'encoder' in name:
                param.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_val = float('inf')
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar):
            image = batch['image'].to(device, non_blocking=True)
            depth = batch['depth'].to(device, non_blocking=True)
            valid = batch['valid'].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                pred = model(image)

                if pred.ndim == 4:
                    pred = pred.squeeze(1)

                if pred.shape[-2:] != depth.shape[-2:]:
                    pred = F.interpolate(
                        pred.unsqueeze(1),
                        size=depth.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1)

                loss = si_rmse_loss_from_disparity(pred, depth, valid)
                loss = loss / args.accum_steps

            scaler.scale(loss).backward()

            should_step = (
                (step + 1) % args.accum_steps == 0
                or (step + 1) == len(train_loader)
            )

            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad(set_to_none=True)

            true_loss = loss.item() * args.accum_steps
            train_losses.append(true_loss)

            writer.add_scalar('train/siRMSE_step', true_loss, global_step)
            writer.add_scalar('train/lr_step', optimizer.param_groups[0]['lr'], global_step)

            pbar.set_postfix(loss=f'{true_loss:.4f}')
            global_step += 1

        train_loss = float(np.mean(train_losses))
        val_loss = evaluate(model, val_loader, device)

        writer.add_scalar('train/siRMSE_epoch', train_loss, epoch)
        writer.add_scalar('val/siRMSE_epoch', val_loss, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        print(
            f'Epoch {epoch}: '
            f'train siRMSE={train_loss:.4f}, '
            f'val siRMSE={val_loss:.4f}'
        )

        if epoch % args.log_images_every == 0:
            log_validation_images(
                writer=writer,
                model=model,
                loader=val_loader,
                device=device,
                epoch=epoch,
                max_images=args.num_log_images
            )

        last_path = os.path.join(args.output_dir, 'last.pth')
        torch.save(model.state_dict(), last_path)

        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(args.output_dir, 'best.pth')
            torch.save(model.state_dict(), best_path)
            print(f'New best checkpoint saved: {best_path}')

    print(f'Best val siRMSE: {best_val:.4f}')
    writer.close()


if __name__ == '__main__':
    main()