import os
import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm  

from dataset import SimpleDepthDataset
from model import UNetBaseline

def silog_loss(pred, target, mask, lambda_=0.5, eps=1e-6):
    """Scale-Invariant Log RMSE (SILog) loss function"""
    valid = mask > 0
    if valid.sum() == 0:
        return pred.sum() * 0.0

    pred = pred[valid]
    target = target[valid]
    
    pred = torch.clamp(pred, min=eps)
    target = torch.clamp(target, min=eps)
    
    log_diff = torch.log(pred) - torch.log(target)
    
    mse = torch.mean(log_diff ** 2)
    mean = torch.mean(log_diff)
    
    return mse - lambda_ * (mean ** 2)

def make_or_load_split(dataset, val_split, split_seed, split_file=None):
    """Create a reproducible train/val split, optionally pinned by filenames on disk."""
    n_total = len(dataset)
    n_val = int(val_split * n_total)
    n_train = n_total - n_val
    if n_train <= 0 or n_val <= 0:
        raise ValueError(
            f"Invalid split: n_total={n_total}, val_split={val_split} gives "
            f"n_train={n_train}, n_val={n_val}"
        )

    rgb_names = [path.name for path in dataset.rgb_files]

    if split_file is not None:
        split_path = Path(split_file)
        if split_path.exists():
            with open(split_path, "r", encoding="utf-8") as f:
                split = json.load(f)

            name_to_idx = {name: idx for idx, name in enumerate(rgb_names)}
            missing = [
                name
                for name in split["train_names"] + split["val_names"]
                if name not in name_to_idx
            ]
            if missing:
                raise ValueError(
                    f"Split file {split_path} references {len(missing)} missing samples. "
                    f"First missing sample: {missing[0]}"
                )

            train_indices = [name_to_idx[name] for name in split["train_names"]]
            val_indices = [name_to_idx[name] for name in split["val_names"]]
            print(f"[*] Loaded split from {split_path}")
            return train_indices, val_indices

    generator = torch.Generator().manual_seed(split_seed)
    indices = torch.randperm(n_total, generator=generator).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    if split_file is not None:
        split_path = Path(split_file)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split = {
            "split_seed": split_seed,
            "val_split": val_split,
            "n_total": n_total,
            "train_names": [rgb_names[idx] for idx in train_indices],
            "val_names": [rgb_names[idx] for idx in val_indices],
        }
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split, f, indent=2)
        print(f"[*] Saved split to {split_path}")

    return train_indices, val_indices

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    # Dataset & DataLoader
    print(f"[*] Loading dataset from {args.data_root}...")
    train_base = SimpleDepthDataset(
        args.data_root,
        img_size=args.img_size,
        max_samples=args.max_samples,
        enable_hflip=args.enable_hflip,
        hflip_prob=args.hflip_prob,
        enable_rotation=args.enable_rotation,
        rotation_deg=args.rotation_deg,
        enable_crop=args.enable_crop,
        crop_scale_min=args.crop_scale_min,
        enable_color_jitter=args.enable_color_jitter,
        color_jitter_prob=args.color_jitter_prob,
        brightness=args.brightness,
        contrast=args.contrast,
        saturation=args.saturation,
        tilt_mode=args.tilt_mode,
        tilt_prob=args.tilt_prob,
        tilt_max_yaw_deg=args.tilt_max_yaw_deg,
        tilt_max_pitch_deg=args.tilt_max_pitch_deg,
        tilt_fov_deg=args.tilt_fov_deg,
        teacher_mask_dir=args.teacher_mask_dir,
    )
    val_base = SimpleDepthDataset(
        args.data_root,
        img_size=args.img_size,
        max_samples=args.max_samples,
        tilt_mode="none",
    )

    train_indices, val_indices = make_or_load_split(
        train_base,
        val_split=args.val_split,
        split_seed=args.split_seed,
        split_file=args.split_file,
    )
    train_dataset = Subset(train_base, train_indices)
    val_dataset = Subset(val_base, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"[*] Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(
        "[*] Basic augmentation: "
        f"hflip={args.enable_hflip}, rotation={args.enable_rotation}, "
        f"crop={args.enable_crop}, color_jitter={args.enable_color_jitter}"
    )
    print(f"[*] Tilt augmentation: mode={args.tilt_mode}, prob={args.tilt_prob}")
    if args.teacher_mask_dir:
        print(f"[*] Teacher reliability masks: enabled for training only ({args.teacher_mask_dir})")
    else:
        print("[*] Teacher reliability masks: disabled")

    # Model & Optimizer
    model = UNetBaseline().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)

    def run_epoch(loader, epoch_idx, is_train=True):
        model.train() if is_train else model.eval()
        total_loss = 0.0
        
        mode_str = "Train" if is_train else "Val"
        pbar = tqdm(loader, desc=f"Epoch [{epoch_idx}/{args.num_epochs}] {mode_str}")
        
        for batch in pbar:
            images = batch["image"].to(device)
            depths = batch["depth"].to(device)
            masks = batch["mask"].to(device)
            
            with torch.set_grad_enabled(is_train):
                preds = model(images)
                loss = silog_loss(preds, depths, masks)
                
                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        return total_loss / len(loader)

    # Training Loop
    for epoch in range(1, args.num_epochs + 1):
        train_loss = run_epoch(train_loader, epoch, is_train=True)
        val_loss = run_epoch(val_loader, epoch, is_train=False)
        
        print(f">>> Epoch [{epoch}/{args.num_epochs}] Summary | Train SILog: {train_loss:.4f} | Val SILog: {val_loss:.4f}\n")
        
        # Save model checkpoint
        checkpoint_path = os.path.join(args.save_dir, f"unet_epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)

    print("[*] Training Complete!")

if __name__ == "__main__":
    user_name = os.environ.get("USER", "student")
    scratch_save_dir = f"/work/scratch/{user_name}/cil-visionavengers-depth/checkpoints"

    parser = argparse.ArgumentParser(description="Monocular Depth Estimation Baseline")
    parser.add_argument("--data_root", type=str, default="/cluster/courses/cil/monocular-depth-estimation/train", help="Dataset path")
    parser.add_argument("--save_dir", type=str, default=scratch_save_dir, help="Where to save models")
    parser.add_argument("--img_size", type=int, default=128, help="Image resolution for training")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.20, help="Ratio of validation data (e.g., 0.20 for 20%)")
    parser.add_argument("--split_seed", type=int, default=42, help="Random seed for deterministic train/val split")
    parser.add_argument("--split_file", type=str, default=None, help="Optional JSON file to save/load train/val filenames")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap for quick debugging runs")
    parser.add_argument("--enable_hflip", action="store_true", help="Enable horizontal flip augmentation on the training split")
    parser.add_argument("--hflip_prob", type=float, default=0.5, help="Probability of applying horizontal flip")
    parser.add_argument("--enable_rotation", action="store_true", help="Enable small in-plane rotations on the training split")
    parser.add_argument("--rotation_deg", type=float, default=3.0, help="Maximum absolute rotation angle in degrees")
    parser.add_argument("--enable_crop", action="store_true", help="Enable random square crop-and-resize augmentation on the training split")
    parser.add_argument("--crop_scale_min", type=float, default=0.9, help="Minimum retained area scale for random square crop")
    parser.add_argument("--enable_color_jitter", action="store_true", help="Enable mild RGB-only color jitter on the training split")
    parser.add_argument("--color_jitter_prob", type=float, default=0.8, help="Probability of applying color jitter")
    parser.add_argument("--brightness", type=float, default=0.1, help="Brightness jitter strength")
    parser.add_argument("--contrast", type=float, default=0.1, help="Contrast jitter strength")
    parser.add_argument("--saturation", type=float, default=0.1, help="Saturation jitter strength")
    parser.add_argument("--tilt_mode", type=str, default="none", choices=["none", "naive", "geometry"], help="Tilt augmentation type for training only")
    parser.add_argument("--tilt_prob", type=float, default=0.5, help="Probability of applying tilt augmentation to a training sample")
    parser.add_argument("--tilt_max_yaw_deg", type=float, default=5.0, help="Maximum absolute yaw angle in degrees")
    parser.add_argument("--tilt_max_pitch_deg", type=float, default=5.0, help="Maximum absolute pitch angle in degrees")
    parser.add_argument("--tilt_fov_deg", type=float, default=60.0, help="Assumed horizontal field of view for approximate intrinsics")
    parser.add_argument("--teacher_mask_dir", type=str, default=None, help="Optional training-only DA3 reliability mask directory")
    
    args = parser.parse_args()
    main(args)
