import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm  

from dataset import SimpleDepthDataset
from model import UNetBaseline

def silog_loss(pred, target, mask, lambda_=0.5, eps=1e-6):
    """Scale-Invariant Log RMSE (SILog) loss function"""
    pred = pred[mask > 0]
    target = target[mask > 0]
    
    pred = torch.clamp(pred, min=eps)
    target = torch.clamp(target, min=eps)
    
    log_diff = torch.log(pred) - torch.log(target)
    
    mse = torch.mean(log_diff ** 2)
    mean = torch.mean(log_diff)
    
    return mse - lambda_ * (mean ** 2)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    # Dataset & DataLoader
    print(f"[*] Loading dataset from {args.data_root}...")
    full_dataset = SimpleDepthDataset(args.data_root, img_size=args.img_size)
    n_total = len(full_dataset)
    n_val = int(args.val_split * n_total)
    n_train = n_total - n_val

    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"[*] Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

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
    
    args = parser.parse_args()
    main(args)