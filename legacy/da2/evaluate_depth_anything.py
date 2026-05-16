"""
Evaluate Depth-Anything-V2-Base on a fraction of CIL training data.

- infer_image() returns relative DISPARITY (high=close); GT is DEPTH (high=far).
- Invert disparity → depth, clip to [0.001, 80], then compute siRMSE.
- siRMSE = sqrt( mean(d²) - mean(d)² )  where d = log(pred) - log(gt).
- Optional: save side-by-side visualisations (RGB | GT depth | Pred depth).
"""

import argparse
import sys
import os
import numpy as np
import cv2
import matplotlib
import matplotlib.cm as cm
import torch
from tqdm import tqdm

sys.path.insert(0, '/work/courses/3dv/team24/Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2

DATA_DIR   = '/cluster/courses/cil/monocular-depth-estimation/train'
CKPT_PATH  = '/home/heelee/checkpoints/depth_anything_v2_vitb.pth'
MODEL_CFG  = {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]}
INPUT_SIZE = 518
MIN_DEPTH  = 0.001
MAX_DEPTH  = 80.0

CMAP = matplotlib.colormaps.get_cmap('Spectral_r')


def depth_to_color(depth: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Map a depth array to a uint8 BGR image using Spectral_r colormap.
    Invalid pixels (valid_mask=False) are shown in dark gray."""
    out = np.full((*depth.shape, 3), 40, dtype=np.uint8)  # dark gray background
    if valid_mask.sum() == 0:
        return out
    vmin = depth[valid_mask].min()
    vmax = depth[valid_mask].max()
    norm = np.zeros_like(depth)
    norm[valid_mask] = (depth[valid_mask] - vmin) / max(vmax - vmin, 1e-6)
    color = (CMAP(norm)[:, :, :3] * 255).astype(np.uint8)[:, :, ::-1]  # RGB→BGR
    out[valid_mask] = color[valid_mask]
    return out


def compute_metrics(pred_disp: np.ndarray, gt: np.ndarray):
    valid = (gt >= MIN_DEPTH) & (gt <= MAX_DEPTH)
    eps   = 1e-6
    pred_depth = np.clip(1.0 / (pred_disp[valid].astype(np.float64) + eps),
                         MIN_DEPTH, MAX_DEPTH)
    g          = gt[valid].astype(np.float64)
    d          = np.log(pred_depth) - np.log(g)
    return float(np.sqrt(np.mean(d ** 2) - np.mean(d) ** 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fraction', type=float, default=0.1)
    parser.add_argument('--seed',     type=int,   default=42)
    parser.add_argument('--output',   type=str,   default='/home/heelee/si_rmse_results.txt')
    parser.add_argument('--vis-dir',  type=str,   default=None,
                        help='Directory to save visualisations. Skipped if not set.')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    model = DepthAnythingV2(**MODEL_CFG)
    model.load_state_dict(torch.load(CKPT_PATH, map_location='cpu'))
    model = model.to(device).eval()

    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)

    all_rgb = sorted(f for f in os.listdir(DATA_DIR) if f.endswith('_rgb.png'))
    total   = len(all_rgb)
    rng     = np.random.default_rng(args.seed)
    indices = rng.choice(total, size=int(total * args.fraction), replace=False)
    indices.sort()
    sample  = [all_rgb[i] for i in indices]
    print(f'Evaluating {len(sample)} / {total} samples ({args.fraction*100:.0f}%)')

    si_rmses = []
    for rgb_name in tqdm(sample, desc='Evaluating'):
        rgb_path = os.path.join(DATA_DIR, rgb_name)
        gt_path  = os.path.join(DATA_DIR, rgb_name.replace('_rgb.png', '_depth.npy'))

        image = cv2.imread(rgb_path)
        if image is None:
            continue
        gt = np.load(gt_path)

        with torch.no_grad():
            pred_disp = model.infer_image(image, INPUT_SIZE)

        if pred_disp.shape != gt.shape:
            pred_disp = cv2.resize(pred_disp, (gt.shape[1], gt.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)

        gt_valid = (gt >= MIN_DEPTH) & (gt <= MAX_DEPTH)
        if gt_valid.sum() < 10:
            continue

        sirmse = compute_metrics(pred_disp, gt)
        si_rmses.append(sirmse)

        if args.vis_dir:
            eps = 1e-6
            pred_depth     = np.clip(1.0 / (pred_disp.astype(np.float64) + eps),
                                     MIN_DEPTH, MAX_DEPTH).astype(np.float32)
            pred_valid     = np.ones(pred_depth.shape, dtype=bool)

            gt_vis   = depth_to_color(gt,         gt_valid)
            pred_vis = depth_to_color(pred_depth, pred_valid)

            # Resize RGB to match depth map size if needed
            if image.shape[:2] != gt.shape:
                image = cv2.resize(image, (gt.shape[1], gt.shape[0]))

            # Add text labels
            def label(img, text):
                out = img.copy()
                cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 0), 1, cv2.LINE_AA)
                return out

            strip = np.ones((gt.shape[0], 8, 3), dtype=np.uint8) * 200
            combined = np.hstack([
                label(image,    'RGB'),
                strip,
                label(gt_vis,   f'GT depth  [siRMSE={sirmse:.3f}]'),
                strip,
                label(pred_vis, 'Pred depth (DA-V2 zero-shot)'),
            ])

            stem = rgb_name.replace('_rgb.png', '')
            cv2.imwrite(os.path.join(args.vis_dir, f'{stem}_vis.jpg'), combined,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])

    si_rmses = np.array(si_rmses)
    summary = (
        f'Samples evaluated : {len(si_rmses)}\n'
        f'\n'
        f'siRMSE : mean={si_rmses.mean():.4f}  median={np.median(si_rmses):.4f}'
        f'  std={si_rmses.std():.4f}\n'
    )
    print('\n' + summary)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(summary)
    print(f'Results saved to {args.output}')
    if args.vis_dir:
        print(f'Visualisations saved to {args.vis_dir}')


if __name__ == '__main__':
    main()
