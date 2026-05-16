# analyze_depth_dataset.py
from pathlib import Path
import numpy as np

ROOT = Path("/cluster/courses/cil/monocular-depth-estimation/train")
MAX_FILES = None  # 빠르게 보려면 100 같은 숫자로 바꾸기

files = sorted(ROOT.glob("*_depth.npy"))
if MAX_FILES is not None:
    files = files[:MAX_FILES]

if not files:
    raise FileNotFoundError(f"No *_depth.npy files found in {ROOT}")

shapes = {}
dtypes = {}
total_pixels = 0
valid_pixels = 0
zero_pixels = 0
neg_pixels = 0
nan_pixels = 0
inf_pixels = 0

per_image_zero_ratio = []
per_image_valid_quantiles = []
global_min = float("inf")
global_max = float("-inf")
weighted_sum = 0.0

quantile_points = [0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0]
quantile_names = ["min", "p0.1", "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "p99.9", "max"]

print(f"Analyzing {len(files)} depth files from {ROOT}")

for i, path in enumerate(files):
    d = np.load(path)

    shapes[d.shape] = shapes.get(d.shape, 0) + 1
    dtypes[str(d.dtype)] = dtypes.get(str(d.dtype), 0) + 1

    finite = np.isfinite(d)
    valid = finite & (d > 0)
    zero = finite & (d == 0)
    neg = finite & (d < 0)

    total_pixels += d.size
    valid_pixels += int(valid.sum())
    zero_pixels += int(zero.sum())
    neg_pixels += int(neg.sum())
    nan_pixels += int(np.isnan(d).sum())
    inf_pixels += int(np.isinf(d).sum())

    per_image_zero_ratio.append(float(zero.mean()))

    if valid.any():
        v = d[valid].astype(np.float64)
        global_min = min(global_min, float(v.min()))
        global_max = max(global_max, float(v.max()))
        weighted_sum += float(v.sum())
        per_image_valid_quantiles.append(np.quantile(v, quantile_points))

    if i < 10:
        if valid.any():
            v = d[valid]
            print(
                f"{path.name}: shape={d.shape}, dtype={d.dtype}, "
                f"zero={zero.mean()*100:.3f}%, "
                f"valid_min={float(v.min()):.6g}, "
                f"valid_p50={float(np.median(v)):.6g}, "
                f"valid_max={float(v.max()):.6g}"
            )
        else:
            print(f"{path.name}: shape={d.shape}, dtype={d.dtype}, no valid depth")

print("\n=== Dataset Summary ===")
print("files:", len(files))
print("shapes:", shapes)
print("dtypes:", dtypes)
print("total_pixels:", total_pixels)
print(f"valid_pixels: {valid_pixels} ({valid_pixels / total_pixels * 100:.3f}%)")
print(f"zero_pixels:  {zero_pixels} ({zero_pixels / total_pixels * 100:.3f}%)")
print(f"neg_pixels:   {neg_pixels} ({neg_pixels / total_pixels * 100:.6f}%)")
print(f"nan_pixels:   {nan_pixels} ({nan_pixels / total_pixels * 100:.6f}%)")
print(f"inf_pixels:   {inf_pixels} ({inf_pixels / total_pixels * 100:.6f}%)")

print("\n=== Per-image zero ratio ===")
print(f"mean:   {np.mean(per_image_zero_ratio) * 100:.3f}%")
print(f"median: {np.median(per_image_zero_ratio) * 100:.3f}%")
print(f"max:    {np.max(per_image_zero_ratio) * 100:.3f}%")

if valid_pixels > 0:
    print("\n=== Valid depth values ===")
    print("global_min:", global_min)
    print("global_max:", global_max)
    print("weighted_mean:", weighted_sum / valid_pixels)

    q = np.vstack(per_image_valid_quantiles)
    median_q = np.median(q, axis=0)

    print("\nMedian of per-image quantiles:")
    for name, value in zip(quantile_names, median_q):
        print(f"{name:>6}: {value:.6g}")

print("\n=== Interpretation hints ===")
print("- If values are mostly in roughly [0.001, 80], this is metric depth in meters.")
print("- If nearer objects have larger values and values look like 1/depth or disparity, then it may be inverse depth/disparity.")
print("- Project spec says training .npy maps are pixel-wise depth measurements in meters, with scale not meaningful between images.")
print("- depth == 0 is being treated by dataset.py as invalid/missing label.")