import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop


class CILDepth(Dataset):
    def __init__(self, data_dir, mode, size=(518, 518), val_fraction=0.05, seed=42):
        self.mode = mode
        self.size = size

        all_rgb = sorted(
            f for f in os.listdir(data_dir) if f.endswith('_rgb.png')
        )

        rng = np.random.default_rng(seed)
        indices = np.arange(len(all_rgb))
        rng.shuffle(indices)
        n_val = max(1, int(len(all_rgb) * val_fraction))

        if mode == 'train':
            chosen = indices[n_val:]
        else:
            chosen = indices[:n_val]

        self.samples = [
            (
                os.path.join(data_dir, all_rgb[i]),
                os.path.join(data_dir, all_rgb[i].replace('_rgb.png', '_depth.npy')),
            )
            for i in sorted(chosen)
        ]

        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([Crop(size[0])] if mode == 'train' else []))

    def __getitem__(self, idx):
        img_path, depth_path = self.samples[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        depth = np.load(depth_path).astype(np.float32)

        sample = self.transform({'image': image, 'depth': depth})

        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])
        sample['valid_mask'] = (sample['depth'] > 0)

        return sample

    def __len__(self):
        return len(self.samples)
