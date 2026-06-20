import torch
import os
import numpy as np
from torch.utils.data import DataLoader

class NpzCalcDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.images = data["images"]
        # Labels are not needed just for pixel math

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Load image and scale strictly to [0, 1] range
        img = self.images[idx].astype(np.float32) / 255.0
        
        # 2. Convert to tensor and add the channel dimension (1, 32, 32)
        img = torch.from_numpy(img).unsqueeze(0)
        
        # Return the raw tensor. NO transforms. NO pre-normalization.
        return img

# Use your actual DATA_DIR path
DATA_DIR = '../dades/dataset_npz/'

# Initialize the stripped-down dataset
calc_ds = NpzCalcDataset(os.path.join(DATA_DIR, "train.npz"))

# DataLoader
calc_loader = DataLoader(calc_ds, batch_size=256, shuffle=False, num_workers=2)

pixel_sum = 0.0
pixel_sum_sq = 0.0
num_pixels = 0

print("Calculating exact Mean and Std Dev...")

for imgs in calc_loader:
    # imgs shape: [batch_size, 1, 32, 32]
    pixel_sum += imgs.sum()
    pixel_sum_sq += (imgs ** 2).sum()
    num_pixels += imgs.numel()

mean = pixel_sum / num_pixels
std = torch.sqrt((pixel_sum_sq / num_pixels) - (mean ** 2))

print(f"Calculated Mean: {mean.item():.4f}")
print(f"Calculated Std Dev: {std.item():.4f}")