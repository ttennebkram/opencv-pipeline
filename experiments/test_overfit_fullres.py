#!/usr/bin/env python3
"""
Full resolution overfit test: 1920x1080 grayscale.
If this can't overfit, the task itself is problematic.
"""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class FullResCNN(nn.Module):
    """CNN for 1920x1080 input -> 8 corners."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # 1920x1080 -> 960x540
            nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            # 960x540 -> 480x270
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 480x270 -> 240x135
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 240x135 -> 120x67
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 120x67 -> 60x33
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 60x33 -> 30x16
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Linear(512, 8)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)


class TinyDataset(Dataset):
    def __init__(self, data_dir, full_res=True):
        self.full_res = full_res
        self.samples = []
        data_path = Path(data_dir)
        for img_path in sorted((data_path / "images").glob("*.jpg")):
            label_path = data_path / "labels" / (img_path.stem + ".json")
            if label_path.exists():
                self.samples.append((img_path, label_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if not self.full_res:
            img = cv2.resize(img, (240, 135), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)
        with open(label_path) as f:
            label = json.load(f)
        corners = torch.tensor(label["corners"], dtype=torch.float32)
        return img, corners


def pixel_error(pred, true):
    pred = pred.copy()
    true = true.copy()
    pred[:, 0::2] = (pred[:, 0::2] + 1) * 960
    pred[:, 1::2] = (pred[:, 1::2] + 1) * 540
    true[:, 0::2] = (true[:, 0::2] + 1) * 960
    true[:, 1::2] = (true[:, 1::2] + 1) * 540
    diff = (pred - true).reshape(-1, 4, 2)
    dist = np.sqrt((diff ** 2).sum(axis=2))
    return dist.mean()


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = TinyDataset("/tmp/tiny_overfit", full_res=True)
    print(f"Samples: {len(dataset)}")
    print(f"Image size: 1920x1080 (full resolution)")

    # Smaller batch size due to memory
    loader = DataLoader(dataset, batch_size=10, shuffle=True)

    model = FullResCNN().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    print()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"{'Epoch':>5} | {'Loss':>12} | {'Pixel Err':>10}")
    print("-" * 35)

    best_err = float('inf')
    for epoch in range(300):
        model.train()
        total_loss = 0
        n = 0
        for imgs, corners in loader:
            imgs, corners = imgs.to(device), corners.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, corners)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)

        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for imgs, corners in loader:
                imgs, corners = imgs.to(device), corners.to(device)
                out = model(imgs)
                all_pred.append(out.cpu().numpy())
                all_true.append(corners.cpu().numpy())

        pred = np.vstack(all_pred)
        true = np.vstack(all_true)
        err = pixel_error(pred, true)
        avg_loss = total_loss / n

        if err < best_err:
            best_err = err

        if epoch % 20 == 0 or epoch < 10 or err < 20:
            print(f"{epoch:5d} | {avg_loss:12.6f} | {err:8.1f}px")

        if err < 5.0:
            print(f"\n*** SUCCESS at epoch {epoch} ***")
            break

    print(f"\nBest error: {best_err:.1f}px")


if __name__ == "__main__":
    main()
