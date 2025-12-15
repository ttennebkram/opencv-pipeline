#!/usr/bin/env python3
"""
Overfit test with skip connections added to our CNN.
"""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class ResBlock(nn.Module):
    """Residual block with skip connection."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1)

        # Skip connection needs to match dimensions
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=stride)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = F.relu(x + residual)  # Skip connection!
        return x


class ResNetCorners(nn.Module):
    """Our CNN with skip connections."""
    def __init__(self):
        super().__init__()
        # Same channel progression as before: 1 -> 32 -> 64 -> 128 -> 256
        # But now with skip connections
        self.layer1 = ResBlock(1, 32, stride=2)    # 240x135 -> 120x67
        self.layer2 = ResBlock(32, 64, stride=2)   # -> 60x33
        self.layer3 = ResBlock(64, 128, stride=2)  # -> 30x16
        self.layer4 = ResBlock(128, 256, stride=2) # -> 15x8

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Linear(256, 8)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)


class TinyDataset(Dataset):
    def __init__(self, data_dir):
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

    dataset = TinyDataset("/tmp/tiny_overfit")
    print(f"Samples: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

    model = ResNetCorners().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} (ResNet-style CNN)")
    print()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"{'Epoch':>5} | {'Loss':>12} | {'Pixel Err':>10} | Output Range")
    print("-" * 60)

    best_err = float('inf')
    for epoch in range(500):
        model.train()
        for imgs, corners in loader:
            imgs, corners = imgs.to(device), corners.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, corners)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for imgs, corners in loader:
                imgs, corners = imgs.to(device), corners.to(device)
                out = model(imgs)
                loss = criterion(out, corners)
                err = pixel_error(out.cpu().numpy(), corners.cpu().numpy())
                out_range = f"[{out.min().item():.2f}, {out.max().item():.2f}]"

        if err < best_err:
            best_err = err

        if epoch % 25 == 0 or epoch < 10 or err < 20:
            print(f"{epoch:5d} | {loss.item():12.6f} | {err:8.1f}px | {out_range}")

        if err < 5.0:
            print(f"\n*** SUCCESS at epoch {epoch} ***")
            break

    print(f"\nBest error: {best_err:.1f}px")

    # Show predictions
    print("\n--- Sample Predictions ---")
    model.eval()
    with torch.no_grad():
        for imgs, corners in loader:
            imgs, corners = imgs.to(device), corners.to(device)
            out = model(imgs)
            for i in range(min(3, len(out))):
                gt = corners[i].cpu().numpy()
                pr = out[i].cpu().numpy()
                err_i = pixel_error(pr.reshape(1,-1), gt.reshape(1,-1))
                print(f"Sample {i}: {err_i:.1f}px")
                print(f"  GT:   {gt.round(2)}")
                print(f"  Pred: {pr.round(2)}")


if __name__ == "__main__":
    main()
