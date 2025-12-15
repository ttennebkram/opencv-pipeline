#!/usr/bin/env python3
"""
1/8 scale experiment: 240x135 input
Goal: See if corners are learnable at low res, compare convergence to full-res AWS run.
"""
import argparse
import json
import os
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class CornerCNN_Eighth(nn.Module):
    """
    CNN for 240x135 input → 8 corner coordinates.

    240/2/2/2/2 = 15, 135/2/2/2/2 = 8.4 → use AdaptiveAvgPool
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # 240x135 → 120x67
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 120x67 → 60x33
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 60x33 → 30x16
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 30x16 → 15x8
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Linear(256, 8)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)


class Dataset_Eighth(Dataset):
    """Load images, resize to 240x135."""

    def __init__(self, data_dir, max_samples=None):
        self.samples = []
        data_path = Path(data_dir)

        # Try worker directories first, then flat structure
        worker_dirs = list(data_path.glob("worker_*"))
        if worker_dirs:
            for worker_dir in sorted(worker_dirs):
                images_dir = worker_dir / "images"
                labels_dir = worker_dir / "labels"
                if not images_dir.exists():
                    continue
                for img_path in sorted(images_dir.glob("*.jpg")):
                    label_path = labels_dir / (img_path.stem + ".json")
                    if label_path.exists():
                        self.samples.append((img_path, label_path))
                        if max_samples and len(self.samples) >= max_samples:
                            return
        else:
            # Flat structure
            images_dir = data_path / "images"
            labels_dir = data_path / "labels"
            if images_dir.exists():
                for img_path in sorted(images_dir.glob("*.jpg")):
                    label_path = labels_dir / (img_path.stem + ".json")
                    if label_path.exists():
                        self.samples.append((img_path, label_path))
                        if max_samples and len(self.samples) >= max_samples:
                            return

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        # Load, convert to grayscale, resize to 240x135
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (240, 135), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)

        with open(label_path) as f:
            label = json.load(f)
        corners = torch.tensor(label["corners"], dtype=torch.float32)

        return img, corners


def pixel_error(pred, true):
    """Compute mean corner pixel error (in 1920x1080 space)."""
    pred = pred.copy()
    true = true.copy()
    # Convert from [-1,1] to pixel coords
    pred[:, 0::2] = (pred[:, 0::2] + 1) * 960  # x
    pred[:, 1::2] = (pred[:, 1::2] + 1) * 540  # y
    true[:, 0::2] = (true[:, 0::2] + 1) * 960
    true[:, 1::2] = (true[:, 1::2] + 1) * 540
    # Per-corner distance, then mean
    diff = (pred - true).reshape(-1, 4, 2)
    dist = np.sqrt((diff ** 2).sum(axis=2))
    return dist.mean()


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_err = 0
    n = 0

    for imgs, corners in loader:
        imgs, corners = imgs.to(device), corners.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, corners)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        total_err += pixel_error(out.detach().cpu().numpy(), corners.cpu().numpy()) * imgs.size(0)
        n += imgs.size(0)

    return total_loss / n, total_err / n


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_err = 0
    n = 0

    with torch.no_grad():
        for imgs, corners in loader:
            imgs, corners = imgs.to(device), corners.to(device)
            out = model(imgs)
            loss = criterion(out, corners)
            total_loss += loss.item() * imgs.size(0)
            total_err += pixel_error(out.cpu().numpy(), corners.cpu().numpy()) * imgs.size(0)
            n += imgs.size(0)

    return total_loss / n, total_err / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/Volumes/SamsungBlue/ml-training/wiki_training')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--max_samples', type=int, default=100000)
    parser.add_argument('--output', default='eighth_model.pth')
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Data
    print(f"Loading data (max {args.max_samples} samples, 240x135)...")
    dataset = Dataset_Eighth(args.data, max_samples=args.max_samples)
    print(f"Found {len(dataset)} samples")

    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Train: {n_train}, Val: {n_val}")

    # Model
    model = CornerCNN_Eighth().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    best_err = float('inf')

    print(f"\n{'Epoch':>5} | {'Train':>10} | {'Val':>10} | {'Time':>6} | Note")
    print("-" * 55)

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_err = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_err = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        note = ""
        if val_err < best_err:
            best_err = val_err
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'val_err': val_err}, args.output)
            note = f"* best"

        print(f"{epoch+1:5d} | {train_err:8.1f}px | {val_err:8.1f}px | {elapsed:5.1f}s | {note}")

    print(f"\nBest val error: {best_err:.1f}px")


if __name__ == "__main__":
    main()
