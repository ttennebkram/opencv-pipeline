#!/usr/bin/env python3
"""ResNet + BatchNorm architecture - PROVEN TO WORK on overfit test."""
import argparse
import json
import os
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ResNetCorners(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = ResBlock(1, 32, stride=2)
        self.layer2 = ResBlock(32, 64, stride=2)
        self.layer3 = ResBlock(64, 128, stride=2)
        self.layer4 = ResBlock(128, 256, stride=2)
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


class Dataset240(Dataset):
    def __init__(self, data_dir, max_samples=None, augment=False):
        self.samples = []
        self.augment = augment
        data_path = Path(data_dir)
        worker_dirs = list(data_path.glob("worker_*"))
        if worker_dirs:
            for wd in sorted(worker_dirs):
                imgs = wd / "images"
                lbls = wd / "labels"
                if not imgs.exists(): continue
                for ip in sorted(imgs.glob("*.jpg")):
                    lp = lbls / (ip.stem + ".json")
                    if lp.exists():
                        self.samples.append((ip, lp))
                        if max_samples and len(self.samples) >= max_samples:
                            return
        else:
            imgs = data_path / "images"
            lbls = data_path / "labels"
            if imgs.exists():
                for ip in sorted(imgs.glob("*.jpg")):
                    lp = lbls / (ip.stem + ".json")
                    if lp.exists():
                        self.samples.append((ip, lp))
                        if max_samples and len(self.samples) >= max_samples:
                            return

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ip, lp = self.samples[idx]
        img = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (240, 135), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        with open(lp) as f:
            lbl = json.load(f)
        corners = np.array(lbl["corners"], dtype=np.float32).reshape(4, 2)

        # Apply augmentations if enabled
        if self.augment:
            # Horizontal flip (50% chance)
            if np.random.random() > 0.5:
                img = np.fliplr(img).copy()
                corners = corners[[1, 0, 3, 2], :]  # Swap TL↔TR, BL↔BR
                corners[:, 0] = -corners[:, 0]  # Negate x

            # Vertical flip (50% chance)
            if np.random.random() > 0.5:
                img = np.flipud(img).copy()
                corners = corners[[3, 2, 1, 0], :]  # Swap TL↔BL, TR↔BR
                corners[:, 1] = -corners[:, 1]  # Negate y

            # Brightness ±15%
            brightness = 1.0 + np.random.uniform(-0.15, 0.15)
            img = np.clip(img * brightness, 0, 1)

            # Contrast ±15%
            contrast = 1.0 + np.random.uniform(-0.15, 0.15)
            mean = img.mean()
            img = np.clip((img - mean) * contrast + mean, 0, 1)

        img = torch.from_numpy(img).unsqueeze(0)
        corners = torch.tensor(corners.flatten(), dtype=torch.float32)
        return img, corners


def pixel_error(pred, true):
    pred, true = pred.copy(), true.copy()
    pred[:, 0::2] = (pred[:, 0::2] + 1) * 960
    pred[:, 1::2] = (pred[:, 1::2] + 1) * 540
    true[:, 0::2] = (true[:, 0::2] + 1) * 960
    true[:, 1::2] = (true[:, 1::2] + 1) * 540
    diff = (pred - true).reshape(-1, 4, 2)
    return np.sqrt((diff ** 2).sum(axis=2)).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="wiki_training_v2")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_samples", type=int, default=120000)
    parser.add_argument("--output", default="resnet_corners.pth")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}", flush=True)

    print(f"Loading data (max {args.max_samples}, 240x135)...", flush=True)

    # Load all samples first to get indices
    ds_all = Dataset240(args.data, max_samples=args.max_samples, augment=False)
    print(f"Found {len(ds_all)} samples", flush=True)

    n_val = int(len(ds_all) * 0.1)
    n_train = len(ds_all) - n_val

    # Split indices
    indices = list(range(len(ds_all)))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Create separate datasets for train (with augment) and val (no augment)
    train_ds = Dataset240(args.data, max_samples=args.max_samples, augment=args.augment)
    val_ds = Dataset240(args.data, max_samples=args.max_samples, augment=False)

    train_ds.samples = [ds_all.samples[i] for i in train_indices]
    val_ds.samples = [ds_all.samples[i] for i in val_indices]

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Augment: {args.augment}", flush=True)

    model = ResNetCorners().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    start_epoch, best_err = 0, float("inf")
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_err = ckpt.get("best_err", float("inf"))
        print(f"Resumed from epoch {start_epoch}, best={best_err:.1f}px", flush=True)

    print(flush=True)
    print("  Ep |     Train |       Val |  Time | Note", flush=True)
    print("-" * 50, flush=True)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()
        model.train()
        train_err = 0
        n = 0
        for imgs, corners in train_ld:
            imgs, corners = imgs.to(device), corners.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, corners)
            loss.backward()
            optimizer.step()
            train_err += pixel_error(out.detach().cpu().numpy(), corners.cpu().numpy()) * imgs.size(0)
            n += imgs.size(0)
        train_err /= n

        model.eval()
        val_err = 0
        n = 0
        with torch.no_grad():
            for imgs, corners in val_ld:
                imgs, corners = imgs.to(device), corners.to(device)
                out = model(imgs)
                val_err += pixel_error(out.cpu().numpy(), corners.cpu().numpy()) * imgs.size(0)
                n += imgs.size(0)
        val_err /= n
        scheduler.step(val_err)

        note = ""
        if val_err < best_err:
            best_err = val_err
            note = "* best"

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch, "best_err": best_err
        }, args.output)

        print(f"{epoch+1:4d} | {train_err:7.1f}px | {val_err:7.1f}px | {time.time()-t0:4.0f}s | {note}", flush=True)

    print(f"\nBest: {best_err:.1f}px", flush=True)


if __name__ == "__main__":
    main()
