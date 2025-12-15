#!/usr/bin/env python3
"""
Tiny overfit test: Can the model memorize 50 samples?
If not, architecture/data/labels have a fundamental problem.
"""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class CornerCNN_Eighth(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Linear(256, 8)

    def forward(self, x):
        x = self.features(x)
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

    # Use ALL data for training (overfit test)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

    model = CornerCNN_Eighth().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    print()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # High LR for fast overfit

    print(f"{'Epoch':>5} | {'Loss':>12} | {'Pixel Err':>10}")
    print("-" * 35)

    for epoch in range(200):
        model.train()
        for imgs, corners in loader:
            imgs, corners = imgs.to(device), corners.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, corners)
            loss.backward()
            optimizer.step()

        # Evaluate on same data
        model.eval()
        with torch.no_grad():
            for imgs, corners in loader:
                imgs, corners = imgs.to(device), corners.to(device)
                out = model(imgs)
                loss = criterion(out, corners)
                err = pixel_error(out.cpu().numpy(), corners.cpu().numpy())

        if epoch % 10 == 0 or epoch < 10 or err < 10:
            print(f"{epoch:5d} | {loss.item():12.6f} | {err:8.1f}px")

        if err < 1.0:
            print(f"\n*** SUCCESS: Near-zero error at epoch {epoch} ***")
            break

    print()
    if err > 50:
        print("FAIL: Cannot overfit tiny dataset. Check architecture/data/labels.")
    elif err > 10:
        print("PARTIAL: Overfitting slowly. May need more epochs or architecture changes.")
    else:
        print("SUCCESS: Model can learn the task. Scale up data.")


if __name__ == "__main__":
    main()
