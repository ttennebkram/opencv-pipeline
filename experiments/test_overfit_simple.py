#!/usr/bin/env python3
"""
Simplest possible overfit test - just a few FC layers.
If this can't overfit, the labels are wrong.
"""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class SimpleFC(nn.Module):
    """Just flatten and FC layers - simplest possible."""
    def __init__(self, input_size=240*135):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 8),
        )
        # Initialize output layer to output in [-1, 1] range
        nn.init.xavier_uniform_(self.fc[-1].weight, gain=0.1)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


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

    model = SimpleFC().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} (simple FC)")
    print()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"{'Epoch':>5} | {'Loss':>12} | {'Pixel Err':>10} | Output Range")
    print("-" * 60)

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

        if epoch % 25 == 0 or epoch < 10 or err < 20:
            print(f"{epoch:5d} | {loss.item():12.6f} | {err:8.1f}px | {out_range}")

        if err < 5.0:
            print(f"\n*** SUCCESS at epoch {epoch} ***")
            break

    # Show sample predictions
    print("\n--- Sample Predictions vs Ground Truth ---")
    model.eval()
    with torch.no_grad():
        for imgs, corners in loader:
            imgs, corners = imgs.to(device), corners.to(device)
            out = model(imgs)
            for i in range(min(3, len(out))):
                print(f"Sample {i}:")
                print(f"  GT:   {corners[i].cpu().numpy().round(2)}")
                print(f"  Pred: {out[i].cpu().numpy().round(2)}")


if __name__ == "__main__":
    main()
