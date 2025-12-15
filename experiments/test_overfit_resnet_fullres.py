#!/usr/bin/env python3
"""
Full resolution (1920x1080) ResNet+BN overfit test.
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


class ResNetFullRes(nn.Module):
    """ResNet for 1920x1080 input."""
    def __init__(self):
        super().__init__()
        # More layers needed for full res
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=2, padding=3, bias=False),  # 960x540
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResBlock(32, 64, stride=2)    # 480x270
        self.layer2 = ResBlock(64, 128, stride=2)   # 240x135
        self.layer3 = ResBlock(128, 256, stride=2)  # 120x67
        self.layer4 = ResBlock(256, 512, stride=2)  # 60x33
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Linear(512, 8)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)


class TinyDataset(Dataset):
    def __init__(self, data_dir, full_res=False):
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


def test_resolution(name, model, dataset, device, epochs=300):
    print(f"\n{'='*50}", flush=True)
    print(f"Testing: {name}", flush=True)
    print(f"{'='*50}", flush=True)

    loader = DataLoader(dataset, batch_size=min(10, len(dataset)), shuffle=True)

    model = model.to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Samples: {len(dataset)}, Parameters: {params:,}", flush=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"\n{'Epoch':>5} | {'Loss':>12} | {'Pixel Err':>10}", flush=True)
    print("-" * 40, flush=True)

    best_err = float('inf')
    for epoch in range(epochs):
        model.train()
        for imgs, corners in loader:
            imgs, corners = imgs.to(device), corners.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, corners)
            loss.backward()
            optimizer.step()

        model.eval()
        total_err = 0
        n = 0
        with torch.no_grad():
            for imgs, corners in loader:
                imgs, corners = imgs.to(device), corners.to(device)
                out = model(imgs)
                loss = criterion(out, corners)
                total_err += pixel_error(out.cpu().numpy(), corners.cpu().numpy()) * imgs.size(0)
                n += imgs.size(0)
        err = total_err / n

        if err < best_err:
            best_err = err

        if epoch % 25 == 0 or epoch < 10 or err < 10:
            print(f"{epoch:5d} | {loss.item():12.6f} | {err:8.1f}px", flush=True)

        if err < 5.0:
            print(f"\n*** SUCCESS at epoch {epoch} ***", flush=True)
            break

    print(f"\nBest error: {best_err:.1f}px", flush=True)
    return best_err


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Test 1/8th resolution (240x135)
    from test_overfit_resnet_bn import ResNetCorners
    ds_eighth = TinyDataset("/tmp/tiny_overfit", full_res=False)
    model_eighth = ResNetCorners()
    err_eighth = test_resolution("1/8th Resolution (240x135)", model_eighth, ds_eighth, device, epochs=200)

    # Test full resolution (1920x1080)
    ds_full = TinyDataset("/tmp/tiny_overfit", full_res=True)
    model_full = ResNetFullRes()
    err_full = test_resolution("Full Resolution (1920x1080)", model_full, ds_full, device, epochs=200)

    print(f"\n{'='*50}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"1/8th (240x135):  {err_eighth:.1f}px", flush=True)
    print(f"Full (1920x1080): {err_full:.1f}px", flush=True)


if __name__ == "__main__":
    main()
