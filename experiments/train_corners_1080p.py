#!/usr/bin/env python3
"""
Train a CNN to predict document corners from 1920x1080 images.

Input: 1920x1080 grayscale image
Output: 8 values (4 corners x 2 coordinates), normalized to [-1, 1]

Architecture:
    Conv 7x7, 32, stride 2  → 960x540
    Conv 3x3, 64, stride 2  → 480x270
    Conv 3x3, 128, stride 2 → 240x135
    Conv 3x3, 256, stride 2 → 120x67
    Conv 3x3, 512, stride 2 → 60x33
    GlobalAveragePool       → 512
    Dense                   → 8

Total: ~1.58M parameters

Usage:
    python3 train_corners_1080p.py --data /path/to/wiki_training --epochs 50
    python3 train_corners_1080p.py --data /path/to/wiki_training --epochs 50 --resume checkpoint.pth
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class CornerCNN1080p(nn.Module):
    """
    CNN for 1920x1080 input → 8 corner coordinates output.

    Each conv layer with stride 2 halves the spatial dimensions.
    After 5 such layers: 1920→960→480→240→120→60, 1080→540→270→135→67→33
    GlobalAveragePool reduces 60x33x512 → 512
    Final dense layer: 512 → 8
    """

    def __init__(self):
        super().__init__()

        # Convolutional layers with stride 2 for downsampling
        self.features = nn.Sequential(
            # Layer 1: 1920x1080x1 → 960x540x32
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Layer 2: 960x540x32 → 480x270x64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Layer 3: 480x270x64 → 240x135x128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Layer 4: 240x135x128 → 120x67x256
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Layer 5: 120x67x256 → 60x33x512
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Global average pooling: 60x33x512 → 512
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Final regression layer: 512 → 8 corner coordinates
        self.regressor = nn.Linear(512, 8)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten: (batch, 512, 1, 1) → (batch, 512)
        x = self.regressor(x)
        return x


class WikiTrainingDataset(Dataset):
    """
    Dataset for wiki training images with corner labels.

    Expects directory structure:
        data_dir/
            worker_0/
                images/sample_000000_straight.jpg
                labels/sample_000000_straight.json
            worker_1/
                ...

    Or flat structure:
        data_dir/
            images/sample_000000_straight.jpg
            labels/sample_000000_straight.json
    """

    def __init__(self, data_dir, augment=False):
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.samples = []

        # Find all worker directories or use flat structure
        worker_dirs = list(self.data_dir.glob("worker_*"))
        if worker_dirs:
            for worker_dir in worker_dirs:
                self._add_samples_from_dir(worker_dir)
        else:
            self._add_samples_from_dir(self.data_dir)

        print(f"Found {len(self.samples)} samples", flush=True)

    def _add_samples_from_dir(self, base_dir):
        """Add samples from a directory with images/ and labels/ subdirs."""
        images_dir = base_dir / "images"
        labels_dir = base_dir / "labels"

        if not images_dir.exists():
            return

        for img_path in images_dir.glob("*.jpg"):
            # Find corresponding label file
            label_name = img_path.stem + ".json"
            label_path = labels_dir / label_name

            if label_path.exists():
                self.samples.append((img_path, label_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        # Load image as grayscale
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Return a different sample if this one fails (file may be incomplete)
            return self.__getitem__((idx + 1) % len(self))

        # Load label
        with open(label_path, 'r') as f:
            label_data = json.load(f)

        corners = np.array(label_data['corners'], dtype=np.float32)

        # Data augmentation
        if self.augment:
            img, corners = self._augment(img, corners)

        # Normalize image to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Add channel dimension: (H, W) → (1, H, W)
        img = np.expand_dims(img, axis=0)

        return torch.from_numpy(img), torch.from_numpy(corners)

    def _augment(self, img, corners):
        """Apply data augmentation."""
        h, w = img.shape

        # Random horizontal flip (50% chance)
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            # Flip corner x-coordinates: x' = -x (since normalized to [-1,1])
            corners = corners.copy()
            corners[0::2] = -corners[0::2]  # Flip x coordinates
            # Swap left and right corners: [TL, TR, BR, BL] → [TR, TL, BL, BR]
            corners = np.array([
                corners[2], corners[3],  # TR becomes TL
                corners[0], corners[1],  # TL becomes TR
                corners[6], corners[7],  # BL becomes BR
                corners[4], corners[5],  # BR becomes BL
            ], dtype=np.float32)

        # Random brightness adjustment (±10%)
        if random.random() < 0.5:
            factor = random.uniform(0.9, 1.1)
            img = np.clip(img * factor, 0, 255).astype(np.uint8)

        # Random contrast adjustment (±10%)
        if random.random() < 0.5:
            factor = random.uniform(0.9, 1.1)
            mean = img.mean()
            img = np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)

        return img, corners


def compute_pixel_error(pred, target, width=1920, height=1080):
    """
    Compute mean pixel error for corners.

    Args:
        pred: Predicted corners, normalized [-1, 1], shape (batch, 8)
        target: Target corners, normalized [-1, 1], shape (batch, 8)
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Mean pixel distance error across all corners
    """
    # Convert from normalized to pixel coordinates
    pred_pixels = pred.clone()
    target_pixels = target.clone()

    # x coordinates (indices 0, 2, 4, 6)
    pred_pixels[:, 0::2] = (pred[:, 0::2] + 1) / 2 * width
    target_pixels[:, 0::2] = (target[:, 0::2] + 1) / 2 * width

    # y coordinates (indices 1, 3, 5, 7)
    pred_pixels[:, 1::2] = (pred[:, 1::2] + 1) / 2 * height
    target_pixels[:, 1::2] = (target[:, 1::2] + 1) / 2 * height

    # Compute Euclidean distance for each corner
    diff = pred_pixels - target_pixels
    diff = diff.view(-1, 4, 2)  # (batch, 4 corners, 2 coords)
    distances = torch.sqrt((diff ** 2).sum(dim=2))  # (batch, 4)

    return distances.mean()


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_pixel_error = 0
    num_batches = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_pixel_error += compute_pixel_error(outputs.detach(), targets).item()
        num_batches += 1

    return total_loss / num_batches, total_pixel_error / num_batches


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_pixel_error = 0
    num_batches = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            total_pixel_error += compute_pixel_error(outputs, targets).item()
            num_batches += 1

    return total_loss / num_batches, total_pixel_error / num_batches


def main():
    parser = argparse.ArgumentParser(description='Train 1080p corner detection CNN')
    parser.add_argument('--data', type=str, required=True, help='Path to training data')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--output', type=str, default='corners_1080p.pth', help='Output model path')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    args = parser.parse_args()

    # Device selection
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}", flush=True)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Apple Silicon)", flush=True)
    else:
        device = torch.device('cpu')
        print("Using CPU", flush=True)

    # Load dataset
    print(f"Loading data from {args.data}...", flush=True)
    full_dataset = WikiTrainingDataset(args.data, augment=args.augment)

    # Split into train/val
    num_val = int(len(full_dataset) * args.val_split)
    num_train = len(full_dataset) - num_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [num_train, num_val],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Training samples: {num_train}", flush=True)
    print(f"Validation samples: {num_val}", flush=True)

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Model
    model = CornerCNN1080p().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_error = float('inf')

    if args.resume:
        print(f"Resuming from {args.resume}...", flush=True)
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_error = checkpoint.get('best_val_error', float('inf'))
        print(f"Resumed at epoch {start_epoch}, best val error: {best_val_error:.2f}px", flush=True)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...", flush=True)
    print("-" * 70, flush=True)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_error = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_error = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start

        # Print progress
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.6f} | Train Err: {train_error:6.2f}px | "
              f"Val Loss: {val_loss:.6f} | Val Err: {val_error:6.2f}px | "
              f"Time: {epoch_time:.1f}s", flush=True)

        # Save best model
        if val_error < best_val_error:
            best_val_error = val_error
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_error': best_val_error,
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, args.output)
            print(f"  → Saved best model: {val_error:.2f}px", flush=True)

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = args.output.replace('.pth', f'_checkpoint_ep{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_error': best_val_error,
            }, checkpoint_path)
            print(f"  → Saved checkpoint: {checkpoint_path}", flush=True)

    print("-" * 70, flush=True)
    print(f"Training complete. Best validation error: {best_val_error:.2f}px", flush=True)
    print(f"Model saved to: {args.output}", flush=True)


if __name__ == "__main__":
    main()
