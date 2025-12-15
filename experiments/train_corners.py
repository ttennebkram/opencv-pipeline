#!/usr/bin/env python3
"""
train_corners.py

Train a neural network to predict normalized corner positions from distorted images.

This is a simpler and more effective approach than predicting homography matrices:
- Output: 8 values (4 corners × 2 coordinates) normalized to [-1, +1]
- No log transform needed - values are naturally bounded
- Homography can be computed from corners at inference time

Usage:
    python3 train_corners.py --data training_data/homography --epochs 50

Prerequisites:
    pip install torch torchvision pillow numpy
"""

import argparse
import os
import sys
import time
import glob
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Unbuffered output for real-time progress
sys.stdout.reconfigure(line_buffering=True)

# Image dimensions for normalization (must match training data generation)
IMG_WIDTH = 1920
IMG_HEIGHT = 1080


class CornersDataset(Dataset):
    """Dataset of distorted images and their normalized corner positions."""

    def __init__(self, data_dir, img_width=128, img_height=72):
        self.data_dir = data_dir
        self.img_width = img_width
        self.img_height = img_height
        self.samples = []

        # Find all label files
        labels_dir = os.path.join(data_dir, "labels")
        images_dir = os.path.join(data_dir, "images")

        if not os.path.exists(labels_dir):
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

        label_files = glob.glob(os.path.join(labels_dir, "*.json"))
        print(f"Found {len(label_files)} label files")

        for label_path in label_files:
            # Get corresponding image path (normal version, not white)
            base_name = os.path.splitext(os.path.basename(label_path))[0]
            image_path = os.path.join(images_dir, base_name + "_normal.jpg")

            if os.path.exists(image_path):
                self.samples.append((image_path, label_path))

        print(f"Loaded {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        # Load image, convert to grayscale, resize
        image = Image.open(img_path).convert("L")  # Grayscale
        image = image.resize((self.img_width, self.img_height), Image.BILINEAR)

        # Convert to tensor and normalize to [0, 1]
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # Add channel dim

        # Load corners from label
        with open(label_path, "r") as f:
            data = json.load(f)

        corners = np.array(data["corners"], dtype=np.float32).reshape(4, 2)

        # Normalize corners to [-1, +1] range
        corners[:, 0] = (corners[:, 0] - IMG_WIDTH / 2) / (IMG_WIDTH / 2)
        corners[:, 1] = (corners[:, 1] - IMG_HEIGHT / 2) / (IMG_HEIGHT / 2)

        target_tensor = torch.from_numpy(corners.flatten())

        return img_tensor, target_tensor


class CornersNet(nn.Module):
    """
    Fully-connected network for corner position estimation.

    Architecture:
    - Input: img_width × img_height grayscale pixels
    - Hidden layers with ReLU activation
    - Output: 8 values (4 corners × 2 coordinates, normalized)
    """

    def __init__(self, input_height=72, input_width=128, hidden_sizes=[512, 256, 64]):
        super(CornersNet, self).__init__()

        self.input_size = input_height * input_width
        self.output_size = 8  # 4 corners × 2 coordinates

        print(f"  Input size: {input_width}x{input_height} = {self.input_size:,} neurons")
        print(f"  Hidden layers: {hidden_sizes}")
        print(f"  Output size: {self.output_size}")

        # Build layers dynamically
        layers = []
        prev_size = self.input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU(inplace=True))
            prev_size = hidden_size

        # Output layer (no activation - regression output)
        layers.append(nn.Linear(prev_size, self.output_size))

        self.fc = nn.Sequential(*layers)

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Total parameters: {total_params:,}")
        print(f"  Memory: ~{total_params * 4 / 1024 / 1024:.1f} MB")

    def forward(self, x):
        # x: (batch, 1, height, width)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten to (batch, pixels)
        return self.fc(x)


def corner_mae(pred, target):
    """
    Compute mean absolute error across all 8 corner values.
    """
    return torch.abs(pred - target).mean()


def corner_pixel_error(pred, target):
    """
    Compute average pixel distance between predicted and target corners.
    Denormalizes back to pixel space for interpretable error.
    """
    # Reshape to (batch, 4, 2)
    pred_corners = pred.view(-1, 4, 2)
    target_corners = target.view(-1, 4, 2)

    # Denormalize
    pred_corners = pred_corners.clone()
    target_corners = target_corners.clone()
    pred_corners[:, :, 0] = pred_corners[:, :, 0] * (IMG_WIDTH / 2) + (IMG_WIDTH / 2)
    pred_corners[:, :, 1] = pred_corners[:, :, 1] * (IMG_HEIGHT / 2) + (IMG_HEIGHT / 2)
    target_corners[:, :, 0] = target_corners[:, :, 0] * (IMG_WIDTH / 2) + (IMG_WIDTH / 2)
    target_corners[:, :, 1] = target_corners[:, :, 1] * (IMG_HEIGHT / 2) + (IMG_HEIGHT / 2)

    # Euclidean distance per corner, then average
    dist = torch.sqrt(((pred_corners - target_corners) ** 2).sum(dim=2))
    return dist.mean()


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_pixel_error = 0.0
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
        total_mae += corner_mae(outputs, targets).item()
        total_pixel_error += corner_pixel_error(outputs, targets).item()
        num_batches += 1

    return (total_loss / num_batches,
            total_mae / num_batches,
            total_pixel_error / num_batches)


def evaluate(model, loader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_pixel_error = 0.0
    num_batches = 0

    criterion = nn.MSELoss()

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)

            loss = criterion(outputs, targets).item()
            mae = corner_mae(outputs, targets).item()
            pixel_error = corner_pixel_error(outputs, targets).item()

            total_loss += loss
            total_mae += mae
            total_pixel_error += pixel_error
            num_batches += 1

    return (total_loss / num_batches,
            total_mae / num_batches,
            total_pixel_error / num_batches)


def main():
    parser = argparse.ArgumentParser(description="Train corner position estimation network")
    parser.add_argument("--data", type=str, default="/Volumes/SamsungBlue/ml-training/homography/training_data",
                        help="Path to training data directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--img_width", type=int, default=128, help="Image width for training")
    parser.add_argument("--img_height", type=int, default=72, help="Image height for training")
    parser.add_argument("--hidden", type=str, default="512,256,64",
                        help="Hidden layer sizes (comma-separated)")
    parser.add_argument("--output", type=str, default="corners_model.pth",
                        help="Output model path")
    parser.add_argument("--weights_output", type=str, default="corners_model.weights",
                        help="Output weights path (for Java inference)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible data shuffling (use same seed across parallel runs for cache efficiency)")
    args = parser.parse_args()

    hidden_sizes = [int(x) for x in args.hidden.split(",")]

    print("=" * 70)
    print("Corner Position Estimation Training")
    print("=" * 70)

    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Device: MPS (Apple Metal) - GPU accelerated")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Image size: {args.img_width}x{args.img_height}")
    print(f"Hidden layers: {hidden_sizes}")
    print(f"Random seed: {args.seed}")
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = CornersDataset(args.data, img_width=args.img_width, img_height=args.img_height)

    if len(dataset) == 0:
        print("ERROR: No samples found!")
        sys.exit(1)

    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    # Use seeded sampler for reproducible batch ordering (enables cache sharing across parallel runs)
    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, persistent_workers=args.workers > 0,
                               generator=train_generator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, persistent_workers=args.workers > 0)

    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    print()

    # Create model
    print("Creating model...")
    model = CornersNet(
        input_height=args.img_height,
        input_width=args.img_width,
        hidden_sizes=hidden_sizes
    )
    model = model.to(device)
    print()

    # Loss and optimizer - Adam works well for this bounded regression
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training loop
    print("Starting training...")
    print("-" * 80)
    print(f"{'Epoch':>5} | {'Train Loss':>12} | {'Val Loss':>12} | {'Val MAE':>10} | {'Pixel Err':>10} | {'Time':>6}")
    print("-" * 80)

    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_mae, train_pixel = train_epoch(model, train_loader, optimizer, criterion, device)

        # Evaluate
        val_loss, val_mae, val_pixel = evaluate(model, val_loader, device)

        # Update scheduler
        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start

        print(f"{epoch+1:5d} | {train_loss:12.6f} | {val_loss:12.6f} | {val_mae:10.4f} | {val_pixel:10.1f}px | {epoch_time:5.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_pixel_error': val_pixel,
                'img_width': args.img_width,
                'img_height': args.img_height,
                'hidden_sizes': hidden_sizes,
            }, args.output)
            print(f"       -> Saved best model (Pixel Error: {val_pixel:.1f}px)")

        # Always save checkpoint for resume capability
        checkpoint_path = args.output.replace('.pth', '_checkpoint.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'val_mae': val_mae,
            'val_pixel_error': val_pixel,
            'best_val_loss': best_val_loss,
            'img_width': args.img_width,
            'img_height': args.img_height,
            'hidden_sizes': hidden_sizes,
        }, checkpoint_path)

    total_time = time.time() - start_time
    print("-" * 80)
    print(f"Training complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {args.output}")

    # Export weights for Java inference
    print(f"\nExporting weights for Java inference...")
    export_weights_for_java(model, args.weights_output, args.img_width, args.img_height, hidden_sizes)
    print(f"Weights saved to: {args.weights_output}")


def export_weights_for_java(model, output_path, img_width, img_height, hidden_sizes):
    """
    Export model weights in a simple binary format for Java inference.

    Format:
    - Magic number: 0x434F524E ("CORN" for corners)
    - Version: 1
    - img_width, img_height (ints)
    - num_layers (int)
    - For each layer: num_weights, weights (big-endian floats)

    Output is 8 normalized corner values [-1, +1]:
    [TL_x, TL_y, TR_x, TR_y, BR_x, BR_y, BL_x, BL_y]

    To denormalize in Java:
        x_pixel = x_norm * (IMG_WIDTH / 2) + (IMG_WIDTH / 2)
        y_pixel = y_norm * (IMG_HEIGHT / 2) + (IMG_HEIGHT / 2)
    """
    import struct

    model.eval()
    model.cpu()

    with open(output_path, 'wb') as f:
        # Magic number and version
        f.write(struct.pack('>I', 0x434F524E))  # "CORN"
        f.write(struct.pack('>I', 1))  # version 1

        # Image dimensions
        f.write(struct.pack('>I', img_width))
        f.write(struct.pack('>I', img_height))

        # Number of hidden layers
        f.write(struct.pack('>I', len(hidden_sizes)))
        for size in hidden_sizes:
            f.write(struct.pack('>I', size))

        # Export each layer's weights and biases
        for name, param in model.named_parameters():
            data = param.detach().numpy().flatten()
            # Write number of elements
            f.write(struct.pack('>I', len(data)))
            # Write each float (big-endian)
            for val in data:
                f.write(struct.pack('>f', float(val)))

    print(f"  Exported {sum(p.numel() for p in model.parameters()):,} parameters")


if __name__ == "__main__":
    main()
