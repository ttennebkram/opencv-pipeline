#!/usr/bin/env python3
"""
train_corners_cnn.py

Train a CNN to predict normalized corner positions from distorted images.

CNN architecture is much more efficient than fully-connected:
- FC [512,256,64]: 4.87M parameters
- CNN: ~400K parameters (12x fewer!)

The CNN leverages spatial structure - corners are local features (edges meeting)
that CNNs naturally detect through hierarchical conv layers.

Usage:
    python3 train_corners_cnn.py --data training_data/homography --epochs 50

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

    def __init__(self, data_dir, img_width=128, img_height=72, augment=False,
                 data_offset=0, data_limit=None):
        self.data_dir = data_dir
        self.img_width = img_width
        self.img_height = img_height
        self.augment = augment  # Enable on-the-fly augmentation
        self.samples = []

        # Find all label files
        labels_dir = os.path.join(data_dir, "labels")
        images_dir = os.path.join(data_dir, "images")

        if not os.path.exists(labels_dir):
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

        label_files = sorted(glob.glob(os.path.join(labels_dir, "*.json")))  # Sort for deterministic order
        print(f"Found {len(label_files)} label files")

        for label_path in label_files:
            # Get corresponding image path (normal version, not white)
            base_name = os.path.splitext(os.path.basename(label_path))[0]
            image_path = os.path.join(images_dir, base_name + "_normal.jpg")

            if os.path.exists(image_path):
                self.samples.append((image_path, label_path))

        # Apply offset and limit for parallel training (slice the data)
        total_samples = len(self.samples)
        if data_offset > 0 or data_limit is not None:
            end_idx = data_offset + data_limit if data_limit else len(self.samples)
            self.samples = self.samples[data_offset:end_idx]
            print(f"Using samples [{data_offset}:{end_idx}] of {total_samples} total")

        print(f"Loaded {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        # Load image, convert to grayscale, resize
        image = Image.open(img_path).convert("L")  # Grayscale
        image = image.resize((self.img_width, self.img_height), Image.BILINEAR)

        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(image, dtype=np.float32) / 255.0

        # Load corners from label
        with open(label_path, "r") as f:
            data = json.load(f)

        corners = np.array(data["corners"], dtype=np.float32).reshape(4, 2)

        # Normalize corners to [-1, +1] range
        corners[:, 0] = (corners[:, 0] - IMG_WIDTH / 2) / (IMG_WIDTH / 2)
        corners[:, 1] = (corners[:, 1] - IMG_HEIGHT / 2) / (IMG_HEIGHT / 2)

        # Apply augmentations if enabled
        # Total variety: 4 geometric × continuous photometric ≈ 100+ unique variations per image
        # - Geometric: 4 flip combinations (none, H, V, both)
        # - Photometric: brightness × contrast × gamma × noise × blur × cutout = continuous
        if self.augment:
            # === GEOMETRIC AUGMENTATIONS (4× multiplier) ===
            # Horizontal flip (50% chance)
            if np.random.random() > 0.5:
                img_array = np.fliplr(img_array).copy()
                # Transform corners: swap TL↔TR (indices 0↔1), BL↔BR (indices 3↔2)
                # Corner order: [TL, TR, BR, BL]
                corners = corners[[1, 0, 3, 2], :]  # Swap pairs
                corners[:, 0] = -corners[:, 0]  # Negate x coordinates

            # Vertical flip (50% chance)
            if np.random.random() > 0.5:
                img_array = np.flipud(img_array).copy()
                # Transform corners: swap TL↔BL (indices 0↔3), TR↔BR (indices 1↔2)
                corners = corners[[3, 2, 1, 0], :]  # Swap pairs
                corners[:, 1] = -corners[:, 1]  # Negate y coordinates

            # === PHOTOMETRIC AUGMENTATIONS (continuous variety) ===
            # Brightness variation (±15%)
            brightness = 1.0 + np.random.uniform(-0.15, 0.15)
            img_array = np.clip(img_array * brightness, 0, 1)

            # Contrast variation (±15%)
            contrast = 1.0 + np.random.uniform(-0.15, 0.15)
            mean = img_array.mean()
            img_array = np.clip((img_array - mean) * contrast + mean, 0, 1)

            # Gamma correction (50% chance, γ=0.8-1.2)
            if np.random.random() > 0.5:
                gamma = np.random.uniform(0.8, 1.2)
                img_array = np.clip(np.power(img_array, gamma), 0, 1)

            # Gaussian noise (50% chance, σ=0.02)
            if np.random.random() > 0.5:
                noise = np.random.normal(0, 0.02, img_array.shape).astype(np.float32)
                img_array = np.clip(img_array + noise, 0, 1)

            # Gaussian blur (30% chance, σ=0.5-1.0)
            if np.random.random() > 0.7:
                from scipy.ndimage import gaussian_filter
                sigma = np.random.uniform(0.5, 1.0)
                img_array = gaussian_filter(img_array, sigma=sigma)

            # Random erasing/cutout (30% chance, 5-15% area)
            if np.random.random() > 0.7:
                h, w = img_array.shape
                area_ratio = np.random.uniform(0.05, 0.15)
                erase_h = int(np.sqrt(area_ratio * h * w * np.random.uniform(0.5, 2.0)))
                erase_w = int(area_ratio * h * w / max(erase_h, 1))
                erase_h = min(erase_h, h - 1)
                erase_w = min(erase_w, w - 1)
                top = np.random.randint(0, h - erase_h)
                left = np.random.randint(0, w - erase_w)
                img_array[top:top+erase_h, left:left+erase_w] = np.random.uniform(0, 1)

        img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # Add channel dim
        target_tensor = torch.from_numpy(corners.flatten())

        return img_tensor, target_tensor


class CornersCNN(nn.Module):
    """
    CNN for corner position estimation.

    Architecture:
    - 4 conv blocks with increasing filters (32 -> 64 -> 128 -> 256)
    - Each block: Conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2
    - Global average pooling to reduce to feature vector
    - Single dense layer to 8 outputs

    For 128x72 input:
    - After conv1 + pool: 64x36x32
    - After conv2 + pool: 32x18x64
    - After conv3 + pool: 16x9x128
    - After conv4 + pool: 8x4x256
    - After GAP: 256
    - Output: 8
    """

    def __init__(self, input_height=72, input_width=128, base_filters=32):
        super(CornersCNN, self).__init__()

        self.input_height = input_height
        self.input_width = input_width

        # Conv blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Output layer
        self.fc = nn.Linear(base_filters * 8, 8)

        # Print architecture info
        self._print_info(base_filters)

    def _print_info(self, base_filters):
        # Calculate feature map sizes
        h, w = self.input_height, self.input_width
        print(f"  Input size: {w}x{h}")

        h, w = h // 2, w // 2
        print(f"  After conv1+pool: {w}x{h}x{base_filters}")

        h, w = h // 2, w // 2
        print(f"  After conv2+pool: {w}x{h}x{base_filters*2}")

        h, w = h // 2, w // 2
        print(f"  After conv3+pool: {w}x{h}x{base_filters*4}")

        h, w = h // 2, w // 2
        print(f"  After conv4+pool: {w}x{h}x{base_filters*8}")

        print(f"  After GAP: {base_filters*8}")
        print(f"  Output: 8")

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Total parameters: {total_params:,}")
        print(f"  Memory: ~{total_params * 4 / 1024 / 1024:.2f} MB")

    def forward(self, x):
        # x: (batch, 1, height, width)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)


def corner_mae(pred, target):
    """Compute mean absolute error across all 8 corner values."""
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


def train_epoch(model, loader, optimizer, criterion, device, epoch_num=0):
    """Train for one epoch with progress output."""
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_pixel_error = 0.0
    num_batches = len(loader)

    for batch_idx, (images, targets) in enumerate(loader):
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

        # Progress output every 10% or every 50 batches
        if (batch_idx + 1) % max(1, num_batches // 10) == 0 or (batch_idx + 1) == num_batches:
            pct = 100.0 * (batch_idx + 1) / num_batches
            avg_loss = total_loss / (batch_idx + 1)
            avg_px = total_pixel_error / (batch_idx + 1)
            print(f"  Epoch {epoch_num}: {batch_idx+1}/{num_batches} ({pct:.0f}%) - loss: {avg_loss:.4f}, px_err: {avg_px:.1f}")

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
    parser = argparse.ArgumentParser(description="Train CNN for corner position estimation")
    parser.add_argument("--data", type=str, default="/Volumes/SamsungBlue/ml-training/homography/training_data",
                        help="Path to training data directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--img_width", type=int, default=128, help="Image width for training")
    parser.add_argument("--img_height", type=int, default=72, help="Image height for training")
    parser.add_argument("--base_filters", type=int, default=32,
                        help="Base number of filters (doubles each layer)")
    parser.add_argument("--output", type=str, default="corners_cnn.pth",
                        help="Output model path")
    parser.add_argument("--weights_output", type=str, default="corners_cnn.weights",
                        help="Output weights path (for Java inference)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: auto-generate unique seed from time+PID)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from checkpoint file (e.g., cnn_r4_lr0_002.pth)")
    parser.add_argument("--augment", action="store_true",
                        help="Enable on-the-fly data augmentation (horizontal flip, brightness, contrast)")
    parser.add_argument("--scheduler", type=str, default="plateau", choices=["plateau", "cosine", "cosine_warm"],
                        help="LR scheduler: plateau (ReduceLROnPlateau), cosine (CosineAnnealing), cosine_warm (with warm restarts)")
    parser.add_argument("--csv_log", type=str, default=None,
                        help="CSV file to log training metrics for graphing (epoch, train_loss, val_loss, val_mae, pixel_error, lr)")
    parser.add_argument("--results_json", type=str, default=None,
                        help="JSON file to save final results (for automated collection)")
    parser.add_argument("--instance_id", type=str, default=None,
                        help="Instance identifier (for multi-machine training)")
    parser.add_argument("--data_offset", type=int, default=0,
                        help="Starting index for data slicing (for parallel training)")
    parser.add_argument("--data_limit", type=int, default=None,
                        help="Maximum number of samples to use (for parallel training)")
    parser.add_argument("--save_checkpoints", type=str, default=None,
                        help="Comma-separated list of epochs to save checkpoints (e.g., '5,10,25,50,75,100')")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save epoch checkpoints")
    args = parser.parse_args()

    # Auto-generate unique seed if not specified (for multi-machine training)
    if args.seed is None:
        import os
        args.seed = (int(time.time() * 1000) % (2**31)) + os.getpid()
        print(f"Auto-generated seed: {args.seed}")

    print("=" * 70)
    print("Corner Position Estimation - CNN Training")
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
    print(f"Base filters: {args.base_filters}")
    print(f"Random seed: {args.seed}")
    print(f"Augmentation: {'enabled' if args.augment else 'disabled'}")
    print(f"LR Scheduler: {args.scheduler}")
    if args.data_offset > 0 or args.data_limit is not None:
        print(f"Data slice: offset={args.data_offset}, limit={args.data_limit}")
    print()

    # Load datasets (separate for train/val to enable augmentation only on train)
    print("Loading dataset...")

    # Training dataset with optional augmentation
    train_dataset_full = CornersDataset(args.data, img_width=args.img_width,
                                        img_height=args.img_height, augment=args.augment,
                                        data_offset=args.data_offset, data_limit=args.data_limit)
    # Validation dataset without augmentation
    val_dataset_full = CornersDataset(args.data, img_width=args.img_width,
                                      img_height=args.img_height, augment=False,
                                      data_offset=args.data_offset, data_limit=args.data_limit)

    if len(train_dataset_full) == 0:
        print("ERROR: No samples found!")
        sys.exit(1)

    # Split into train/val (80/20) using same seed for both datasets
    train_size = int(0.8 * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size

    # Get indices for split (same indices for both datasets)
    all_indices = list(range(len(train_dataset_full)))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(all_indices)
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:]

    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)

    # Use seeded sampler for reproducible batch ordering
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
    print("Creating CNN model...")
    model = CornersCNN(
        input_height=args.img_height,
        input_width=args.img_width,
        base_filters=args.base_filters
    )
    model = model.to(device)
    print()

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Create scheduler based on choice
    if args.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    elif args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.scheduler == "cosine_warm":
        # Warm restarts: T_0=10 epochs, then doubles each restart (10, 20, 40...)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    best_pixel_error = float('inf')

    if args.resume:
        if os.path.exists(args.resume):
            print(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            best_pixel_error = checkpoint.get('val_pixel_error', float('inf'))
            print(f"  Resuming from epoch {start_epoch}, best pixel error: {best_pixel_error:.1f}px")
            print()
        else:
            print(f"WARNING: Checkpoint not found: {args.resume}")
            print("  Starting from scratch...")
            print()

    # Training loop
    print("Starting training...")
    print("-" * 80)
    print(f"{'Epoch':>5} | {'Train Loss':>12} | {'Val Loss':>12} | {'Val MAE':>10} | {'Pixel Err':>10} | {'Time':>6}")
    print("-" * 80)

    # Initialize CSV logging if requested
    csv_file = None
    if args.csv_log:
        csv_file = open(args.csv_log, 'w')
        csv_file.write("epoch,train_loss,val_loss,val_mae,pixel_error,lr\n")
        print(f"Logging to: {args.csv_log}")

    # Parse checkpoint epochs
    checkpoint_epochs = set()
    if args.save_checkpoints:
        checkpoint_epochs = set(int(e.strip()) for e in args.save_checkpoints.split(','))
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        print(f"Will save checkpoints at epochs: {sorted(checkpoint_epochs)}")
        print(f"Checkpoint directory: {args.checkpoint_dir}")

    start_time = time.time()

    for epoch in range(start_epoch, start_epoch + args.epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_mae, train_pixel = train_epoch(model, train_loader, optimizer, criterion, device, epoch + 1)

        # Evaluate
        val_loss, val_mae, val_pixel = evaluate(model, val_loader, device)

        # Update scheduler (ReduceLROnPlateau needs val_loss, cosine schedulers don't)
        if args.scheduler == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        epoch_time = time.time() - epoch_start

        print(f"{epoch+1:5d} | {train_loss:12.6f} | {val_loss:12.6f} | {val_mae:10.4f} | {val_pixel:10.1f}px | {epoch_time:5.1f}s")

        # Log to CSV if enabled
        if csv_file:
            current_lr = optimizer.param_groups[0]['lr']
            csv_file.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f},{val_mae:.4f},{val_pixel:.1f},{current_lr:.6f}\n")
            csv_file.flush()  # Ensure data is written immediately for real-time monitoring

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_pixel_error = val_pixel
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_pixel_error': val_pixel,
                'img_width': args.img_width,
                'img_height': args.img_height,
                'base_filters': args.base_filters,
                'model_type': 'cnn',
            }, args.output)
            print(f"       -> Saved best model (Pixel Error: {val_pixel:.1f}px)")

        # Save checkpoint at specified epochs
        if (epoch + 1) in checkpoint_epochs:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch+1:03d}.pth")
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_pixel_error': val_pixel,
                'train_loss': train_loss,
                'img_width': args.img_width,
                'img_height': args.img_height,
                'base_filters': args.base_filters,
                'model_type': 'cnn',
                'lr': optimizer.param_groups[0]['lr'],
                'args': vars(args),
            }
            torch.save(checkpoint_data, checkpoint_path)

            # Also export weights for Java inference
            weights_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch+1:03d}.weights")
            export_weights_for_java(model, weights_path, args.img_width, args.img_height, args.base_filters)

            # Save results JSON for this checkpoint
            results_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch+1:03d}.json")
            import socket
            checkpoint_results = {
                "epoch": epoch + 1,
                "val_loss": float(val_loss),
                "val_mae": float(val_mae),
                "val_pixel_error": float(val_pixel),
                "train_loss": float(train_loss),
                "lr": float(optimizer.param_groups[0]['lr']),
                "total_time_sec": float(time.time() - start_time),
                "instance_id": args.instance_id or socket.gethostname(),
            }
            with open(results_path, 'w') as f:
                json.dump(checkpoint_results, f, indent=2)

            print(f"       -> Checkpoint saved: {checkpoint_path}")

    total_time = time.time() - start_time
    print("-" * 80)
    print(f"Training complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best pixel error: {best_pixel_error:.1f}px")
    print(f"Model saved to: {args.output}")

    # Close CSV file
    if csv_file:
        csv_file.close()
        print(f"Training log saved to: {args.csv_log}")

    # Export weights for Java inference
    print(f"\nExporting weights for Java inference...")
    export_weights_for_java(model, args.weights_output, args.img_width, args.img_height, args.base_filters)
    print(f"Weights saved to: {args.weights_output}")

    # Save results JSON for automated collection
    if args.results_json:
        import socket
        results = {
            "instance_id": args.instance_id or socket.gethostname(),
            "seed": args.seed,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "scheduler": args.scheduler,
            "augment": args.augment,
            "img_size": f"{args.img_width}x{args.img_height}",
            "base_filters": args.base_filters,
            "final_val_loss": float(val_loss),
            "final_pixel_error": float(val_pixel),
            "best_val_loss": float(best_val_loss),
            "best_pixel_error": float(best_pixel_error),
            "training_time_sec": float(total_time),
            "device": str(device),
        }
        with open(args.results_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.results_json}")


def export_weights_for_java(model, output_path, img_width, img_height, base_filters, device=None):
    """
    Export CNN weights in a simple binary format for Java inference.

    Format:
    - Magic number: 0x434E4E43 ("CNNC" for CNN Corners)
    - Version: 1
    - img_width, img_height (ints)
    - base_filters (int)
    - For each layer: num_weights, weights (big-endian floats)

    Note: This function temporarily moves model to CPU for export, then restores it.
    """
    import struct

    # Remember original device and mode
    original_device = next(model.parameters()).device
    was_training = model.training

    model.eval()
    model.cpu()

    with open(output_path, 'wb') as f:
        # Magic number and version
        f.write(struct.pack('>I', 0x434E4E43))  # "CNNC"
        f.write(struct.pack('>I', 1))  # version 1

        # Image dimensions and architecture
        f.write(struct.pack('>I', img_width))
        f.write(struct.pack('>I', img_height))
        f.write(struct.pack('>I', base_filters))

        # Export each layer's weights and biases
        for name, param in model.named_parameters():
            data = param.detach().numpy().flatten()
            # Write number of elements
            f.write(struct.pack('>I', len(data)))
            # Write each float (big-endian)
            for val in data:
                f.write(struct.pack('>f', float(val)))

    print(f"  Exported {sum(p.numel() for p in model.parameters()):,} parameters")

    # Restore model to original device and mode
    model.to(original_device)
    if was_training:
        model.train()


if __name__ == "__main__":
    main()
