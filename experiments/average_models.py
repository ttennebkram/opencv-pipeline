#!/usr/bin/env python3
"""
Average weights from multiple CNN model .pth files.
Creates an averaged model while preserving all individual models.

NOTE: BatchNorm running statistics (running_mean, running_var) must be
recalibrated after averaging weights. Use --recalibrate with a data path.
"""

import torch
import torch.nn as nn
import os
import glob
import json
import argparse
import numpy as np
from PIL import Image


def average_model_weights(model_paths, output_path):
    """
    Load multiple .pth files and average their weights.

    Args:
        model_paths: List of paths to .pth model files
        output_path: Path to save the averaged model

    Returns:
        dict with averaging stats
    """
    if not model_paths:
        raise ValueError("No model paths provided")

    print(f"Loading {len(model_paths)} models...")

    # Load all state dicts
    state_dicts = []
    for path in model_paths:
        print(f"  Loading {os.path.basename(path)}...")
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        # Handle nested model_state_dict format
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        state_dicts.append(state_dict)

    # Get the keys from the first model
    keys = state_dicts[0].keys()
    print(f"\nModel has {len(keys)} keys")

    # Average all weights
    print("Averaging weights...")
    averaged_state_dict = {}
    tensor_count = 0
    skipped_count = 0

    # BatchNorm running statistics should NOT be averaged - they track dataset
    # statistics during training. Averaging them corrupts the normalization.
    # We'll copy these from the first model instead.
    batchnorm_stats = ('running_mean', 'running_var', 'num_batches_tracked')

    for key in keys:
        val = state_dicts[0][key]

        # Check if this is a BatchNorm running statistic
        is_bn_stat = any(stat in key for stat in batchnorm_stats)

        if isinstance(val, torch.Tensor):
            if is_bn_stat:
                # Don't average BatchNorm running stats - copy from first model
                averaged_state_dict[key] = val
                skipped_count += 1
                print(f"  {key}: (BatchNorm stat - copied from model 1)")
            else:
                # Average learnable parameters
                tensors = [sd[key].float() for sd in state_dicts]
                stacked = torch.stack(tensors, dim=0)
                averaged = torch.mean(stacked, dim=0)
                averaged_state_dict[key] = averaged
                tensor_count += 1

                # Print some stats for major layers
                if 'weight' in key and len(averaged.shape) >= 2:
                    print(f"  {key}: shape={list(averaged.shape)}, mean={averaged.mean().item():.6f}, std={averaged.std().item():.6f}")
        else:
            # Non-tensor (like epoch count) - just copy from first model
            averaged_state_dict[key] = val
            print(f"  {key}: (non-tensor) = {val}")

    print(f"\nAveraged {tensor_count} tensor parameters")
    print(f"Skipped {skipped_count} BatchNorm running statistics")

    # Save the averaged model
    print(f"\nSaving averaged model to {output_path}")
    torch.save(averaged_state_dict, output_path)

    # Verify file was saved
    file_size = os.path.getsize(output_path)
    print(f"Saved: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")

    return {
        'num_models': len(model_paths),
        'num_parameters': len(keys),
        'output_path': output_path,
        'output_size_bytes': file_size
    }


class CornersCNN(nn.Module):
    """CNN for corner position estimation (must match training architecture)."""

    def __init__(self, input_height=72, input_width=128, base_filters=32):
        super(CornersCNN, self).__init__()
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
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_filters * 8, 8)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def recalibrate_batchnorm(model, data_dir, num_batches=100, batch_size=32, img_size=(128, 72)):
    """
    Recalibrate BatchNorm running statistics by running forward passes on training data.
    This is necessary after averaging weights from multiple models.
    """
    print(f"\nRecalibrating BatchNorm statistics using data from {data_dir}...")

    images_dir = os.path.join(data_dir, "images")
    if not os.path.exists(images_dir):
        print(f"  WARNING: Images directory not found: {images_dir}")
        return

    # Get list of images
    image_files = sorted(glob.glob(os.path.join(images_dir, "*_normal.jpg")))
    if not image_files:
        print(f"  WARNING: No images found in {images_dir}")
        return

    print(f"  Found {len(image_files)} images")

    # Put model in train mode (enables BatchNorm stat tracking)
    model.train()

    # Reset running stats
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.reset_running_stats()

    # Run forward passes
    img_width, img_height = img_size
    samples_processed = 0

    with torch.no_grad():
        for batch_idx in range(min(num_batches, len(image_files) // batch_size)):
            batch_images = []
            for i in range(batch_size):
                idx = batch_idx * batch_size + i
                if idx >= len(image_files):
                    break

                # Load and preprocess image
                img = Image.open(image_files[idx]).convert('L')
                img = img.resize((img_width, img_height), Image.BILINEAR)
                arr = np.array(img, dtype=np.float32) / 255.0
                batch_images.append(arr)

            if not batch_images:
                break

            # Stack into batch tensor
            batch_tensor = torch.from_numpy(np.stack(batch_images)).unsqueeze(1)

            # Forward pass (updates BatchNorm running stats)
            _ = model(batch_tensor)
            samples_processed += len(batch_images)

            if (batch_idx + 1) % 20 == 0:
                print(f"  Processed {samples_processed} samples...")

    print(f"  Recalibrated using {samples_processed} samples")

    # Put back in eval mode
    model.eval()


def main():
    parser = argparse.ArgumentParser(description='Average CNN model weights')
    parser.add_argument('--input_dir', type=str, default='aws_models',
                       help='Directory containing model_seed*.pth files')
    parser.add_argument('--output', type=str, default='aws_models/model_averaged.pth',
                       help='Output path for averaged model')
    parser.add_argument('--pattern', type=str, default='model_seed*.pth',
                       help='Glob pattern to match model files')
    parser.add_argument('--recalibrate', type=str, default=None,
                       help='Path to training data directory for BatchNorm recalibration')
    parser.add_argument('--select_best', action='store_true',
                       help='Instead of averaging, select the model with lowest validation error')
    args = parser.parse_args()

    # Find all model files
    pattern = os.path.join(args.input_dir, args.pattern)
    model_paths = sorted(glob.glob(pattern))

    if not model_paths:
        print(f"No model files found matching {pattern}")
        return 1

    print(f"Found {len(model_paths)} model files:")
    for path in model_paths:
        print(f"  {path}")
    print()

    # Select best model instead of averaging
    if args.select_best:
        print("Selecting best model by validation error...")
        best_path = None
        best_error = float('inf')

        for path in model_paths:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            if 'val_pixel_error' in checkpoint:
                error = checkpoint['val_pixel_error']
                print(f"  {os.path.basename(path)}: {error:.2f}px")
                if error < best_error:
                    best_error = error
                    best_path = path

        if best_path:
            print(f"\nBest model: {best_path} ({best_error:.2f}px)")
            # Copy best model to output
            import shutil
            shutil.copy(best_path, args.output)
            print(f"Copied to: {args.output}")

            print("\n" + "="*50)
            print("SELECTION COMPLETE")
            print("="*50)
            print(f"Best model: {os.path.basename(best_path)}")
            print(f"Validation error: {best_error:.2f}px")
            print(f"Output: {args.output}")
            return 0
        else:
            print("No models with validation error found!")
            return 1

    # Average the models
    stats = average_model_weights(model_paths, args.output)

    # Recalibrate BatchNorm if data path provided
    if args.recalibrate:
        # Load averaged weights into model
        model = CornersCNN()
        model.load_state_dict(torch.load(args.output, map_location='cpu', weights_only=True))

        # Recalibrate
        recalibrate_batchnorm(model, args.recalibrate)

        # Save recalibrated model
        print(f"\nSaving recalibrated model to {args.output}")
        torch.save(model.state_dict(), args.output)
        file_size = os.path.getsize(args.output)
        print(f"Saved: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")

    print("\n" + "="*50)
    print("AVERAGING COMPLETE")
    print("="*50)
    print(f"Models averaged: {stats['num_models']}")
    print(f"Parameters: {stats['num_parameters']}")
    print(f"Output: {stats['output_path']}")
    if args.recalibrate:
        print(f"BatchNorm recalibrated: Yes (using {args.recalibrate})")
    else:
        print(f"BatchNorm recalibrated: No (use --recalibrate for better results)")
    print()
    print("Individual models preserved in:")
    for path in model_paths:
        print(f"  {path}")

    return 0


if __name__ == '__main__':
    exit(main())
