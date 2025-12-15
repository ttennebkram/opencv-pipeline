#!/usr/bin/env python3
"""
visualize_corners_prediction.py

Visualize corner predictions by:
1. Loading a random distorted training image
2. Running inference to get predicted corner positions
3. Computing homography from corners
4. Applying ground truth vs predicted inverse transforms
5. Showing side-by-side comparison

Usage:
    python3 visualize_corners_prediction.py [--model corners_model.pth] [--sample N]
"""

import argparse
import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import cv2

# Image dimensions for denormalization (must match training data generation)
IMG_WIDTH = 1920
IMG_HEIGHT = 1080


class CornersNet(nn.Module):
    """Same architecture as training script."""

    def __init__(self, input_height=72, input_width=128, hidden_sizes=[512, 256, 64]):
        super(CornersNet, self).__init__()

        self.input_size = input_height * input_width
        self.output_size = 8  # 4 corners × 2 coordinates

        layers = []
        prev_size = self.input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU(inplace=True))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, self.output_size))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.fc(x)


def load_model(model_path):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    img_width = checkpoint.get('img_width', 128)
    img_height = checkpoint.get('img_height', 72)
    hidden_sizes = checkpoint.get('hidden_sizes', [512, 256, 64])

    model = CornersNet(
        input_height=img_height,
        input_width=img_width,
        hidden_sizes=hidden_sizes
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Print model info
    val_pixel_error = checkpoint.get('val_pixel_error', None)
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"  Loaded from epoch {epoch}")
    if val_pixel_error:
        print(f"  Validation pixel error: {val_pixel_error:.1f}px")

    return model, img_width, img_height


def load_random_sample(data_dir, sample_idx=None):
    """Load a random sample from the training data."""
    labels_dir = os.path.join(data_dir, "labels")
    images_dir = os.path.join(data_dir, "images")

    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.json')])

    if sample_idx is None:
        sample_idx = random.randint(0, len(label_files) - 1)
    else:
        sample_idx = sample_idx % len(label_files)

    label_file = label_files[sample_idx]
    base_name = os.path.splitext(label_file)[0]

    # Load label
    with open(os.path.join(labels_dir, label_file), 'r') as f:
        label_data = json.load(f)

    # Load distorted image (normal version)
    image_path = os.path.join(images_dir, base_name + "_normal.jpg")
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None, None, None

    image = cv2.imread(image_path)

    return image, label_data, sample_idx


def preprocess_image(image, img_width, img_height):
    """Preprocess image for inference."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize to model input size
    resized = cv2.resize(gray, (img_width, img_height), interpolation=cv2.INTER_LINEAR)

    # Normalize to [0, 1] and convert to tensor
    normalized = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims

    return tensor


def run_inference(model, image_tensor):
    """Run inference and return predicted corners (normalized)."""
    with torch.no_grad():
        output = model(image_tensor)
    return output.numpy()[0]


def denormalize_corners(corners_norm):
    """Convert normalized corners [-1, +1] to pixel coordinates."""
    corners = corners_norm.reshape(4, 2).copy()
    corners[:, 0] = corners[:, 0] * (IMG_WIDTH / 2) + (IMG_WIDTH / 2)
    corners[:, 1] = corners[:, 1] * (IMG_HEIGHT / 2) + (IMG_HEIGHT / 2)
    return corners


def corners_to_homography(src_corners, dst_corners):
    """Compute homography from source to destination corners."""
    src = np.array(src_corners, dtype=np.float32)
    dst = np.array(dst_corners, dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H


def apply_homography(image, H, output_size=None):
    """Apply homography transform to image."""
    if output_size is None:
        output_size = (image.shape[1], image.shape[0])

    return cv2.warpPerspective(image, H, output_size, flags=cv2.INTER_LINEAR)


def draw_corners(image, corners, color, label, thickness=3):
    """Draw corners and connecting lines on image."""
    img = image.copy()
    pts = corners.astype(np.int32)

    # Draw polygon
    cv2.polylines(img, [pts], True, color, thickness)

    # Draw corner points
    for i, pt in enumerate(pts):
        cv2.circle(img, tuple(pt), 8, color, -1)
        cv2.putText(img, str(i), (pt[0]+10, pt[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Add label
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    return img


def create_comparison_image(original, gt_corners, pred_corners, gt_corrected, pred_corrected, diff_image):
    """Create a 2x3 grid comparison image."""
    target_h, target_w = 360, 640  # Third of 1080p

    def resize_img(img):
        return cv2.resize(img, (target_w, target_h))

    # Draw corners on original
    orig_with_gt = draw_corners(original.copy(), gt_corners, (0, 255, 0), "Ground Truth")
    orig_with_pred = draw_corners(original.copy(), pred_corners, (0, 0, 255), "Predicted")

    # Overlay both on one image
    orig_overlay = draw_corners(original.copy(), gt_corners, (0, 255, 0), "GT (green)")
    orig_overlay = draw_corners(orig_overlay, pred_corners, (0, 0, 255), "Pred (red)")

    # Resize all
    orig_gt_resized = resize_img(orig_with_gt)
    orig_pred_resized = resize_img(orig_with_pred)
    orig_overlay_resized = resize_img(orig_overlay)
    gt_resized = resize_img(gt_corrected)
    pred_resized = resize_img(pred_corrected)
    diff_resized = resize_img(diff_image)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    cv2.putText(gt_resized, "GT Inverse Applied", (10, 30), font, font_scale, (0, 255, 0), thickness)
    cv2.putText(pred_resized, "Predicted Inverse Applied", (10, 30), font, font_scale, (0, 0, 255), thickness)
    cv2.putText(diff_resized, "Difference (amplified 3x)", (10, 30), font, font_scale, (255, 255, 0), thickness)

    # Create 2x3 grid
    top_row = np.hstack([orig_gt_resized, orig_pred_resized, orig_overlay_resized])
    bottom_row = np.hstack([gt_resized, pred_resized, diff_resized])
    grid = np.vstack([top_row, bottom_row])

    return grid


def main():
    parser = argparse.ArgumentParser(description="Visualize corner predictions")
    parser.add_argument("--model", type=str, default="corners_model.pth",
                        help="Path to trained model")
    parser.add_argument("--data", type=str,
                        default="/Volumes/SamsungBlue/ml-training/homography/training_data",
                        help="Path to training data")
    parser.add_argument("--sample", type=int, default=None,
                        help="Specific sample index (random if not specified)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path (displays if not specified)")
    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        alt_path = os.path.join(os.path.dirname(__file__), args.model)
        if os.path.exists(alt_path):
            args.model = alt_path
        else:
            print(f"Model not found at {alt_path} either")
            sys.exit(1)

    print("=" * 70)
    print("Corner Prediction Visualization")
    print("=" * 70)

    # Load model
    print(f"Loading model: {args.model}")
    model, img_width, img_height = load_model(args.model)
    print(f"  Input size: {img_width}x{img_height}")

    # Load random sample
    print(f"\nLoading sample from: {args.data}")
    image, label_data, sample_idx = load_random_sample(args.data, args.sample)

    if image is None:
        print("Failed to load sample")
        sys.exit(1)

    print(f"  Sample index: {sample_idx}")
    print(f"  Image size: {image.shape[1]}x{image.shape[0]}")

    # Extract ground truth corners from label
    gt_corners = np.array(label_data['corners'], dtype=np.float32).reshape(4, 2)
    print(f"\nGround Truth Corners:\n{gt_corners}")

    # Ground truth inverse (from JSON)
    gt_inverse = np.array(label_data['inverse']).reshape(3, 3)

    # Preprocess and run inference
    print("\nRunning inference...")
    image_tensor = preprocess_image(image, img_width, img_height)
    pred_corners_norm = run_inference(model, image_tensor)

    # Denormalize predicted corners
    pred_corners = denormalize_corners(pred_corners_norm)
    print(f"\nPredicted Corners:\n{pred_corners}")

    # Compute corner errors
    corner_errors = np.sqrt(((gt_corners - pred_corners) ** 2).sum(axis=1))
    print(f"\nPer-corner errors (pixels):")
    for i, err in enumerate(corner_errors):
        print(f"  Corner {i}: {err:.1f}px")
    print(f"  Average: {corner_errors.mean():.1f}px")

    # Standard document corners (full frame)
    doc_corners = np.array([
        [0, 0],              # Top-left
        [IMG_WIDTH, 0],      # Top-right
        [IMG_WIDTH, IMG_HEIGHT],  # Bottom-right
        [0, IMG_HEIGHT]      # Bottom-left
    ], dtype=np.float32)

    # Compute inverse homographies (from distorted corners back to full frame)
    try:
        pred_inverse = corners_to_homography(pred_corners, doc_corners)
    except Exception as e:
        print(f"Error computing predicted inverse: {e}")
        pred_inverse = np.eye(3)

    print(f"\nGround Truth Inverse:\n{gt_inverse}")
    print(f"\nPredicted Inverse:\n{pred_inverse}")

    # Apply transforms
    print("\nApplying transforms...")

    # Apply ground truth inverse to "correct" the distorted image
    gt_corrected = apply_homography(image, gt_inverse)

    # Apply predicted inverse
    try:
        pred_corrected = apply_homography(image, pred_inverse)
    except Exception as e:
        print(f"Error applying predicted inverse: {e}")
        pred_corrected = np.zeros_like(image)

    # Compute difference image
    diff = cv2.absdiff(gt_corrected, pred_corrected)
    # Amplify difference for visibility
    diff = cv2.convertScaleAbs(diff, alpha=3.0)

    # Create comparison grid
    comparison = create_comparison_image(image, gt_corners, pred_corners,
                                         gt_corrected, pred_corrected, diff)

    # Save or display
    if args.output:
        cv2.imwrite(args.output, comparison)
        print(f"\nSaved comparison to: {args.output}")
    else:
        output_path = "/tmp/corners_comparison.png"
        cv2.imwrite(output_path, comparison)
        print(f"\nSaved comparison to: {output_path}")
        # Try to open it
        import subprocess
        subprocess.run(["open", output_path], check=False)

    print("\nDone!")


if __name__ == "__main__":
    main()
