#!/usr/bin/env python3
"""
visualize_prediction.py

Visualize homography predictions by:
1. Loading a random distorted training image
2. Running inference to get predicted homography matrices
3. Applying ground truth vs predicted inverse transforms
4. Showing side-by-side comparison

Usage:
    python3 visualize_prediction.py [--model homography_corners.pth] [--sample N]
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

# Import the model class from training script
sys.path.insert(0, os.path.dirname(__file__))


class HomographyNet(nn.Module):
    """Same architecture as training script."""

    def __init__(self, input_height=72, input_width=128, hidden_sizes=[512, 256, 64]):
        super(HomographyNet, self).__init__()

        self.input_size = input_height * input_width
        self.output_size = 18

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


def inverse_log_transform(y):
    """Inverse of sign-preserving log transform: sign(y) * (exp(|y|) - 1)"""
    return np.sign(y) * (np.exp(np.abs(y)) - 1)


def load_model(model_path):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location='cpu')

    img_width = checkpoint.get('img_width', 128)
    img_height = checkpoint.get('img_height', 72)
    hidden_sizes = checkpoint.get('hidden_sizes', [512, 256, 64])

    model = HomographyNet(
        input_height=img_height,
        input_width=img_width,
        hidden_sizes=hidden_sizes
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

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
    """Run inference and return raw output (log-transformed)."""
    with torch.no_grad():
        output = model(image_tensor)
    return output.numpy()[0]


def apply_homography(image, H, output_size=None):
    """Apply homography transform to image."""
    if output_size is None:
        output_size = (image.shape[1], image.shape[0])

    return cv2.warpPerspective(image, H, output_size, flags=cv2.INTER_LINEAR)


def create_comparison_image(original, gt_corrected, pred_corrected, diff_image):
    """Create a 2x2 grid comparison image."""
    h, w = original.shape[:2]

    # Resize all to same size
    target_h, target_w = 540, 960  # Half of 1080p

    def resize_img(img):
        return cv2.resize(img, (target_w, target_h))

    original_resized = resize_img(original)
    gt_resized = resize_img(gt_corrected)
    pred_resized = resize_img(pred_corrected)
    diff_resized = resize_img(diff_image)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    color = (0, 255, 0)

    cv2.putText(original_resized, "Original (Distorted)", (10, 40), font, font_scale, color, thickness)
    cv2.putText(gt_resized, "Ground Truth Inverse", (10, 40), font, font_scale, color, thickness)
    cv2.putText(pred_resized, "Predicted Inverse", (10, 40), font, font_scale, color, thickness)
    cv2.putText(diff_resized, "Difference (GT - Pred)", (10, 40), font, font_scale, color, thickness)

    # Create 2x2 grid
    top_row = np.hstack([original_resized, gt_resized])
    bottom_row = np.hstack([pred_resized, diff_resized])
    grid = np.vstack([top_row, bottom_row])

    return grid


def main():
    parser = argparse.ArgumentParser(description="Visualize homography predictions")
    parser.add_argument("--model", type=str, default="homography_corners.pth",
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
        print("Looking for model in experiments directory...")
        alt_path = os.path.join(os.path.dirname(__file__), args.model)
        if os.path.exists(alt_path):
            args.model = alt_path
        else:
            print(f"Model not found at {alt_path} either")
            sys.exit(1)

    print("=" * 70)
    print("Homography Prediction Visualization")
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

    # Extract ground truth matrices (already in original scale in JSON)
    gt_homography = np.array(label_data['homography']).reshape(3, 3)
    gt_inverse = np.array(label_data['inverse']).reshape(3, 3)

    print(f"\nGround Truth Homography:\n{gt_homography}")
    print(f"\nGround Truth Inverse:\n{gt_inverse}")

    # Preprocess and run inference
    print("\nRunning inference...")
    image_tensor = preprocess_image(image, img_width, img_height)
    pred_log = run_inference(model, image_tensor)

    # Convert from log scale back to original scale
    pred_raw = inverse_log_transform(pred_log)

    pred_homography = pred_raw[:9].reshape(3, 3)
    pred_inverse = pred_raw[9:].reshape(3, 3)

    print(f"\nPredicted Homography:\n{pred_homography}")
    print(f"\nPredicted Inverse:\n{pred_inverse}")

    # Compute errors
    h_error = np.abs(gt_homography - pred_homography).mean()
    inv_error = np.abs(gt_inverse - pred_inverse).mean()
    print(f"\nHomography MAE: {h_error:.4f}")
    print(f"Inverse MAE: {inv_error:.4f}")

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
    comparison = create_comparison_image(image, gt_corrected, pred_corrected, diff)

    # Save or display
    if args.output:
        cv2.imwrite(args.output, comparison)
        print(f"\nSaved comparison to: {args.output}")
    else:
        # Display
        cv2.imshow("Homography Prediction Comparison", comparison)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("\nDone!")


if __name__ == "__main__":
    main()
