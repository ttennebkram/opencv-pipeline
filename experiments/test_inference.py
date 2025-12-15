#!/usr/bin/env python3
"""
test_inference.py

Generate random test images, run CNN inference to predict corners,
then dewarp the image using OpenCV perspective transform.

Usage:
    python3 test_inference.py --model aws_models/model_averaged.pth --samples 5
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
import cv2

# Image dimensions (matching training data)
IMG_WIDTH = 1920
IMG_HEIGHT = 1080

# Training input size
TRAIN_WIDTH = 128
TRAIN_HEIGHT = 72


class CornersCNN(nn.Module):
    """CNN for corner position estimation (must match training architecture)."""

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

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def generate_test_image(style='training'):
    """
    Generate a test image with a paper/document.

    style: 'training' - match training data style (simple colored quad)
           'detailed' - with registration marks and content

    Returns the image and ground truth corners.
    """
    import random

    # Random scale and position (same as training data)
    scale = random.uniform(0.2, 0.8)
    w = IMG_WIDTH * scale
    h = IMG_HEIGHT * scale

    # Random center position
    cx = random.uniform(w/2, IMG_WIDTH - w/2)
    cy = random.uniform(h/2, IMG_HEIGHT - h/2)

    # Base rectangle corners (TL, TR, BR, BL)
    corners = np.array([
        [cx - w/2, cy - h/2],  # TL
        [cx + w/2, cy - h/2],  # TR
        [cx + w/2, cy + h/2],  # BR
        [cx - w/2, cy + h/2],  # BL
    ], dtype=np.float32)

    # Apply perspective distortion (same as training: 15% gaussian)
    for i in range(4):
        corners[i, 0] += random.gauss(0, w * 0.15)
        corners[i, 1] += random.gauss(0, h * 0.15)
        margin = 20
        corners[i, 0] = max(margin, min(IMG_WIDTH - margin, corners[i, 0]))
        corners[i, 1] = max(margin, min(IMG_HEIGHT - margin, corners[i, 1]))

    if style == 'training':
        # Match training data style exactly
        # Background: solid, gradient, or noisy (dark colors)
        bg_type = random.choice(['solid', 'gradient', 'noisy'])
        if bg_type == 'solid':
            bg_color = tuple(random.randint(30, 150) for _ in range(3))
            img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), bg_color)
        elif bg_type == 'gradient':
            img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT))
            pixels = img.load()
            c1 = tuple(random.randint(30, 100) for _ in range(3))
            c2 = tuple(random.randint(30, 100) for _ in range(3))
            for y in range(IMG_HEIGHT):
                t = y / IMG_HEIGHT
                for x in range(IMG_WIDTH):
                    r = int(c1[0] * (1-t) + c2[0] * t)
                    g = int(c1[1] * (1-t) + c2[1] * t)
                    b = int(c1[2] * (1-t) + c2[2] * t)
                    pixels[x, y] = (r, g, b)
        else:  # noisy
            arr = np.random.randint(30, 120, (IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
            img = Image.fromarray(arr)

        draw = ImageDraw.Draw(img)

        # Paper content (same types as training)
        content_type = random.choice(['gradient', 'solid', 'text_like'])

        # Create paper content
        if content_type == 'gradient':
            c1 = tuple(random.randint(150, 255) for _ in range(3))
            c2 = tuple(random.randint(150, 255) for _ in range(3))
            paper_color = tuple((c1[i] + c2[i]) // 2 for i in range(3))
        elif content_type == 'solid':
            paper_color = tuple(random.randint(180, 255) for _ in range(3))
        else:
            paper_color = (250, 250, 250)

        # Draw paper polygon
        polygon = [(corners[i, 0], corners[i, 1]) for i in range(4)]
        draw.polygon(polygon, fill=paper_color)

        # Add orientation marker (L-shape in top-left) - same as training
        marker_color = tuple(random.randint(20, 80) for _ in range(3))
        tl_x, tl_y = corners[0]
        # Simple L-shape near TL corner
        m = 20  # margin from corner
        size = 40
        thick = 8
        draw.rectangle([tl_x + m, tl_y + m, tl_x + m + thick, tl_y + m + size], fill=marker_color)
        draw.rectangle([tl_x + m, tl_y + m + size - thick, tl_x + m + size, tl_y + m + size], fill=marker_color)

    else:  # detailed style
        # Create background
        bg_color = tuple(random.randint(40, 100) for _ in range(3))
        img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), bg_color)
        draw = ImageDraw.Draw(img)

        # Draw the paper as a filled polygon
        paper_color = tuple(random.randint(220, 255) for _ in range(3))
        polygon = [(corners[i, 0], corners[i, 1]) for i in range(4)]
        draw.polygon(polygon, fill=paper_color, outline=(100, 100, 100))

        # Add registration marks at corners
        mark_color = (0, 0, 0)
        mark_size = 15
        for i, (x, y) in enumerate(corners):
            draw.line([(x - mark_size, y), (x + mark_size, y)], fill=mark_color, width=3)
            draw.line([(x, y - mark_size), (x, y + mark_size)], fill=mark_color, width=3)
            labels = ['TL', 'TR', 'BR', 'BL']
            draw.text((x + 10, y + 10), labels[i], fill=mark_color)

        # Add orientation marker
        tl_x, tl_y = corners[0]
        draw.rectangle([tl_x + 20, tl_y + 20, tl_x + 60, tl_y + 30], fill=(50, 50, 50))
        draw.rectangle([tl_x + 20, tl_y + 20, tl_x + 30, tl_y + 70], fill=(50, 50, 50))

        # Add text-like lines
        min_x, max_x = corners[:, 0].min(), corners[:, 0].max()
        min_y, max_y = corners[:, 1].min(), corners[:, 1].max()
        for y_offset in range(80, int(max_y - min_y) - 40, 25):
            line_width = random.uniform(0.4, 0.8) * (max_x - min_x - 80)
            y_pos = min_y + y_offset
            x_start = min_x + 40
            draw.rectangle([x_start, y_pos, x_start + line_width, y_pos + 8], fill=(80, 80, 80))

    return img, corners


def preprocess_for_inference(img):
    """Convert PIL image to tensor for inference."""
    # Convert to grayscale
    gray = img.convert('L')
    # Resize to training size
    resized = gray.resize((TRAIN_WIDTH, TRAIN_HEIGHT), Image.BILINEAR)
    # Convert to numpy and normalize
    arr = np.array(resized, dtype=np.float32) / 255.0
    # Add batch and channel dimensions
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return tensor


def denormalize_corners(normalized_corners):
    """Convert normalized [-1, 1] corners back to pixel coordinates."""
    corners = normalized_corners.copy()
    corners[:, 0] = corners[:, 0] * (IMG_WIDTH / 2) + (IMG_WIDTH / 2)
    corners[:, 1] = corners[:, 1] * (IMG_HEIGHT / 2) + (IMG_HEIGHT / 2)
    return corners


def dewarp_image(img, predicted_corners):
    """
    Use predicted corners to dewarp the image back to a rectangle.
    """
    # Convert PIL to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Source points (predicted corners): TL, TR, BR, BL
    src_pts = predicted_corners.astype(np.float32)

    # Destination rectangle (standard aspect ratio for paper ~11:8.5)
    aspect = 11.0 / 8.5
    out_height = 600
    out_width = int(out_height * aspect)

    dst_pts = np.array([
        [0, 0],
        [out_width - 1, 0],
        [out_width - 1, out_height - 1],
        [0, out_height - 1]
    ], dtype=np.float32)

    # Compute homography
    H, _ = cv2.findHomography(src_pts, dst_pts)

    # Warp the image
    dewarped = cv2.warpPerspective(img_cv, H, (out_width, out_height))

    return Image.fromarray(cv2.cvtColor(dewarped, cv2.COLOR_BGR2RGB))


def draw_corners_on_image(img, corners, color=(255, 0, 0), label=""):
    """Draw corner markers on image."""
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)

    mark_size = 20
    for i, (x, y) in enumerate(corners):
        # Draw X marker
        draw.line([(x - mark_size, y - mark_size), (x + mark_size, y + mark_size)], fill=color, width=4)
        draw.line([(x - mark_size, y + mark_size), (x + mark_size, y - mark_size)], fill=color, width=4)

    # Draw polygon connecting corners
    polygon = [(corners[i, 0], corners[i, 1]) for i in range(4)]
    polygon.append(polygon[0])  # Close the polygon
    draw.line(polygon, fill=color, width=3)

    # Add label
    if label:
        draw.text((10, 10), label, fill=color)

    return img_copy


def load_ensemble(model_dir, pattern="model_seed*.pth"):
    """Load multiple models for ensemble prediction."""
    import glob
    model_paths = sorted(glob.glob(os.path.join(model_dir, pattern)))

    if not model_paths:
        raise FileNotFoundError(f"No models found matching {model_dir}/{pattern}")

    models = []
    for path in model_paths:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        # Get architecture params
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            img_width = checkpoint.get('img_width', TRAIN_WIDTH)
            img_height = checkpoint.get('img_height', TRAIN_HEIGHT)
            base_filters = checkpoint.get('base_filters', 32)
        else:
            state_dict = checkpoint
            img_width, img_height, base_filters = TRAIN_WIDTH, TRAIN_HEIGHT, 32

        model = CornersCNN(input_height=img_height, input_width=img_width, base_filters=base_filters)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
        print(f"  Loaded {os.path.basename(path)}")

    return models


def ensemble_predict(models, input_tensor):
    """Run all models and average their predictions."""
    predictions = []
    with torch.no_grad():
        for model in models:
            output = model(input_tensor)
            predictions.append(output)

    # Stack and average
    stacked = torch.stack(predictions, dim=0)
    averaged = torch.mean(stacked, dim=0)
    return averaged


def main():
    parser = argparse.ArgumentParser(description="Test CNN inference on synthetic images")
    parser.add_argument("--model", type=str, default="aws_models/model_averaged.pth",
                        help="Path to trained model (or directory for ensemble)")
    parser.add_argument("--samples", type=int, default=5, help="Number of test samples")
    parser.add_argument("--output_dir", type=str, default="inference_test",
                        help="Output directory for results")
    parser.add_argument("--style", type=str, default="training", choices=["training", "detailed"],
                        help="Image style: 'training' (simple quads) or 'detailed' (with marks)")
    parser.add_argument("--ensemble", action="store_true",
                        help="Use ensemble prediction (average outputs from multiple models)")
    parser.add_argument("--ensemble_pattern", type=str, default="model_seed*.pth",
                        help="Glob pattern for ensemble models")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model(s)
    ensemble_models = None
    model = None

    if args.ensemble:
        print(f"Loading ensemble from: {args.model}")
        ensemble_models = load_ensemble(args.model, args.ensemble_pattern)
        print(f"Loaded {len(ensemble_models)} models for ensemble prediction")
    else:
        print(f"Loading model: {args.model}")
        checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)

        # Check if this is a full checkpoint or just a state_dict
        if 'model_state_dict' in checkpoint:
            # Full checkpoint format
            img_width = checkpoint.get('img_width', TRAIN_WIDTH)
            img_height = checkpoint.get('img_height', TRAIN_HEIGHT)
            base_filters = checkpoint.get('base_filters', 32)
            state_dict = checkpoint['model_state_dict']
        else:
            # Raw state_dict (e.g., averaged model)
            img_width = TRAIN_WIDTH
            img_height = TRAIN_HEIGHT
            base_filters = 32
            state_dict = checkpoint

        print(f"  Image size: {img_width}x{img_height}")
        print(f"  Base filters: {base_filters}")

        # Create model
        model = CornersCNN(input_height=img_height, input_width=img_width, base_filters=base_filters)
        model.load_state_dict(state_dict)
        model.eval()

    print(f"\nGenerating {args.samples} test images...")

    total_error = 0.0

    for i in range(args.samples):
        print(f"\n--- Sample {i+1}/{args.samples} ---")

        # Generate test image
        img, gt_corners = generate_test_image(style=args.style)

        # Run inference
        input_tensor = preprocess_for_inference(img)
        if ensemble_models:
            output = ensemble_predict(ensemble_models, input_tensor)
        else:
            with torch.no_grad():
                output = model(input_tensor)

        # Convert output to corners
        pred_normalized = output.numpy().reshape(4, 2)
        pred_corners = denormalize_corners(pred_normalized)

        # Calculate error
        error = np.sqrt(((pred_corners - gt_corners) ** 2).sum(axis=1)).mean()
        total_error += error

        print(f"  Ground truth corners:")
        for j, (x, y) in enumerate(gt_corners):
            print(f"    {['TL', 'TR', 'BR', 'BL'][j]}: ({x:.1f}, {y:.1f})")

        print(f"  Predicted corners:")
        for j, (x, y) in enumerate(pred_corners):
            print(f"    {['TL', 'TR', 'BR', 'BL'][j]}: ({x:.1f}, {y:.1f})")

        print(f"  Average corner error: {error:.1f} pixels")

        # Create visualization
        # 1. Original with ground truth (green)
        img_gt = draw_corners_on_image(img, gt_corners, color=(0, 255, 0), label="Ground Truth")

        # 2. Original with prediction (red)
        img_pred = draw_corners_on_image(img, pred_corners, color=(255, 0, 0), label="Predicted")

        # 3. Both overlaid
        img_both = draw_corners_on_image(img, gt_corners, color=(0, 255, 0), label="Green=GT, Red=Pred")
        img_both = draw_corners_on_image(img_both, pred_corners, color=(255, 0, 0), label="")

        # 4. Dewarped using predicted corners
        img_dewarped = dewarp_image(img, pred_corners)

        # Save images
        img.save(os.path.join(args.output_dir, f"sample_{i+1}_original.jpg"))
        img_gt.save(os.path.join(args.output_dir, f"sample_{i+1}_ground_truth.jpg"))
        img_pred.save(os.path.join(args.output_dir, f"sample_{i+1}_predicted.jpg"))
        img_both.save(os.path.join(args.output_dir, f"sample_{i+1}_comparison.jpg"))
        img_dewarped.save(os.path.join(args.output_dir, f"sample_{i+1}_dewarped.jpg"))

        print(f"  Saved to {args.output_dir}/sample_{i+1}_*.jpg")

    avg_error = total_error / args.samples
    print(f"\n{'='*50}")
    print(f"Average corner error across {args.samples} samples: {avg_error:.1f} pixels")
    print(f"Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
