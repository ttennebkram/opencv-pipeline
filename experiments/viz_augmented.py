#!/usr/bin/env python3
"""Visualize augmented images with ground truth corners."""
import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_sample(data_dir, idx=0):
    """Load a sample image and corners."""
    data_path = Path(data_dir)
    imgs = sorted((data_path / "images").glob("*.jpg"))
    if idx >= len(imgs):
        idx = 0
    img_path = imgs[idx]
    label_path = data_path / "labels" / (img_path.stem + ".json")

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (240, 135), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0

    with open(label_path) as f:
        lbl = json.load(f)
    corners = np.array(lbl["corners"], dtype=np.float32).reshape(4, 2)

    return img, corners, img_path.name

def apply_augmentation(img, corners, h_flip=False, v_flip=False, brightness=1.0, contrast=1.0):
    """Apply augmentation and transform corners accordingly."""
    img = img.copy()
    corners = corners.copy()

    if h_flip:
        img = np.fliplr(img).copy()
        corners = corners[[1, 0, 3, 2], :]  # Swap TL↔TR, BL↔BR
        corners[:, 0] = -corners[:, 0]  # Negate x

    if v_flip:
        img = np.flipud(img).copy()
        corners = corners[[3, 2, 1, 0], :]  # Swap TL↔BL, TR↔BR
        corners[:, 1] = -corners[:, 1]  # Negate y

    img = np.clip(img * brightness, 0, 1)
    mean = img.mean()
    img = np.clip((img - mean) * contrast + mean, 0, 1)

    return img, corners

def corners_to_pixels(corners, width=240, height=135):
    """Convert normalized [-1,1] corners to pixel coordinates."""
    pixels = corners.copy()
    pixels[:, 0] = (corners[:, 0] + 1) * width / 2
    pixels[:, 1] = (corners[:, 1] + 1) * height / 2
    return pixels

def draw_corners(ax, img, corners, title=""):
    """Draw image with corner overlay."""
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    pixels = corners_to_pixels(corners, img.shape[1], img.shape[0])

    # Draw quadrilateral
    pts = np.vstack([pixels, pixels[0]])  # Close the polygon
    ax.plot(pts[:, 0], pts[:, 1], 'g-', linewidth=2, label='Ground Truth')

    # Draw corner points with labels
    labels = ['TL', 'TR', 'BR', 'BL']
    colors = ['red', 'blue', 'green', 'orange']
    for i, (pt, lbl, c) in enumerate(zip(pixels, labels, colors)):
        ax.scatter(pt[0], pt[1], c=c, s=100, zorder=5)
        ax.annotate(lbl, (pt[0]+5, pt[1]+5), color=c, fontsize=10, fontweight='bold')

    ax.set_title(title)
    ax.axis('off')

def main():
    data_dir = "/Volumes/SamsungBlue/ml-training/wiki_training_v2"

    # Try multiple samples
    for sample_idx in [0, 1, 5, 10]:
        try:
            img, corners, name = load_sample(data_dir, sample_idx)
        except:
            print(f"Could not load sample {sample_idx}")
            continue

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Augmentation Test: {name}', fontsize=14)

        # Original
        draw_corners(axes[0, 0], img, corners, "Original")

        # Horizontal flip
        img_h, corners_h = apply_augmentation(img, corners, h_flip=True)
        draw_corners(axes[0, 1], img_h, corners_h, "H-Flip")

        # Vertical flip
        img_v, corners_v = apply_augmentation(img, corners, v_flip=True)
        draw_corners(axes[0, 2], img_v, corners_v, "V-Flip")

        # Both flips
        img_hv, corners_hv = apply_augmentation(img, corners, h_flip=True, v_flip=True)
        draw_corners(axes[1, 0], img_hv, corners_hv, "H+V Flip")

        # Brightness up
        img_b, corners_b = apply_augmentation(img, corners, brightness=1.15)
        draw_corners(axes[1, 1], img_b, corners_b, "Brightness +15%")

        # Contrast up
        img_c, corners_c = apply_augmentation(img, corners, contrast=1.15)
        draw_corners(axes[1, 2], img_c, corners_c, "Contrast +15%")

        plt.tight_layout()
        output_path = f'/tmp/augment_test_{sample_idx}.png'
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}", flush=True)
        plt.close()

    print("Done! Check /tmp/augment_test_*.png", flush=True)

if __name__ == "__main__":
    main()
