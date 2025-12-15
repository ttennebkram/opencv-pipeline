#!/usr/bin/env python3
"""
generate_synthetic.py

Generate synthetic training images from corner labels only.
This creates training data without needing the original 125GB+ images.

The idea: We only need the corner positions to train corner detection.
The image content can be synthetic - random gradients, noise, shapes.
The network learns to find corners regardless of content.

Usage:
    # Generate 10,000 samples
    python3 generate_synthetic.py --output training_data --samples 10000

    # Generate from existing labels (use same corners but synthetic content)
    python3 generate_synthetic.py --labels_dir /path/to/labels --output training_data
"""

import argparse
import os
import sys
import json
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Image dimensions (matching Java generator)
IMG_WIDTH = 1920
IMG_HEIGHT = 1080

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)


def random_corners():
    """
    Generate random corner positions for a distorted quadrilateral.
    Simulates a paper/document viewed from an angle.
    """
    # Start with a rectangle and perturb corners
    # Base rectangle centered in frame, varying size
    scale = random.uniform(0.2, 0.8)  # 20-80% of frame

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

    # Apply random perspective-like distortion
    for i in range(4):
        # Perturb each corner by up to 20% of the rectangle size
        corners[i, 0] += random.gauss(0, w * 0.15)
        corners[i, 1] += random.gauss(0, h * 0.15)

        # Clamp to image bounds with margin
        margin = 20
        corners[i, 0] = max(margin, min(IMG_WIDTH - margin, corners[i, 0]))
        corners[i, 1] = max(margin, min(IMG_HEIGHT - margin, corners[i, 1]))

    return corners


def add_orientation_marker(img, draw):
    """
    Add an asymmetric orientation marker to break 180-degree rotation ambiguity.
    Always adds something in the top-left that's distinct from other corners.
    """
    width, height = img.size
    marker_type = random.choice(['L_shape', 'corner_box', 'header_line', 'logo_circle'])

    # Dark color for visibility
    marker_color = tuple(random.randint(20, 80) for _ in range(3))

    if marker_type == 'L_shape':
        # L-shaped marker in top-left
        margin = int(width * 0.05)
        size = int(min(width, height) * 0.15)
        thickness = max(3, size // 10)
        # Vertical bar
        draw.rectangle([margin, margin, margin + thickness, margin + size], fill=marker_color)
        # Horizontal bar
        draw.rectangle([margin, margin + size - thickness, margin + size, margin + size], fill=marker_color)

    elif marker_type == 'corner_box':
        # Filled rectangle in top-left corner only
        margin = int(width * 0.03)
        box_w = int(width * 0.12)
        box_h = int(height * 0.08)
        draw.rectangle([margin, margin, margin + box_w, margin + box_h], fill=marker_color)

    elif marker_type == 'header_line':
        # Thick header line at top (like a document header)
        margin = int(width * 0.05)
        line_height = int(height * 0.04)
        line_width = int(width * random.uniform(0.3, 0.7))
        draw.rectangle([margin, margin, margin + line_width, margin + line_height], fill=marker_color)

    elif marker_type == 'logo_circle':
        # Small circle in top-left (like a logo)
        margin = int(width * 0.05)
        radius = int(min(width, height) * 0.06)
        cx, cy = margin + radius, margin + radius
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=marker_color)


def generate_synthetic_content(width, height):
    """
    Generate random synthetic content that could be inside a document.
    Returns a PIL Image.

    IMPORTANT: Always includes an asymmetric orientation marker in the top-left
    to disambiguate 0° from 180° rotation.
    """
    content_type = random.choice(['gradient', 'noise', 'text_like', 'mixed', 'solid'])

    img = None
    draw = None

    if content_type == 'gradient':
        # Random gradient
        img = Image.new('RGB', (width, height))
        pixels = img.load()

        # Random gradient direction and colors
        c1 = tuple(random.randint(50, 255) for _ in range(3))
        c2 = tuple(random.randint(50, 255) for _ in range(3))

        horizontal = random.random() > 0.5
        for y in range(height):
            for x in range(width):
                t = x / width if horizontal else y / height
                r = int(c1[0] * (1-t) + c2[0] * t)
                g = int(c1[1] * (1-t) + c2[1] * t)
                b = int(c1[2] * (1-t) + c2[2] * t)
                pixels[x, y] = (r, g, b)
        draw = ImageDraw.Draw(img)

    elif content_type == 'noise':
        # Random noise pattern
        arr = np.random.randint(100, 255, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)

    elif content_type == 'text_like':
        # Simulate text with horizontal lines
        img = Image.new('RGB', (width, height), color=(250, 250, 250))
        draw = ImageDraw.Draw(img)

        line_height = random.randint(15, 30)
        margin = random.randint(20, 50)

        for y in range(margin, height - margin, line_height):
            # Random line length (like text paragraphs)
            line_width = random.uniform(0.3, 0.95) * (width - 2*margin)
            gray = random.randint(20, 80)
            draw.rectangle(
                [margin, y, margin + line_width, y + line_height//3],
                fill=(gray, gray, gray)
            )

    elif content_type == 'mixed':
        # Mix of shapes and patterns
        img = Image.new('RGB', (width, height), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)

        # Random rectangles and lines
        for _ in range(random.randint(5, 20)):
            x1 = random.randint(0, width - 1)
            y1 = random.randint(0, height - 1)
            x2 = random.randint(0, width - 1)
            y2 = random.randint(0, height - 1)
            # Ensure proper ordering for rectangles
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            color = tuple(random.randint(50, 200) for _ in range(3))

            if random.random() > 0.5:
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            else:
                draw.line([x1, y1, x2, y2], fill=color, width=random.randint(1, 5))

    else:  # solid
        # Solid color with slight variation
        base = tuple(random.randint(180, 255) for _ in range(3))
        img = Image.new('RGB', (width, height), color=base)
        draw = ImageDraw.Draw(img)

    # Always add orientation marker to disambiguate 0° from 180° rotation
    add_orientation_marker(img, draw)

    return img


def apply_perspective_transform(content_img, corners, output_size):
    """
    Apply perspective transform to place content at the given corners.
    Uses PIL's quad transform.

    corners: (4, 2) array of [x, y] positions for TL, TR, BR, BL
    """
    # Create output image (background)
    background = random_background(output_size[0], output_size[1])

    # Create a mask for the quadrilateral
    mask = Image.new('L', output_size, 0)
    mask_draw = ImageDraw.Draw(mask)
    polygon = [(corners[i, 0], corners[i, 1]) for i in range(4)]
    mask_draw.polygon(polygon, fill=255)

    # Get bounding box
    min_x = max(0, int(corners[:, 0].min()))
    max_x = min(output_size[0], int(corners[:, 0].max()) + 1)
    min_y = max(0, int(corners[:, 1].min()))
    max_y = min(output_size[1], int(corners[:, 1].max()) + 1)

    if max_x <= min_x or max_y <= min_y:
        return background

    # Simple approach: resize content to fit in bounding box, then paste with mask
    # This isn't true perspective, but it's fast and the network should still learn corner detection
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y

    content_resized = content_img.resize((bbox_w, bbox_h), Image.BILINEAR)

    # Create the transformed content
    transformed = Image.new('RGB', output_size, (0, 0, 0))
    transformed.paste(content_resized, (min_x, min_y))

    # Composite with background using mask
    result = Image.composite(transformed, background, mask)

    # Add some noise/artifacts for realism
    if random.random() > 0.5:
        result = result.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1)))

    return result


def random_background(width, height):
    """Generate a random background (desk/table surface)."""
    bg_type = random.choice(['solid', 'gradient', 'noisy'])

    if bg_type == 'solid':
        color = tuple(random.randint(30, 150) for _ in range(3))
        return Image.new('RGB', (width, height), color)
    elif bg_type == 'gradient':
        img = Image.new('RGB', (width, height))
        pixels = img.load()
        c1 = tuple(random.randint(30, 100) for _ in range(3))
        c2 = tuple(random.randint(30, 100) for _ in range(3))
        for y in range(height):
            t = y / height
            for x in range(width):
                r = int(c1[0] * (1-t) + c2[0] * t)
                g = int(c1[1] * (1-t) + c2[1] * t)
                b = int(c1[2] * (1-t) + c2[2] * t)
                pixels[x, y] = (r, g, b)
        return img
    else:
        arr = np.random.randint(30, 120, (height, width, 3), dtype=np.uint8)
        return Image.fromarray(arr)


def generate_sample(sample_id):
    """Generate a single training sample with synthetic content."""
    # Generate random corners
    corners = random_corners()

    # Generate synthetic document content
    content = generate_synthetic_content(800, 600)  # Arbitrary internal resolution

    # Apply perspective to place content at corners
    result = apply_perspective_transform(content, corners, (IMG_WIDTH, IMG_HEIGHT))

    return result, corners


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--output", type=str, default="synthetic_training_data",
                        help="Output directory")
    parser.add_argument("--samples", type=int, default=10000,
                        help="Number of samples to generate")
    parser.add_argument("--labels_dir", type=str, default=None,
                        help="Use corners from existing labels (optional)")
    parser.add_argument("--img_size", type=str, default="256,144",
                        help="Output image size (width,height)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Starting index for sample numbering (for parallel generation)")
    args = parser.parse_args()

    # Combine seed with start_index for parallel generation with different random sequences
    effective_seed = args.seed + args.start_index
    random.seed(effective_seed)
    np.random.seed(effective_seed)

    output_w, output_h = map(int, args.img_size.split(','))

    # Create output directories
    images_dir = os.path.join(args.output, "images")
    labels_dir = os.path.join(args.output, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    print(f"=== Synthetic Training Data Generator ===")
    print(f"Output: {args.output}")
    print(f"Samples: {args.samples}")
    print(f"Image size: {output_w}x{output_h}")
    print()

    start_idx = args.start_index
    for i in range(args.samples):
        sample_id = start_idx + i
        # Generate sample
        img, corners = generate_sample(sample_id)

        # Resize to target size
        img_small = img.resize((output_w, output_h), Image.BILINEAR)

        # Save image
        img_path = os.path.join(images_dir, f"sample_{sample_id:06d}_normal.jpg")
        img_small.save(img_path, "JPEG", quality=90)

        # Save label
        label = {
            "id": f"sample_{sample_id:06d}",
            "corners": corners.flatten().tolist(),
        }
        label_path = os.path.join(labels_dir, f"sample_{sample_id:06d}.json")
        with open(label_path, 'w') as f:
            json.dump(label, f)

        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{args.samples} samples (IDs {start_idx}-{start_idx + i})")

    print()
    print(f"Done! Generated {args.samples} samples in {args.output}/")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {labels_dir}")


if __name__ == "__main__":
    main()
