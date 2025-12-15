#!/usr/bin/env python3
"""
Generate training data from Wikipedia ZIM archive.

Creates 8.5x11 renders of Wikipedia pages with:
1. One straight-on view (ground truth corners at edges)
2. N random perspective transforms with corner labels

Features:
- Images from ZIM served via local HTTP server
- 50% dark mode / 50% light mode
- Random neutral background colors
- Random page scrolling (0-3 pages)

Requirements:
    pip install libzim playwright opencv-python numpy
    playwright install chromium

Usage:
    python3 generate_wiki_training.py --pages 10 --transforms 5 --output ./wiki_training
    python3 generate_wiki_training.py --pages 100 --transforms 10 --output /Volumes/SamsungBlue/ml-training/wiki_training
"""

import argparse
import base64
import json
import os
import random
import sys
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import cv2
import numpy as np
from libzim.reader import Archive
from playwright.sync_api import sync_playwright

# Default paths
LOCAL_ZIM = "/Volumes/DevBlueB/kiwix/wikipedia_en_top_maxi_2025-09.zim"
AWS_ZIM = "/mnt/data/home/ubuntu/wikipedia_en_top_maxi_2025-09.zim"

# Page dimensions (8.5x11 inches at 300 DPI, but we render at screen resolution)
RENDER_WIDTH = 850   # 8.5 inches * 100 pixels/inch
RENDER_HEIGHT = 1100  # 11 inches * 100 pixels/inch

# Output dimensions (1080p with document centered)
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080

# Neutral background colors (RGB tuples)
BACKGROUND_COLORS = [
    (128, 128, 128),  # Gray
    (140, 130, 120),  # Warm gray
    (120, 125, 135),  # Cool gray
    (150, 145, 140),  # Light warm gray
    (110, 115, 120),  # Dark cool gray
    (160, 155, 150),  # Pale beige
    (100, 100, 105),  # Dark gray-blue
    (135, 130, 125),  # Taupe
    (145, 140, 130),  # Sand
    (115, 120, 125),  # Slate
    (85, 80, 75),     # Dark brown-gray
    (170, 165, 160),  # Light beige
    (95, 90, 85),     # Charcoal
    (125, 120, 115),  # Medium gray-brown
    (155, 150, 145),  # Light gray
]


def get_zim_path():
    """Find the ZIM file on local or AWS."""
    if os.path.exists(LOCAL_ZIM):
        return LOCAL_ZIM
    elif os.path.exists(AWS_ZIM):
        return AWS_ZIM
    else:
        raise FileNotFoundError(f"ZIM file not found at {LOCAL_ZIM} or {AWS_ZIM}")


def get_article_paths(zim):
    """Get list of article paths from ZIM archive."""
    paths = []
    image_exts = ['.png', '.jpg', '.jpeg', '.svg', '.gif', '.webp', '.ico']
    skip_exts = ['.css', '.js', '.woff', '.woff2', '.ttf', '.eot']

    for i in range(zim.entry_count):
        entry = zim._get_entry_by_id(i)
        path = entry.path
        path_lower = path.lower()

        # Skip images
        if any(ext in path_lower for ext in image_exts):
            continue
        # Skip CSS/JS/fonts
        if any(ext in path_lower for ext in skip_exts):
            continue
        # Skip special pages
        if path.startswith('-/') or path.startswith('_/'):
            continue
        # Skip very short paths (likely not real articles)
        if len(path) < 3:
            continue

        paths.append(path)
    return paths


def get_random_article_html(zim, article_paths):
    """Get HTML content of a random article."""
    path = random.choice(article_paths)
    try:
        entry = zim.get_entry_by_path(path)
        content = bytes(entry.get_item().content).decode('utf-8', errors='ignore')
        return path, content
    except Exception as e:
        print(f"Error reading {path}: {e}", flush=True)
        return None, None


def get_resource_as_data_uri(zim, path):
    """Get a resource from ZIM as a data URI for embedding."""
    try:
        entry = zim.get_entry_by_path(path)
        content = bytes(entry.get_item().content)

        # Determine MIME type
        path_lower = path.lower()
        if path_lower.endswith('.png'):
            mime = 'image/png'
        elif path_lower.endswith('.jpg') or path_lower.endswith('.jpeg'):
            mime = 'image/jpeg'
        elif path_lower.endswith('.gif'):
            mime = 'image/gif'
        elif path_lower.endswith('.svg'):
            mime = 'image/svg+xml'
        elif path_lower.endswith('.webp'):
            mime = 'image/webp'
        else:
            mime = 'application/octet-stream'

        b64 = base64.b64encode(content).decode('ascii')
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


def embed_images_in_html(html, zim):
    """Replace image src attributes with embedded data URIs."""
    import re

    def replace_src(match):
        src = match.group(1)
        # Remove leading ./ or /
        clean_path = src.lstrip('./')

        # Try to get the image from ZIM
        data_uri = get_resource_as_data_uri(zim, clean_path)
        if data_uri:
            return f'src="{data_uri}"'

        # Also try with common prefixes
        for prefix in ['', 'I/', 'images/']:
            data_uri = get_resource_as_data_uri(zim, prefix + clean_path)
            if data_uri:
                return f'src="{data_uri}"'

        # Return original if not found
        return match.group(0)

    # Replace src="..." patterns
    html = re.sub(r'src="([^"]+)"', replace_src, html)
    html = re.sub(r"src='([^']+)'", replace_src, html)

    return html


def render_page_to_image(page, html_content, zim, scroll_pages=0, dark_mode=False):
    """
    Render HTML to an image using Playwright.

    Args:
        page: Playwright page object
        html_content: HTML string to render
        zim: ZIM archive for loading images
        scroll_pages: Number of page-heights to scroll down (0-3)
        dark_mode: Whether to use dark mode styling

    Returns:
        numpy array (BGR image)
    """
    # Set viewport to 8.5x11 proportions
    page.set_viewport_size({"width": RENDER_WIDTH, "height": RENDER_HEIGHT})

    # Embed images as data URIs
    html_with_images = embed_images_in_html(html_content, zim)

    # Dark mode styles
    if dark_mode:
        bg_color = "#1a1a1a"
        text_color = "#e0e0e0"
        link_color = "#6db3f2"
    else:
        bg_color = "white"
        text_color = "black"
        link_color = "#0645ad"

    # Create styled HTML
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: Georgia, 'Times New Roman', serif;
                font-size: 14px;
                line-height: 1.6;
                margin: 40px;
                background: {bg_color};
                color: {text_color};
            }}
            a {{
                color: {link_color};
            }}
            img {{
                max-width: 100%;
                height: auto;
            }}
            table {{
                border-collapse: collapse;
                margin: 10px 0;
            }}
            th, td {{
                border: 1px solid {text_color};
                padding: 5px;
            }}
            /* Hide Wikipedia navigation elements */
            .mw-jump-link, .navbox, .catlinks, .mw-editsection,
            .sistersitebox, .side-box, .metadata, .noprint {{
                display: none !important;
            }}
        </style>
    </head>
    <body>
        {html_with_images}
    </body>
    </html>
    """

    # Load the content
    page.set_content(styled_html, wait_until='load')
    page.wait_for_timeout(500)  # Wait for images to render

    # Scroll if requested
    if scroll_pages > 0:
        scroll_amount = int(scroll_pages * RENDER_HEIGHT)
        page.evaluate(f"window.scrollTo(0, {scroll_amount})")
        page.wait_for_timeout(100)

    # Take screenshot
    screenshot_bytes = page.screenshot(type='png')

    # Convert to numpy array
    nparr = np.frombuffer(screenshot_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return img


def get_random_background_color():
    """Get a random neutral background color as BGR tuple."""
    rgb = random.choice(BACKGROUND_COLORS)
    # Convert RGB to BGR for OpenCV
    return (rgb[2], rgb[1], rgb[0])


def create_straight_on_view(page_img, bg_color=None):
    """
    Create a straight-on view of the page on a 1080p canvas.

    Returns:
        (output_image, corners) where corners are the 4 corner coordinates
    """
    if bg_color is None:
        bg_color = get_random_background_color()

    h, w = page_img.shape[:2]

    # Calculate scale to fit page in output with some margin
    margin = 50
    max_width = OUTPUT_WIDTH - 2 * margin
    max_height = OUTPUT_HEIGHT - 2 * margin

    scale = min(max_width / w, max_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize page
    resized = cv2.resize(page_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create output canvas with random background color
    output = np.full((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), bg_color, dtype=np.uint8)

    # Center the page
    x_offset = (OUTPUT_WIDTH - new_w) // 2
    y_offset = (OUTPUT_HEIGHT - new_h) // 2

    output[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    # Calculate corners (top-left, top-right, bottom-right, bottom-left)
    corners = [
        [x_offset, y_offset],
        [x_offset + new_w, y_offset],
        [x_offset + new_w, y_offset + new_h],
        [x_offset, y_offset + new_h]
    ]

    return output, corners


def apply_random_homography(page_img, bg_color=None):
    """
    Apply a random perspective transform to the page image.

    Returns:
        (output_image, corners) where corners are the transformed corner coordinates
    """
    if bg_color is None:
        bg_color = get_random_background_color()

    h, w = page_img.shape[:2]

    # Original corners
    src_corners = np.float32([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ])

    # Random transform parameters
    # Scale: 20-80% of output size
    scale = random.uniform(0.2, 0.8)

    # Random position (can be partially off-screen)
    center_x = random.uniform(0.2, 0.8) * OUTPUT_WIDTH
    center_y = random.uniform(0.2, 0.8) * OUTPUT_HEIGHT

    # Calculate base size
    base_w = w * scale * (OUTPUT_HEIGHT / h)
    base_h = h * scale * (OUTPUT_HEIGHT / h)

    # Clamp to reasonable size
    base_w = min(base_w, OUTPUT_WIDTH * 0.9)
    base_h = min(base_h, OUTPUT_HEIGHT * 0.9)

    # Random rotation (full 360)
    angle = random.uniform(0, 360)
    angle_rad = np.radians(angle)

    # Random perspective distortion (tilt) - more oblique angles
    tilt_x = random.uniform(-0.5, 0.5)
    tilt_y = random.uniform(-0.5, 0.5)

    # Build destination corners with perspective
    half_w = base_w / 2
    half_h = base_h / 2

    # Start with rectangle centered at origin
    dst_corners = np.float32([
        [-half_w, -half_h],
        [half_w, -half_h],
        [half_w, half_h],
        [-half_w, half_h]
    ])

    # Apply perspective distortion (trapezoid effect)
    perspective_factor_y = 1 + tilt_y
    perspective_factor_x = 1 + tilt_x

    dst_corners[0][0] *= perspective_factor_x
    dst_corners[1][0] *= (2 - perspective_factor_x)
    dst_corners[0][1] *= perspective_factor_y
    dst_corners[1][1] *= perspective_factor_y
    dst_corners[2][1] *= (2 - perspective_factor_y)
    dst_corners[3][1] *= (2 - perspective_factor_y)

    # Apply rotation
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    for i in range(4):
        dst_corners[i] = rotation_matrix @ dst_corners[i]

    # Translate to center position
    dst_corners[:, 0] += center_x
    dst_corners[:, 1] += center_y

    # Compute homography
    H, _ = cv2.findHomography(src_corners, dst_corners)

    # Apply transform with random background color
    output = np.full((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), bg_color, dtype=np.uint8)
    cv2.warpPerspective(page_img, H, (OUTPUT_WIDTH, OUTPUT_HEIGHT),
                        dst=output, borderMode=cv2.BORDER_TRANSPARENT)

    corners = dst_corners.tolist()
    return output, corners


def normalize_corners(corners, width=OUTPUT_WIDTH, height=OUTPUT_HEIGHT):
    """Normalize corners to [-1, 1] range."""
    normalized = []
    for x, y in corners:
        nx = (x / width) * 2 - 1
        ny = (y / height) * 2 - 1
        normalized.append([nx, ny])
    return normalized


def save_sample(output_dir, sample_id, image, corners, article_path,
                is_straight=False, dark_mode=False):
    """Save image and metadata."""
    # Save image with descriptive suffix
    if is_straight:
        img_filename = f"sample_{sample_id:06d}_straight.jpg"
    else:
        img_filename = f"sample_{sample_id:06d}_rotated.jpg"
    img_path = os.path.join(output_dir, "images", img_filename)
    cv2.imwrite(img_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Normalize corners for training
    norm_corners = normalize_corners(corners)

    # Flatten corners for training: [x0, y0, x1, y1, x2, y2, x3, y3]
    corners_flat = [coord for point in norm_corners for coord in point]

    # Save metadata
    metadata = {
        "id": sample_id,
        "image": img_filename,
        "article": article_path,
        "is_straight": is_straight,
        "dark_mode": dark_mode,
        "corners": corners_flat,
        "corners_pixels": [coord for point in corners for coord in point]
    }

    if is_straight:
        json_filename = f"sample_{sample_id:06d}_straight.json"
    else:
        json_filename = f"sample_{sample_id:06d}_rotated.json"
    json_path = os.path.join(output_dir, "labels", json_filename)
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return img_path


def main():
    parser = argparse.ArgumentParser(description='Generate Wikipedia training data')
    parser.add_argument('--zim', type=str, help='Path to ZIM file (auto-detected if not specified)')
    parser.add_argument('--pages', type=int, default=10, help='Number of pages to render')
    parser.add_argument('--transforms', type=int, default=5, help='Homographic transforms per page')
    parser.add_argument('--output', type=str, default='./wiki_training', help='Output directory')
    parser.add_argument('--start_id', type=int, default=0, help='Starting sample ID')
    args = parser.parse_args()

    # Find ZIM file
    zim_path = args.zim if args.zim else get_zim_path()
    print(f"Using ZIM file: {zim_path}", flush=True)

    # Create output directories
    os.makedirs(os.path.join(args.output, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "labels"), exist_ok=True)

    # Open ZIM archive
    print("Opening ZIM archive...", flush=True)
    zim = Archive(zim_path)
    print(f"ZIM has {zim.entry_count} entries", flush=True)

    # Get article paths
    print("Indexing articles...", flush=True)
    article_paths = get_article_paths(zim)
    print(f"Found {len(article_paths)} articles", flush=True)

    sample_id = args.start_id

    with sync_playwright() as p:
        print("Launching browser...", flush=True)
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        for page_num in range(args.pages):
            # Get random article
            article_path, html = get_random_article_html(zim, article_paths)
            if html is None:
                continue

            # Random scroll (0-3 pages)
            scroll_pages = random.uniform(0, 3)

            # Random dark mode (50% chance)
            dark_mode = random.random() < 0.5

            # Random background color (same for all transforms of this page)
            bg_color = get_random_background_color()

            mode_str = "dark" if dark_mode else "light"
            print(f"Page {page_num + 1}/{args.pages}: {article_path} ({mode_str}, scroll: {scroll_pages:.1f})", flush=True)

            try:
                # Render page
                page_img = render_page_to_image(page, html, zim, scroll_pages, dark_mode)

                # Create straight-on view
                straight_img, straight_corners = create_straight_on_view(page_img, bg_color)
                save_sample(args.output, sample_id, straight_img, straight_corners,
                           article_path, is_straight=True, dark_mode=dark_mode)
                sample_id += 1

                # Create homographic transforms
                for t in range(args.transforms):
                    # Each transform gets a potentially different background
                    transform_bg = get_random_background_color()
                    warped_img, warped_corners = apply_random_homography(page_img, transform_bg)
                    save_sample(args.output, sample_id, warped_img, warped_corners,
                               article_path, is_straight=False, dark_mode=dark_mode)
                    sample_id += 1

            except Exception as e:
                print(f"  Error processing page: {e}", flush=True)
                import traceback
                traceback.print_exc()
                continue

        browser.close()

    print(f"\nGenerated {sample_id - args.start_id} samples in {args.output}", flush=True)


if __name__ == "__main__":
    main()
