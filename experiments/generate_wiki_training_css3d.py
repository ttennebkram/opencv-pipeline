#!/usr/bin/env python3
"""
Generate training data from Wikipedia ZIM archive using CSS 3D transforms.

This version re-renders the page for each perspective transform using CSS 3D,
giving crisp vector-quality text at any angle (no rasterization artifacts).

Requirements:
    pip install libzim playwright opencv-python numpy
    playwright install chromium

Usage:
    python3 generate_wiki_training_css3d.py --pages 10 --transforms 5 --output ./wiki_training
"""

import argparse
import base64
import json
import math
import os
import random
import re

import cv2
import numpy as np
from libzim.reader import Archive
from playwright.sync_api import sync_playwright

# Default paths
LOCAL_ZIM = "/Volumes/DevBlueB/kiwix/wikipedia_en_top_maxi_2025-09.zim"
AWS_ZIM = "/mnt/data/home/ubuntu/wikipedia_en_top_maxi_2025-09.zim"

# Page dimensions (8.5x11 aspect ratio)
PAGE_WIDTH = 850
PAGE_HEIGHT = 1100

# Output dimensions
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

        if any(ext in path_lower for ext in image_exts):
            continue
        if any(ext in path_lower for ext in skip_exts):
            continue
        if path.startswith('-/') or path.startswith('_/'):
            continue
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
        print(f"Error reading {path}: {e}")
        return None, None


def get_resource_as_data_uri(zim, path):
    """Get a resource from ZIM as a data URI for embedding."""
    try:
        entry = zim.get_entry_by_path(path)
        content = bytes(entry.get_item().content)

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
    def replace_src(match):
        src = match.group(1)
        clean_path = src.lstrip('./')

        data_uri = get_resource_as_data_uri(zim, clean_path)
        if data_uri:
            return f'src="{data_uri}"'

        for prefix in ['', 'I/', 'images/']:
            data_uri = get_resource_as_data_uri(zim, prefix + clean_path)
            if data_uri:
                return f'src="{data_uri}"'

        return match.group(0)

    html = re.sub(r'src="([^"]+)"', replace_src, html)
    html = re.sub(r"src='([^']+)'", replace_src, html)
    return html


def generate_random_transform():
    """
    Generate random 3D transform parameters.

    Returns:
        dict with: rotateX, rotateY, rotateZ, scale, translateX, translateY
    """
    return {
        # Rotation around X axis (tilt forward/back) - up to ±40 degrees
        'rotateX': random.uniform(-40, 40),
        # Rotation around Y axis (tilt left/right) - up to ±40 degrees
        'rotateY': random.uniform(-40, 40),
        # Rotation around Z axis (in-plane rotation) - full 360
        'rotateZ': random.uniform(0, 360),
        # Scale: 30-80% of output size
        'scale': random.uniform(0.3, 0.8),
        # Position: allow some off-center placement
        'translateX': random.uniform(-15, 15),  # percent of output width
        'translateY': random.uniform(-15, 15),  # percent of output height
    }


def compute_corners_from_transform(transform, page_w, page_h, output_w, output_h):
    """
    Compute the 4 corner positions after CSS 3D transform.

    CSS 3D transforms apply in this order (when using transform property):
    translate -> rotate -> scale (applied right to left in the string)

    We need to simulate this to find where corners end up.
    """
    # Extract parameters
    rx = math.radians(transform['rotateX'])
    ry = math.radians(transform['rotateY'])
    rz = math.radians(transform['rotateZ'])
    scale = transform['scale']
    tx_pct = transform['translateX']
    ty_pct = transform['translateY']

    # Page corners relative to center (before transform)
    half_w = page_w / 2
    half_h = page_h / 2
    corners = np.array([
        [-half_w, -half_h, 0, 1],  # top-left
        [half_w, -half_h, 0, 1],   # top-right
        [half_w, half_h, 0, 1],    # bottom-right
        [-half_w, half_h, 0, 1],   # bottom-left
    ], dtype=np.float64)

    # Build rotation matrices
    # Rotate around X
    Rx = np.array([
        [1, 0, 0, 0],
        [0, math.cos(rx), -math.sin(rx), 0],
        [0, math.sin(rx), math.cos(rx), 0],
        [0, 0, 0, 1]
    ])

    # Rotate around Y
    Ry = np.array([
        [math.cos(ry), 0, math.sin(ry), 0],
        [0, 1, 0, 0],
        [-math.sin(ry), 0, math.cos(ry), 0],
        [0, 0, 0, 1]
    ])

    # Rotate around Z
    Rz = np.array([
        [math.cos(rz), -math.sin(rz), 0, 0],
        [math.sin(rz), math.cos(rz), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Scale matrix
    S = np.array([
        [scale, 0, 0, 0],
        [0, scale, 0, 0],
        [0, 0, scale, 0],
        [0, 0, 0, 1]
    ])

    # Combined transform: Scale -> RotateZ -> RotateY -> RotateX
    # (CSS applies right-to-left, so we multiply in this order)
    M = Rx @ Ry @ Rz @ S

    # Apply transform to corners
    transformed = (M @ corners.T).T

    # Perspective projection (simple orthographic for now - CSS perspective is complex)
    # For CSS 3D, the perspective is set on the container
    # We'll use a perspective distance and project
    perspective = 1000  # pixels - matches CSS perspective value

    projected = []
    for corner in transformed:
        x, y, z, w = corner
        # Perspective divide
        if perspective + z != 0:
            factor = perspective / (perspective + z)
        else:
            factor = 1
        px = x * factor
        py = y * factor
        projected.append([px, py])

    projected = np.array(projected)

    # Translate to output center + offset
    # CSS translate(%) uses element's own dimensions, not output dimensions
    center_x = output_w / 2 + (tx_pct / 100) * page_w
    center_y = output_h / 2 + (ty_pct / 100) * page_h

    projected[:, 0] += center_x
    projected[:, 1] += center_y

    return projected.tolist()


def render_with_css3d_transform(page, html_content, zim, transform,
                                 scroll_pages=0, dark_mode=False, bg_color_rgb=(128, 128, 128)):
    """
    Render HTML with CSS 3D transform applied.

    Returns:
        numpy array (BGR image)
    """
    # Set viewport to output size
    page.set_viewport_size({"width": OUTPUT_WIDTH, "height": OUTPUT_HEIGHT})

    # Embed images
    html_with_images = embed_images_in_html(html_content, zim)

    # Colors
    if dark_mode:
        page_bg = "#1a1a1a"
        text_color = "#e0e0e0"
        link_color = "#6db3f2"
    else:
        page_bg = "white"
        text_color = "black"
        link_color = "#0645ad"

    # Background color for canvas
    canvas_bg = f"rgb({bg_color_rgb[0]}, {bg_color_rgb[1]}, {bg_color_rgb[2]})"

    # Build CSS transform string
    tx = transform['translateX']
    ty = transform['translateY']
    rx = transform['rotateX']
    ry = transform['rotateY']
    rz = transform['rotateZ']
    scale = transform['scale']

    # CSS transforms are applied right-to-left
    css_transform = f"rotateX({rx}deg) rotateY({ry}deg) rotateZ({rz}deg) scale({scale})"

    # Calculate scroll offset
    scroll_offset = int(scroll_pages * PAGE_HEIGHT)

    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            html, body {{
                width: {OUTPUT_WIDTH}px;
                height: {OUTPUT_HEIGHT}px;
                overflow: hidden;
                background: {canvas_bg};
            }}
            .perspective-container {{
                width: 100%;
                height: 100%;
                display: flex;
                justify-content: center;
                align-items: center;
                perspective: 1000px;
                perspective-origin: 50% 50%;
            }}
            .page {{
                width: {PAGE_WIDTH}px;
                height: {PAGE_HEIGHT}px;
                background: {page_bg};
                overflow: hidden;
                transform-style: preserve-3d;
                transform: translate({tx}%, {ty}%) {css_transform};
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }}
            .content {{
                width: 100%;
                padding: 40px;
                font-family: Georgia, 'Times New Roman', serif;
                font-size: 14px;
                line-height: 1.6;
                color: {text_color};
                transform: translateY(-{scroll_offset}px);
            }}
            .content a {{
                color: {link_color};
            }}
            .content img {{
                max-width: 100%;
                height: auto;
            }}
            .content table {{
                border-collapse: collapse;
                margin: 10px 0;
            }}
            .content th, .content td {{
                border: 1px solid {text_color};
                padding: 5px;
            }}
            .mw-jump-link, .navbox, .catlinks, .mw-editsection,
            .sistersitebox, .side-box, .metadata, .noprint {{
                display: none !important;
            }}
        </style>
    </head>
    <body>
        <div class="perspective-container">
            <div class="page">
                <div class="corner-marker" id="c0" style="position:absolute;left:0;top:0;width:1px;height:1px;"></div>
                <div class="corner-marker" id="c1" style="position:absolute;right:0;top:0;width:1px;height:1px;"></div>
                <div class="corner-marker" id="c2" style="position:absolute;right:0;bottom:0;width:1px;height:1px;"></div>
                <div class="corner-marker" id="c3" style="position:absolute;left:0;bottom:0;width:1px;height:1px;"></div>
                <div class="content">
                    {html_with_images}
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    page.set_content(styled_html, wait_until='load')
    page.wait_for_timeout(500)

    screenshot_bytes = page.screenshot(type='png')
    nparr = np.frombuffer(screenshot_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Get actual corner positions from browser using marker elements
    corners_from_browser = page.evaluate('''() => {
        // Get the actual screen positions of corner markers
        // These are transformed along with the page element
        const corners = [];
        for (let i = 0; i < 4; i++) {
            const marker = document.getElementById('c' + i);
            const rect = marker.getBoundingClientRect();
            corners.push([rect.left, rect.top]);
        }
        return corners;
    }''')

    return img, corners_from_browser


def get_random_background_color():
    """Get a random neutral background color as RGB tuple."""
    return random.choice(BACKGROUND_COLORS)


def normalize_corners(corners, width=OUTPUT_WIDTH, height=OUTPUT_HEIGHT):
    """Normalize corners to [-1, 1] range."""
    normalized = []
    for x, y in corners:
        nx = (x / width) * 2 - 1
        ny = (y / height) * 2 - 1
        normalized.append([nx, ny])
    return normalized


def save_sample(output_dir, sample_id, image, corners, article_path,
                is_straight=False, dark_mode=False, transform=None,
                scroll_pages=0, bg_color=None):
    """Save image and metadata with full transform parameters for reproducibility."""
    if is_straight:
        img_filename = f"sample_{sample_id:06d}_straight.jpg"
        json_filename = f"sample_{sample_id:06d}_straight.json"
    else:
        img_filename = f"sample_{sample_id:06d}_rotated.jpg"
        json_filename = f"sample_{sample_id:06d}_rotated.json"

    img_path = os.path.join(output_dir, "images", img_filename)
    cv2.imwrite(img_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])

    norm_corners = normalize_corners(corners)
    corners_flat = [coord for point in norm_corners for coord in point]

    metadata = {
        "id": sample_id,
        "image": img_filename,
        "article": article_path,
        "is_straight": is_straight,
        "dark_mode": dark_mode,
        "corners": corners_flat,
        "corners_pixels": [coord for point in corners for coord in point],
        # Full transform parameters for reproducibility
        "transform": transform if transform else {},
        "scroll_pages": scroll_pages,
        "bg_color_rgb": list(bg_color) if bg_color else None,
        # Generation parameters
        "page_width": PAGE_WIDTH,
        "page_height": PAGE_HEIGHT,
        "output_width": OUTPUT_WIDTH,
        "output_height": OUTPUT_HEIGHT,
        "generator_version": "2.0_browser_corners"
    }

    json_path = os.path.join(output_dir, "labels", json_filename)
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return img_path


def main():
    parser = argparse.ArgumentParser(description='Generate Wikipedia training data with CSS 3D')
    parser.add_argument('--zim', type=str, help='Path to ZIM file')
    parser.add_argument('--pages', type=int, default=10, help='Number of pages to render')
    parser.add_argument('--transforms', type=int, default=5, help='Transforms per page')
    parser.add_argument('--output', type=str, default='./wiki_training_css3d', help='Output directory')
    parser.add_argument('--start_id', type=int, default=0, help='Starting sample ID')
    args = parser.parse_args()

    zim_path = args.zim if args.zim else get_zim_path()
    print(f"Using ZIM file: {zim_path}", flush=True)

    os.makedirs(os.path.join(args.output, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "labels"), exist_ok=True)

    print("Opening ZIM archive...", flush=True)
    zim = Archive(zim_path)
    print(f"ZIM has {zim.entry_count} entries", flush=True)

    print("Indexing articles...", flush=True)
    article_paths = get_article_paths(zim)
    print(f"Found {len(article_paths)} articles", flush=True)

    sample_id = args.start_id

    with sync_playwright() as p:
        print("Launching browser...", flush=True)
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(device_scale_factor=1)  # Force 1:1 pixels
        page = context.new_page()

        for page_num in range(args.pages):
            article_path, html = get_random_article_html(zim, article_paths)
            if html is None:
                continue

            scroll_pages = random.uniform(0, 3)
            dark_mode = random.random() < 0.5

            mode_str = "dark" if dark_mode else "light"
            print(f"Page {page_num + 1}/{args.pages}: {article_path} ({mode_str}, scroll: {scroll_pages:.1f})", flush=True)

            try:
                # Straight-on view (no rotation, centered)
                straight_transform = {
                    'rotateX': 0, 'rotateY': 0, 'rotateZ': 0,
                    'scale': 0.85, 'translateX': 0, 'translateY': 0
                }
                bg_color = get_random_background_color()

                straight_img, straight_corners = render_with_css3d_transform(
                    page, html, zim, straight_transform,
                    scroll_pages, dark_mode, bg_color
                )
                save_sample(args.output, sample_id, straight_img, straight_corners,
                           article_path, is_straight=True, dark_mode=dark_mode,
                           transform=straight_transform, scroll_pages=scroll_pages, bg_color=bg_color)
                sample_id += 1

                # Random perspective transforms
                for t in range(args.transforms):
                    transform = generate_random_transform()
                    bg_color = get_random_background_color()

                    warped_img, corners = render_with_css3d_transform(
                        page, html, zim, transform,
                        scroll_pages, dark_mode, bg_color
                    )
                    save_sample(args.output, sample_id, warped_img, corners,
                               article_path, is_straight=False, dark_mode=dark_mode,
                               transform=transform, scroll_pages=scroll_pages, bg_color=bg_color)
                    sample_id += 1

                    print(f"  Transform {t+1}/{args.transforms}: rX={transform['rotateX']:.1f} rY={transform['rotateY']:.1f} rZ={transform['rotateZ']:.1f}", flush=True)

            except Exception as e:
                print(f"  Error: {e}", flush=True)
                import traceback
                traceback.print_exc()
                continue

        browser.close()

    print(f"\nGenerated {sample_id - args.start_id} samples in {args.output}", flush=True)


if __name__ == "__main__":
    main()
