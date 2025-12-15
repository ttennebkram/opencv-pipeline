#!/usr/bin/env python3
"""Quick visualization of eighth model corner predictions."""
import argparse
import json
import random
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn

IMG_W, IMG_H = 1920, 1080

class CornerCNN_Eighth(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Linear(256, 8)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)


def denorm(corners_norm):
    """Convert normalized [-1,1] corners to pixels."""
    c = np.array(corners_norm).reshape(4, 2)
    c[:, 0] = c[:, 0] * (IMG_W / 2) + (IMG_W / 2)
    c[:, 1] = c[:, 1] * (IMG_H / 2) + (IMG_H / 2)
    return c


def draw_quad(img, corners, color, label, thickness=3):
    pts = corners.astype(np.int32)
    cv2.polylines(img, [pts], True, color, thickness)
    for i, pt in enumerate(pts):
        cv2.circle(img, tuple(pt), 10, color, -1)
        cv2.putText(img, str(i), (pt[0]+12, pt[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(img, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="eighth_v2.pth")
    parser.add_argument("--data", default="/Volumes/SamsungBlue/ml-training/wiki_training_v2")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--count", type=int, default=1, help="Number of samples to visualize")
    args = parser.parse_args()

    # Load model
    model_path = Path(__file__).parent / args.model
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    model = CornerCNN_Eighth()
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    epoch = ckpt.get('epoch', '?')
    val_err = ckpt.get('val_pixel_error', None)
    print(f"Loaded model from epoch {epoch}, val_px_err={val_err:.1f}px" if val_err else f"Loaded model from epoch {epoch}")

    # Find samples
    data_path = Path(args.data)
    images_dir = data_path / "images"
    labels_dir = data_path / "labels"

    samples = sorted(images_dir.glob("*.jpg"))
    print(f"Found {len(samples)} images")

    for i in range(args.count):
        if args.sample is not None:
            idx = (args.sample + i) % len(samples)
        else:
            idx = random.randint(0, len(samples) - 1)

        img_path = samples[idx]
        label_path = labels_dir / (img_path.stem + ".json")

        # Load image
        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (240, 135), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)

        # Load ground truth
        with open(label_path) as f:
            label = json.load(f)
        gt_corners = denorm(label['corners'])

        # Inference
        with torch.no_grad():
            pred_norm = model(tensor).numpy()[0]
        pred_corners = denorm(pred_norm)

        # Compute error
        errors = np.sqrt(((gt_corners - pred_corners) ** 2).sum(axis=1))
        avg_err = errors.mean()

        # Draw
        vis = img.copy()
        vis = draw_quad(vis, gt_corners, (0, 255, 0), "GT (green)")
        vis = draw_quad(vis, pred_corners, (0, 0, 255), "")

        # Add error text
        cv2.putText(vis, f"Pred (red) - Avg err: {avg_err:.1f}px", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(vis, f"Sample: {img_path.stem}", (10, IMG_H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Save and open
        out_path = f"/tmp/viz_eighth_{i}.png"
        # Resize for display (half size)
        vis_small = cv2.resize(vis, (IMG_W // 2, IMG_H // 2))
        cv2.imwrite(out_path, vis_small)
        print(f"Sample {idx}: avg error = {avg_err:.1f}px -> {out_path}")

        import subprocess
        subprocess.run(["open", out_path], check=False)


if __name__ == "__main__":
    main()
