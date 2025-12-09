# Homography Training Data Generator

This directory contains tools for generating training data for neural networks that estimate perspective distortion (homography) from images.

See the main [CLAUDE.md](../CLAUDE.md) for project overview and build instructions.

## Quick Start

```bash
# Continuous mode (default) - runs until Ctrl+C
./r2 --output training_data

# Generate specific amount (100 pages × 30 samples = 3000 frames)
./r2 --pages 100 --samples_per_page 30 --output training_data

# Preview mode (fewer samples, visible window)
./r2 --pages 5 --samples_per_page 10 --output test_run

# Estimate mode - show projections without generating
./r2 --estimate --pages 1000
```

## What We're Generating

The goal is to train a CNN to detect the 4 corners of a paper/document in a camera view, even when:
- The paper is rotated at any angle
- The camera is viewing at an angle (perspective distortion)
- The paper is partially off-screen
- The lighting varies (shadows, dark mode)

### Output Files

```
output_dir/
├── human_content.avi      # For human review: content sorted by rotation angle
├── human_white.avi        # For human review: white page sorted by rotation angle
├── training_content.avi   # For training: content in random order
├── training_white.avi     # For training: white page in random order
├── training_labels.json   # JSON array of per-frame labels
├── images/                # Individual frame images (PNG)
└── labels/                # Individual frame labels (JSON)
```

### Video Pairs

Each frame exists in two forms:
1. **Content video**: Wikipedia page with shadows/effects (what the camera sees)
2. **White video**: Pure white quadrilateral showing exact paper geometry (ground truth visualization)

Human review videos are sorted by rotation angle for smooth playback. Training videos maintain random order to avoid temporal patterns.

### JSON Label Format

```json
{
  "frame": 0,
  "id": "page_0_sample_5",
  "h": [h00, h01, h02, h10, h11, h12, h20, h21, h22],
  "inv": [inverse homography values],
  "corners": [x0, y0, x1, y1, x2, y2, x3, y3],
  "dark_mode": true
}
```

- `h`: 3×3 homography matrix (row-major) mapping source → destination
- `inv`: Inverse homography (destination → source)
- `corners`: Paper corner positions in pixels (TL, TR, BR, BL)
- `dark_mode`: Whether dark mode CSS was applied to the page

### Training Target

**⚠️ IMPORTANT: RESIZE IMAGES FOR TRAINING! ⚠️**

The generator outputs 1920×1080 images by default for high-quality source data. **You MUST resize these significantly before training:**

| Target Size | Use Case |
|-------------|----------|
| 224×224 | Typical CNN input (ResNet, VGG, etc.) |
| 320×240 | Faster training, good balance |
| 160×120 | Very fast training/prototyping |

Training on 1080p directly will be extremely slow, waste memory, and provide no benefit!

**Train on normalized corners, not homography values.**

The homography matrix values can range from -200M to +200M depending on the transform. Corners are bounded to image dimensions and can be normalized to [-1, +1]:

```python
# Normalize corners to [-1, +1] range
corners_normalized = [
    (x - width/2) / (width/2),
    (y - height/2) / (height/2),
    ...
]
```

Network output: 8 values (4 corners × 2 coordinates)

## Data Generation Parameters

### Perspective Transforms

| Parameter | Range | Distribution |
|-----------|-------|--------------|
| Paper rotation | 0° - 360° | Uniform |
| Camera tilt X | ±30% | Uniform |
| Camera tilt Y | ±30% | Uniform |
| Paper scale | 5% - 80% | Squared (biased toward small) |
| Paper position | -20% to 120% | Uniform |

The squared distribution for scale means more samples with smaller/distant papers.

### Visual Variations

- **Dark mode**: 50% of pages (CSS color inversion)
- **Scroll position**: 0-3 pages down (random)
- **Shadows**: 50% of samples, multiple types:
  - Gradient shadows (directional)
  - Edge vignettes
  - Cast shadows with perspective
  - Corner occlusion
  - 30% "hard black" shadows (pixels → 0)

## Command Line Options

```
./r2 [options]

Options:
  --samples_per_page N Distortions per page (default: 30)
  --pages N            Number of Wikipedia pages (-1 = continuous, default: -1)
  --output DIR         Output directory (default: training_data/homography)
  --fps N              Video frame rate (default: 30)
  --width W            Output image width (default: 1920)
  --height H           Output image height (default: 1080)
  --estimate           Show projections based on previous runs, don't generate
  --help               Show help message
```

## Stats Caching

The generator tracks statistics from each run in `~/.homography_generator_stats.json`:

- **Cumulative stats**: Total frames, runtime, average throughput across all runs
- **Last run details**: Frames, runtime, fps, pages, dark/light mode split

These stats are used to provide accurate projections for future runs. Use `--estimate` mode to see projections without generating data.

## Implementation Notes

- Uses JavaFX WebView to render Wikipedia pages
- OpenCV for perspective transforms and video writing
- Heavy processing runs on background thread (UI stays responsive)
- Thread-safe frame buffering with synchronized access
- Graceful shutdown on Ctrl+C (flushes remaining frames)

## Next Steps

1. Generate large dataset (1000+ pages × 30 samples = 30,000+ frames)
2. Train CNN with PyTorch (see `train_homography.py`)
3. Export model weights for Java inference
4. Create `HomographyEstimatorProcessor` for the pipeline editor
