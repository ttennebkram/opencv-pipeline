# OpenCV Pipeline Editor

A graphical node-based pipeline editor for OpenCV image processing operations.

## IMPORTANT: Git Policy

**DO NOT create git commits or push without explicit user approval.**

- You may SUGGEST a commit when work is complete
- Wait for the user to test and confirm before committing
- Never commit code in a potentially broken state
- The user prefers to verify changes work before committing

**Commit message format:**
- Use only ONE attribution line at the end: `Co-Authored-By: Claude <noreply@anthropic.com>`
- Do NOT include the emoji/link line `🤖 Generated with [Claude Code](...)`

## IMPORTANT: Verification

**Do not claim a fix works until it has been tested.**

- After making changes, compile with `mvn compile` to catch errors
- Do not say "this should fix it" or "this is now fixed" - say "I've made changes, please test"
- If you're unsure whether something works, say so
- The user will verify functionality before considering something complete

## IMPORTANT: Python Coding Standards

**All print statements must use `flush=True`**

Python buffers stdout by default, which causes log output to appear delayed when monitoring remote training jobs or background processes. Always use:

```python
print("message", flush=True)
print(f"Epoch {epoch}: {error:.1f}px", flush=True)
```

This ensures immediate output visibility for monitoring long-running ML training.

## Build & Run

```bash
# Compile
mvn compile

# Run (normal)
mvn exec:exec

# Run with auto-start (loads last file and starts pipeline)
mvn exec:exec@start

# Run with custom arguments
mvn exec:exec@run -Dpipeline.args="file.json --start"

# Package uber-jar
mvn clean package
java -jar target/opencv-pipeline.jar

# Build all platform JARs for release
./build-all-platforms.sh
# Creates: target/opencv-pipeline-{mac-aarch64,mac,linux,linux-aarch64,win}.jar
```

## Tech Stack

- **Java 17** with **JavaFX 23** for UI
- **OpenCV 4.9.0** via openpnp wrapper (includes native libs for all platforms)
- **Gson** for JSON serialization
- **Maven** build system

## Project Structure

```
src/main/java/com/ttennebkram/pipeline/
├── PipelineEditorApp.java       # Main application, toolbar, canvas
├── PipelineEditorLauncher.java  # Entry point (for JavaFX module workaround)
├── fx/
│   ├── FXNode.java              # Visual node representation
│   ├── FXConnection.java        # Connection between nodes
│   ├── FXPipelineExecutor.java  # Runs the pipeline, manages Mat objects
│   ├── FXPipelineSerializer.java # Save/load JSON, thumbnails
│   ├── FXNodeRegistry.java      # Discovers and lists available node types
│   ├── FXNodeFactory.java       # Creates FXNode instances
│   ├── FXPropertiesDialog.java  # Node property editor dialogs
│   ├── FXContainerEditorWindow.java # Editor for Container sub-pipelines
│   ├── FXHelpBrowser.java       # Help documentation browser
│   ├── NodeRenderer.java        # Renders nodes on canvas
│   └── processors/              # All image processing nodes
│       ├── FXProcessor.java         # Base interface
│       ├── FXProcessorBase.java     # Common implementation
│       ├── FXProcessorInfo.java     # Annotation for auto-discovery
│       ├── FXProcessorRegistry.java # Maps nodeType -> processor class
│       ├── FXDualInputProcessor.java # Base for dual-input nodes
│       ├── FXMultiOutputProcessor.java # Base for multi-output nodes
│       └── *Processor.java          # Individual processor implementations
├── processing/                  # Legacy processor interfaces
├── effects/                     # Standalone effects (not pipeline nodes)
│   └── refraction/
│       └── RefractionCutGlass.java  # Cut-glass refraction effect (WIP, needs BaseEffect)
└── util/
    └── MatTracker.java          # Tracks OpenCV Mat allocations for leak detection
```

**Note on Effects vs Processors:**
- **Processors** (`fx/processors/`): Pipeline nodes with UI, serialization, auto-discovery via `@FXProcessorInfo`
- **Effects** (`effects/`): Standalone OpenCV effects, not integrated into pipeline yet. Port of Python effects.

## Adding a New Processor

1. Create a class in `fx/processors/` extending `FXProcessorBase` (or `FXDualInputProcessor` for two inputs, `FXMultiOutputProcessor` for multiple outputs)

2. Add the `@FXProcessorInfo` annotation:
```java
@FXProcessorInfo(
    nodeType = "MyFilter",           // Internal name (no spaces)
    displayName = "My Filter",        // Shown in UI
    buttonName = "MyFilt",            // Optional: shorter toolbar name
    category = "Filter",              // Category in toolbar
    description = "Does something\nSecond line"
)
public class MyFilterProcessor extends FXProcessorBase {
```

3. Implement required methods:
   - `process(Mat input)` - the actual image processing
   - `buildPropertiesDialog(FXPropertiesDialog dialog)` - UI for parameters
   - `serializeProperties(JsonObject json)` / `deserializeProperties(JsonObject json)`
   - `syncFromFXNode(FXNode node)` / `syncToFXNode(FXNode node)`

4. The processor is auto-discovered via annotation scanning (no registration needed)

## Categories

Toolbar categories appear in this order (defined in `FXNodeRegistry.CATEGORY_ORDER`):
1. Sources
2. Basic
3. Blur
4. Content
5. Edges
6. Filter
7. Morphology
8. Transform
9. Detection
10. Dual Input
11. Utility
12. Visualization
13. Container I/O

## Key Patterns

### Node Status Text
Processors can display extra runtime information (like computed values) via `FXNode.statusText`:
```java
// In process() method, update statusText for display on the node
if (fxNode != null) {
    fxNode.statusText = String.format("Value: %.0f", computedValue);
}
```
- statusText is displayed after "Work:" in the node's stats line
- Cleared when pipeline starts (in `clearPipelineStats()`)
- Persisted to JSON and restored on load

### Mat Memory Management
OpenCV Mats must be manually released. Use `MatTracker` for debugging leaks:
```java
MatTracker.register(mat, "description");
MatTracker.release(mat);
MatTracker.dumpStats();  // Print allocation info
```

### Properties Dialog Helpers
`FXPropertiesDialog` provides common controls:
- `addSlider(label, min, max, value)`
- `addOddKernelSlider(label, value, max)` - for odd-only kernel sizes
- `addCheckBox(label, checked)`
- `addComboBox(label, options, selected)`
- `addColorPicker(label, color)`

### Serialization Helpers in FXProcessorBase
```java
getJsonInt(json, "key", defaultVal)
getJsonDouble(json, "key", defaultVal)
getJsonString(json, "key", defaultVal)
getInt(properties, "key", defaultVal)  // from FXNode.properties map
```

## File Format

Pipelines are saved as JSON with `.json` extension. Thumbnails are cached in `~/.opencv-pipeline/thumbnails/`.

## Known Issues

- Watch for file handle leaks - always close InputStreams, use try-with-resources
- FFT nodes have complex multi-channel output modes (grayscale vs 4-channel)
- Container nodes have their own sub-pipeline serialized inline

## ML Experiments (experiments branch)

The `experiments` branch contains experimental ML/deep learning code for training CNNs.

### Running ML Demos

```bash
git checkout experiments

# Auto-detect best backend (Python+GPU if available, falls back to Java)
mvn exec:exec@ml -Dml.class=MLFacade

# Force pure Java backend (no Python needed)
mvn exec:exec@ml -Dml.class=MLFacade '-Dml.args=--djl'

# Run individual components
mvn exec:exec@ml -Dml.class=PythonMLBridge    # Python bridge demo
mvn exec:exec@ml -Dml.class=DJLTrainer        # Pure Java trainer
mvn exec:exec@ml -Dml.class=JavaCNNInference  # Train + Java inference
```

### ML Architecture

- **MLFacade**: Auto-selects best backend (Python+GPU > Python+CPU > Java/DJL)
- **PythonMLBridge**: Step-by-step Python/PyTorch calls with MPS/CUDA GPU support
- **DJLTrainer**: Pure Java fallback using DJL when Python unavailable
- **JavaCNNInference**: Pure Java inference (~0.3ms/image) - no Python needed at runtime

### Performance

| Backend | Training (3 epochs) | Inference |
|---------|---------------------|-----------|
| Python + MPS (Mac) | ~10s | - |
| Python + CUDA (NVIDIA) | ~10s | - |
| Java DJL (CPU) | ~22s | - |
| Java Inference | - | 0.3ms/image |

Training exports portable weight files that work with JavaCNNInference on any platform.

## Homography Estimation Experiment (WIP)

**Goal**: Train a neural network to recognize perspective distortion in images and output the corner positions of a paper/document in the camera view.

### Quick Start

```bash
cd experiments

# Continuous mode (default) - runs until Ctrl+C
# Data stored on external SSD: /Volumes/SamsungBlue/ml-training/homography/training_data
./r2

# Run for limited time (recommended - disk fills in ~2 hours at full speed!)
./r2 --seconds 300   # 5 minutes

# Generate specific amount (by page count)
./r2 --samples_per_page 30 --pages 100

# Estimate mode - show projections without generating
./r2 --estimate --pages 1000

# Train the model
python3 train_homography.py --epochs 50
```

### Files

- `experiments/HomographyTrainingDataGenerator.java` - Renders Wikipedia pages, applies random perspective transforms, generates training videos and JSON labels
- `experiments/r2` - Shell script to compile and run the generator with JavaFX

### Training Data Generator Features

The generator simulates a camera viewing a paper document on a surface:

1. **Web Page Rendering**:
   - Renders random Wikipedia pages using JavaFX WebView
   - 50% dark mode (CSS injection) for variety
   - Random page scrolling (0-3 pages down) to avoid position bias
   - **Note**: Stage must be shown for WebView to render (headless mode moves window off-screen)

2. **Perspective Transforms**:
   - Paper rotation: full 360°
   - Camera tilt: ±30% perspective distortion (realistic viewing angles)
   - Paper scale: 5-80% of frame (biased toward smaller/distant papers via squared distribution)
   - Paper position: -20% to 120% of frame (can be partially off-screen)

3. **Shadow Effects** (50% of samples):
   - Gradient shadows (directional)
   - Edge vignettes
   - Cast shadows with perspective
   - Corner occlusion
   - 30% of shadows are "hard black" (pixels → 0)

4. **Video Output** (4 videos at 30 fps):
   - `human_content.avi` - Content with shadows, sorted by rotation angle (for review)
   - `human_white.avi` - White page geometry, sorted by rotation angle (for review)
   - `training_content.avi` - Content with shadows, random order (for training)
   - `training_white.avi` - White page geometry, random order (for training)

5. **Threading**: Heavy OpenCV processing runs on background thread, keeping UI responsive

6. **JSON Labels** (per frame):
```json
{
  "frame": 0,
  "id": "page_0_sample_5",
  "h": [9 homography values],
  "inv": [9 inverse values],
  "corners": [x0, y0, x1, y1, x2, y2, x3, y3],
  "dark_mode": true
}
```

7. **Stats Tracking**:
   - Dark/light mode page counts printed at end
   - Thread-safe frame buffering for concurrent access
   - Stats cached to `~/.homography_generator_stats.json` for projections
   - `--estimate` mode shows projections without generating data
   - Live stats monitoring in multi-worker mode (every 5 seconds)

8. **Output Resolution**:
   - Default: 1920×1080 (1080p) for high-quality source data
   - **IMPORTANT**: Resize significantly for training (224×224, 320×240, etc.)
   - Training on 1080p directly is wasteful and slow

9. **Resource Limits**:
   - `--seconds N` - Stop after N seconds
   - `--max_frames N` - Stop after N frames
   - **WARNING**: At full speed (~50 fps with 10 workers), generates ~200KB/frame
   - This fills ~36GB/hour - **disk will fill in ~2 hours** on a typical machine!
   - **KNOWN BUG**: `--max_frames` doesn't work in multi-worker mode - use Ctrl+C or `--seconds`

### Training Approach

**Network output:** 18 log-transformed values (homography matrix 9 + inverse matrix 9)

**Log transform** compresses extreme value ranges:
- Raw homography values can range from -200M to +200M
- Log transform: `sign(x) * log1p(|x|)` compresses to ~[-19, +19]
- Inverse (for Java inference): `sign(y) * (exp(|y|) - 1)`

### Corner Prediction Training (Current Approach)

**Output:** 8 normalized corner values (4 corners × 2 coordinates) in range [-1, +1]
- Much more stable than homography matrix values (which range -200M to +200M)
- Homography computed from corners at inference time using OpenCV

**Two architectures available:**

| Architecture | Parameters | Model Size | Script |
|--------------|------------|------------|--------|
| FC [512,256,64] | 4.87M | ~19 MB | `train_corners.py` |
| CNN (4 conv blocks) | **391K** | ~1.5 MB | `train_corners_cnn.py` |

**CRITICAL: Plain CNNs fail - use ResNet+BatchNorm**

Plain CNNs without skip connections suffer from vanishing gradients and fail to learn corner detection (stuck at ~400px error). The solution is ResNet-style architecture:

| Architecture | Parameters | Overfit Test | Result |
|--------------|------------|--------------|--------|
| Plain CNN + BN | 391K | 118px | Slow, unstable |
| Plain CNN (no BN) | 1.58M | 399px | **FAILS completely** |
| **ResNet + BN** | 1.2M | **4.9px** | **SUCCESS** |

**Working Architecture** (train_resnet.py):
```python
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=stride) if stride != 1 or in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)  # Skip connection!

class ResNetCorners(nn.Module):
    def __init__(self):
        # 240×135 input → 8 corner coordinates
        self.layer1 = ResBlock(1, 32, stride=2)    # → 120×67
        self.layer2 = ResBlock(32, 64, stride=2)   # → 60×33
        self.layer3 = ResBlock(64, 128, stride=2)  # → 30×16
        self.layer4 = ResBlock(128, 256, stride=2) # → 15×8
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Linear(256, 8)
```

**Key insights:**
- Skip connections let gradients flow directly (no vanishing)
- BatchNorm stabilizes training
- Either one alone is insufficient - need BOTH
- 1/8th resolution (240×135) works better than full res (less noise, faster)

**Always run tiny overfit test first** (50 samples, train to near-zero loss) before scaling up. If it can't overfit, architecture is broken.

### Progressive Overfit Playbook (CURRENT METHODOLOGY)

**This is the systematic approach for validating model capacity before full training.**

The idea: Before investing in full dataset training, prove the model can memorize progressively larger subsets. This catches architecture/optimization issues early with fast, cheap experiments.

#### The Algorithm

```
overfit_samples = 50
TARGET = 5.0  # pixels

while overfit_samples <= dataset_size:
    1. LOAD overfit_samples from dataset
    2. TRAIN until pixel_error < TARGET
       - Warm start from previous checkpoint (or fresh weights for first)
       - NO train/val split (pure memorization test)
       - NO augmentation
       - LR=0.0005 with ReduceLROnPlateau(patience=50, factor=0.5)
       - Eval on SAME data as training

    3. DECISION GATE:
       IF pixel_error < TARGET within 2000 epochs:
           ✅ PASS - model can memorize at this scale
           overfit_samples *= sqrt(2)  # ~1.41× increase
       ELSE:
           ⚠️ ALERT - investigate before proceeding

# Once overfit_samples >= dataset_size:
SWITCH to GENERAL TRAINING:
    - 90/10 train/val split
    - No augmentation initially
    - Target: val_error < 4px
```

**Why √2 scaling?** Originally used 10× (50→500→5K), but 5K stalled. Switching to √2 (~1.41×) provides finer granularity to identify exactly where capacity limits appear.

#### Progressive Overfit Results (2025-12-15)

| Samples | Best Error | Epochs | Status |
|---------|------------|--------|--------|
| 50 | 4.5px | 149 | ✅ PASS |
| 500 | 4.6px | 430 | ✅ PASS |
| 707 | 5.0px | 302 | ✅ PASS |
| 1000 | 4.3px | 303 | ✅ PASS |
| 1414 | 4.1px | 308 | ✅ PASS |
| 2000 | 4.1px | 335 | ✅ PASS |
| 2828 | 4.7px | 641 | ✅ PASS |
| 4000 | 4.7px | 303 | ✅ PASS |
| 5656 | ? | ? | 🔄 RUNNING |
| 8000 | - | - | Pending |
| ... | - | - | Continue √2 scaling |

**Checkpoints:** `/Volumes/SamsungBlue/ml-training/progressive_checkpoints/`

#### Diagnostic Playbook

Each "chapter" follows this structure:

1. **Hypothesis**: What might be limiting progress
2. **Cheap test(s)**: Low-cost experiments to isolate the issue
3. **Decision gate**: Based on evidence, choose ONE action:
   - **Action A - Fix pipeline**: When tests show preprocessing jitter, quantization artifacts, coordinate-space issues
   - **Action B - Improve supervision**: When tests show label instability, multiple correct solutions, loss↓ but error flat
   - **Action C - Increase capacity**: ONLY when A and B are ruled out AND plateau remains

**Meta-rule**: Do not proceed until the chosen action is executed AND tests re-run.

#### Key Findings

- **ResNet+BatchNorm required**: Plain CNN stuck at ~400px; ResNet reaches <5px
- **No train/val split for overfit**: Split causes apparent "overfitting" even on memorization test
- **LR scheduler essential**: Model finds solutions but bounces; scheduler converges
- **Epoch time ~3.1× per 10×**: Chapters get longer but still manageable

#### Current Training Commands

```bash
# SSH to AWS
ssh -i ~/.ssh/ml-training.pem ubuntu@98.91.248.103

# Monitor current overfit test
tail -20 /opt/dlami/nvme/overfit_4000.log

# Create next √2 step (example: 4000 → 5656)
sed 's/4000/5656/g; s/progressive_2828/progressive_4000/g' ~/overfit_4000.py > ~/overfit_5656.py

# Run next test
source /opt/pytorch/bin/activate && cd /opt/dlami/nvme && \
  nohup python3 ~/overfit_5656.py > overfit_5656.log 2>&1 &

# Backup checkpoints locally
scp -i ~/.ssh/ml-training.pem ubuntu@98.91.248.103:/opt/dlami/nvme/progressive_*.pth \
  /Volumes/SamsungBlue/ml-training/progressive_checkpoints/

# Full training (after all overfit stages pass)
python3 ~/train_resnet.py --data wiki_training_v3 --epochs 200 \
  --batch_size 256 --lr 0.0005 --max_samples 300000 \
  --output resnet_general.pth
```

### ML Development Checklist & Metrics (Legacy)

**Older checklist - superseded by Progressive Overfit Playbook above.**

#### Overfit Test Results

| Date | Location | Architecture | Samples | Best Error | Epochs | Status |
|------|----------|--------------|---------|------------|--------|--------|
| 2025-12-15 | Local MPS | ResNet+BN | 50 | 4.9px | 123 | ✅ PASS |
| 2025-12-15 | AWS CUDA | ResNet+BN | 50 | 4.5px | 149 | ✅ PASS |
| 2025-12-15 | AWS CUDA | ResNet+BN | 500 | 4.6px | 430 | ✅ PASS |

#### Augmentation Verification
Visualize augmented images with ground truth corners to verify transforms are correct.
- Horizontal flip: corners swap TL↔TR, BL↔BR, negate X
- Vertical flip: corners swap TL↔BL, TR↔BR, negate Y
- Test script: `experiments/viz_augmented.py` → `/tmp/augment_test_*.png`

| Date | Augmentation | Corners Correct | Status |
|------|--------------|-----------------|--------|
| 2025-12-15 | H-flip, V-flip, brightness, contrast | Yes | ✅ PASS |

#### Common Failure Modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Train stuck ~400px | No skip connections | Use ResNet architecture |
| Train drops, Val explodes | Overfitting | Add augmentation, dropout |
| Both stuck high | Bad LR or architecture | Reduce LR, check overfit test |
| Overfit test fails | Architecture broken | Fix architecture before proceeding |

**Legacy CNN Architecture** (train_corners_cnn.py) - DEPRECATED:
```
Input: 128×72×1 grayscale
Conv 3×3, 32 filters + BN + ReLU + MaxPool → 64×36×32
Conv 3×3, 64 filters + BN + ReLU + MaxPool → 32×18×64
Conv 3×3, 128 filters + BN + ReLU + MaxPool → 16×9×128
Conv 3×3, 256 filters + BN + ReLU + MaxPool → 8×4×256
GlobalAveragePool → 256
Dense → 8 output values
```

**Why CNN is better for this task:**
- **12× fewer parameters** - faster training, less overfitting
- **Spatial hierarchy** - early layers detect edges, later layers detect corners
- **Translation invariance** - learns corner features regardless of position
- **Parameter sharing** - same conv filters applied across entire image

**Training commands:**
```bash
cd experiments

# Train CNN (recommended)
python3 train_corners_cnn.py --epochs 50 --lr 0.002

# Train with data augmentation (horizontal flip, brightness, contrast)
python3 train_corners_cnn.py --epochs 50 --lr 0.002 --augment

# Train with cosine annealing LR scheduler
python3 train_corners_cnn.py --epochs 50 --lr 0.002 --scheduler cosine

# Combined: augmentation + cosine annealing
python3 train_corners_cnn.py --epochs 50 --lr 0.002 --augment --scheduler cosine

# Resume training from checkpoint
python3 train_corners_cnn.py --epochs 50 --lr 0.0005 --resume cnn_best.pth

# Train FC (for comparison)
python3 train_corners.py --epochs 50 --lr 0.0005

# Parallel LR sweep (10 experiments)
for lr in 0.0001 0.0003 0.0005 0.0007 0.001 0.002 0.003 0.005 0.007 0.01; do
  nice -n 10 python3 train_corners_cnn.py --epochs 5 --lr $lr \
    --output "cnn_lr${lr}.pth" 2>&1 &
done
```

**CNN Training Features:**
- `--augment`: On-the-fly data augmentation (horizontal flip, ±10% brightness, ±10% contrast)
- `--scheduler`: LR scheduler choice:
  - `plateau` (default): ReduceLROnPlateau - reactive, reduces LR when validation plateaus
  - `cosine`: CosineAnnealingLR - smooth decay from initial LR to near-zero
  - `cosine_warm`: CosineAnnealingWarmRestarts - periodic restarts (T_0=10, T_mult=2)
- `--resume`: Continue training from a saved checkpoint (preserves optimizer state)

### Training Data Requirements

**Rule of thumb:** 10× to 100× training examples per parameter for good generalization.

| Model | Parameters | Min Training Ticks | Max Training Ticks |
|-------|------------|-------------------|-------------------|
| CNN | 391K | **4M** (10×) | **40M** (100×) |
| FC | 4.87M | 49M (10×) | 487M (100×) |

**Epochs needed** (with current 67K samples, 54K training):

| Target | CNN Epochs | FC Epochs |
|--------|------------|-----------|
| Minimum (10×) | **~75** | ~910 |
| Maximum (100×) | **~745** | ~9,100 |

**Current status:** With 67K samples at 50 epochs, we're seeing each sample ~50 times.
- CNN: 50 × 54K = 2.7M ticks (68% of minimum 4M target)
- Need ~75 epochs to reach minimum, ~745 epochs for thorough training

**Implication:** Current CNN training runs (50 epochs) are likely undertrained. Consider:
1. Running longer (75-150 epochs minimum)
2. Generating more training data
3. Using data augmentation (`--augment`) to effectively multiply dataset size

### FC Training Results (Current Dataset: ~67K samples)

| Resolution | Parameters | Best Pixel Error | Notes |
|------------|------------|------------------|-------|
| 128×72 FC | 4.87M | ~379px | lr=0.0005 optimal |

**Observations:**
- Optimal LR for FC: 0.0003-0.002 range (all within 10px of each other)
- LR >= 0.005 causes instability
- LR = 0.01 diverges completely

### Implementation Plan

1. [x] Create training data generator (`experiments/HomographyTrainingDataGenerator.java`)
   - Uses JavaFX WebView to render Wikipedia pages
   - Random perspective transforms simulating paper on table
   - Shadow effects for realism
   - Dual video output (content + geometry)
2. [x] Create Python training script (`experiments/train_homography.py`)
   - Fully-connected network with hidden layers [512, 256, 64]
   - MSE loss on log-transformed homography + inverse (18 values)
   - SGD with momentum, ReduceLROnPlateau scheduler
   - Exports weights for Java inference
3. [x] Initial training run with ~11K samples
4. [ ] Generate larger training dataset (50K+ samples)
5. [ ] Retrain and compare results
6. [ ] Export trained model for Java inference
7. [ ] Create `HomographyEstimatorProcessor` - runs inference in pipeline

### Alternative: High-Resolution Document Scans

Instead of low-res camera captures, train on 300 DPI scans of 8.5×11" paper:
- **Input size**: 2550×3300 pixels (8.4 million grayscale)
- **Use case**: Document deskewing, perspective correction for scanned documents

**Why CNNs excel here** (vs fully-connected):
- **Parameter sharing**: Conv kernels reuse weights across spatial locations
- **Hierarchical pooling**: Progressively reduce 8.4M pixels → manageable feature vector
- **Translation invariance**: Detect features regardless of position

**Example CNN architecture** (2550×3300 → 8 corner values):
```
Layer                  Output Shape      Parameters
─────────────────────────────────────────────────────
Conv 3×3, 32 filters   2550×3300×32      320
MaxPool 2×2            1275×1650×32      -
Conv 3×3, 64 filters   1275×1650×64      18,496
MaxPool 2×2            637×825×64        -
Conv 3×3, 128 filters  637×825×128       73,856
MaxPool 2×2            318×412×128       -
Conv 3×3, 256 filters  318×412×256       295,168
MaxPool 2×2            159×206×256       -
GlobalAveragePool      256               -
Dense → 8              8                 2,056
─────────────────────────────────────────────────────
Total: ~390K parameters (~1.5 MB model)
```

**Comparison**:
| Approach | Input Size | Parameters | Model Size |
|----------|------------|------------|------------|
| Fully-connected (160×120) | 19,200 | 9.6M | ~39 MB |
| CNN (2550×3300) | 8.4M | 390K | ~1.5 MB |

CNNs use **25× fewer parameters** while handling **437× more pixels**!

### Related

- `calibrate` branch: CalibrateProcessor for projector alignment using detected corners
- Uses OpenCV `Calib3d.findHomography()` and `Imgproc.warpPerspective()`

## OCR Training Experiment (Planned)

**Goal**: Train a CNN to recognize characters in high-resolution document scans, outputting a grid of character predictions.

### Input/Output Specification

- **Input**: 2550 × 3300 grayscale image (300 DPI scan of 8.5" × 11" paper)
- **Output**: 80 × 66 grid of character predictions (80 chars/line × 66 lines)
- **Character set**: ~96 classes (printable ASCII 32-126 + blank)

### Why CNN? (Fully-Connected Won't Work)

Fully-connected layers require a weight for every input→output connection:

| Layer Size | Parameters | Memory |
|------------|------------|--------|
| 8.4M → 841K | 7 trillion | ~26 TB |
| 8.4M → 84K | 708 billion | ~2.6 TB |
| 8.4M → 8.4K | 71 billion | ~260 GB |

CNNs solve this with **local connectivity** and **weight sharing**:
- Each neuron connects only to a small local region (e.g., 3×3 pixels)
- Same weights reused across entire image
- Result: ~400K parameters instead of trillions

### CNN Architecture

**Convolution layers** learn to detect features (edges, curves, character parts).
**Pooling layers** downsample by taking the max of each NxN region:

```
MaxPool 2×2 example:
┌───┬───┬───┬───┐       ┌───┬───┐
│ 1 │ 3 │ 5 │ 2 │       │ 4 │ 6 │  (max of each 2×2 block)
├───┼───┼───┼───┤   →   ├───┼───┤
│ 4 │ 2 │ 6 │ 1 │       │ 8 │ 9 │
├───┼───┼───┼───┤       └───┴───┘
│ 7 │ 8 │ 3 │ 9 │
├───┼───┼───┼───┤
│ 0 │ 5 │ 7 │ 4 │
└───┴───┴───┴───┘
```

**Proposed architecture** (2550×3300 → 80×66×96):

```
Layer              Output Shape         Parameters
─────────────────────────────────────────────────────
Input              2550 × 3300 × 1      -
Conv 3×3, 32       2550 × 3300 × 32     320
MaxPool 2×2        1275 × 1650 × 32     -
Conv 3×3, 64       1275 × 1650 × 64     18,496
MaxPool 2×2        637 × 825 × 64       -
Conv 3×3, 128      637 × 825 × 128      73,856
MaxPool 2×2        318 × 412 × 128      -
Conv 3×3, 256      318 × 412 × 256      295,168
MaxPool 4×4        79 × 103 × 256       -
Conv 1×1, 96       79 × 103 × 96        24,672  (96 = character classes)
─────────────────────────────────────────────────────
Total: ~412K parameters (~1.6 MB)
```

Final output is ~80×103 grid with 96 channels (one per character class). Pooling can be tuned to hit exactly 80×66.

### Training Data Generation Plan

1. Render Wikipedia article text (content only, no sidebar/logo)
2. Use various fixed-width/monospace fonts (Courier, Consolas, Monaco, etc.)
3. Font size chosen so 80 characters fit across 8.5" at 300 DPI
4. Output: grayscale image + JSON with 66 lines of up to 80 characters each

### Plan B: Cell-Based Shared Network (Alternative to CNN)

Instead of sliding convolutions, use **fixed non-overlapping cells** with a shared fully-connected network:

**Cell dimensions:**
- Horizontal: 2550 ÷ 80 = 31.875 → ~32 pixels per cell
- Vertical: 3300 ÷ 66 = 50 pixels per cell
- Each cell: 32 × 50 = **1,600 pixels**

**Architecture per cell:**
```
1,600 input pixels
      ↓
Hidden layer 1 (256 neurons)  — 409,856 params
      ↓
Hidden layer 2 (64 neurons)   — 16,448 params
      ↓
Output: 96 classes            — 6,240 params
─────────────────────────────────────────
Total: ~432K parameters (shared across all 5,280 cells)
```

**Key insight:** Same network processes every cell. Training shows it 5,280 examples per image (each cell + its label). Network learns "given these 1,600 pixels, what character?" regardless of position.

**Comparison to CNN:**
- CNN: Sliding overlapping windows, learns hierarchical features
- Cell-based: Fixed grid, fully-connected within each cell
- Both use weight sharing, both ~400K parameters

### Implementation Steps

1. [ ] Create `OCRTrainingDataGenerator.java`
   - Fetch Wikipedia article text via API (plain text, no HTML)
   - Render in monospace font at 300 DPI
   - Cycle through available fixed-width fonts
   - Save image + character grid labels
2. [ ] Create `train_ocr.py` with CNN architecture (Plan A)
3. [ ] Generate training dataset
4. [ ] Train and evaluate CNN
5. [ ] (Plan B) Implement cell-based shared network for comparison
6. [ ] Export best model for Java inference

## Cloud GPU Training Options

For faster training than local Apple MPS, consider cloud GPU instances.

### Recommended: AWS G5 Instances

**Best value for small/medium CNNs** (like our 391K parameter corner estimator).

| Instance | GPU | VRAM | vCPUs | RAM | On-Demand | Spot |
|----------|-----|------|-------|-----|-----------|------|
| **g5.xlarge** | 1× A10G | 24 GB | 4 | 16 GB | ~$1.00/hr | ~$0.30-0.40/hr |
| g5.2xlarge | 1× A10G | 24 GB | 8 | 32 GB | ~$1.20/hr | ~$0.40-0.50/hr |
| g5.4xlarge | 1× A10G | 24 GB | 16 | 64 GB | ~$1.60/hr | ~$0.50-0.60/hr |

**NVIDIA A10G specs:**
- 24 GB GDDR6 VRAM
- 31.2 TFLOPS FP32
- 125 TFLOPS Tensor (FP16)
- 600 GB/s memory bandwidth

**Quick start:**
```bash
# 1. Launch g5.xlarge with "Deep Learning AMI (PyTorch)"
# 2. Upload training data
scp -r training_data/ ec2-user@<instance>:~/

# 3. Upload training script
scp train_corners_cnn.py ec2-user@<instance>:~/

# 4. SSH in and train
ssh ec2-user@<instance>
python3 train_corners_cnn.py --epochs 200 --lr 0.002 --augment

# 5. Download results
scp ec2-user@<instance>:~/cnn_*.pth .
scp ec2-user@<instance>:~/cnn_*.weights .
```

### Google Cloud Alternative

| Instance | GPU | VRAM | On-Demand |
|----------|-----|------|-----------|
| a2-highgpu-1g | 1× A100 | 40 GB | ~$3.67/hr |
| a2-highgpu-2g | 2× A100 | 80 GB | ~$7.35/hr |

**When to choose Google Cloud:**
- Need >24 GB VRAM (large models)
- Already using GCP infrastructure
- Training large language models or diffusion models

### Budget Alternatives

| Provider | GPU | Price | Notes |
|----------|-----|-------|-------|
| Lambda Labs | A10 | ~$0.60/hr | Simple, ML-focused |
| RunPod | A10G | ~$0.40/hr | Good for quick experiments |
| Vast.ai | Various | ~$0.20-0.50/hr | Community GPUs, variable quality |

### Cost Estimates (200-epoch CNN training)

| Platform | Time | Cost |
|----------|------|------|
| Local MPS (M1/M2) | ~21 hours | $0 |
| AWS g5.xlarge (on-demand) | ~2-4 hours | $2-4 |
| AWS g5.xlarge (spot) | ~2-4 hours | **$0.60-1.60** |
| Google Cloud A100 | ~1-2 hours | $4-7 |

**Recommendation:** AWS g5.xlarge Spot for best price/performance

## AWS Parallel CPU Training (No GPU)

For hyperparameter sweeps and multi-seed experiments, run multiple training processes in parallel on a single CPU instance. This is more cost-effective than GPU for small CNNs when running many experiments.

### Current AWS State (2025-12-15)

**ACTIVE INSTANCE - GPU Training:**
- **Instance:** `i-0563f8c98bdc5d818` (g5.xlarge with A10G GPU)
- **IP:** `98.91.248.103`
- **Task:** Training CNN on 118K wiki samples (100 epochs)
- **Data:** `/opt/dlami/nvme/wiki_training_v2/` (118,453 images)
- **Log:** `/opt/dlami/nvme/train_118k.log`
- **Output model:** `/opt/dlami/nvme/eighth_118k.pth`

**Monitor training:**
```bash
ssh -i ~/.ssh/ml-training.pem ubuntu@98.91.248.103 "tail -20 /opt/dlami/nvme/train_118k.log"
```

**Training parameters:**
- Resolution: 240×135 (1/8 scale)
- Batch size: 256
- Learning rate: 0.002
- Epochs: 100

**S3 bucket:** `s3://ml-training-wiki-homography/wiki_training_v2/`
- 118,453 images + labels (~22 GB)

**Local data:** `/Volumes/SamsungBlue/ml-training/wiki_training_v2/`

**Previous results:**
| Dataset | Best Pixel Error |
|---------|------------------|
| 5K samples | 197px |
| 54K samples | 122px |
| 116K samples (4 epochs) | 191px |

### Setup

**Prerequisites:**
- AWS CLI configured (`aws configure`)
- SSH key pair: `~/.ssh/ml-training.pem`
- Security group with SSH access (port 22)

**Base AMI:** `ami-0dd2d28e9f738aa69` - Ubuntu 22.04 with:
- Python 3.10 + venv at `~/ml-env`
- PyTorch 2.9.1, scipy 1.15.3
- 10K synthetic training images at `~/training/data/synthetic`
- Training scripts at `~/training/scripts`

### Recommended Instance

| Instance | vCPUs | RAM | Storage | On-Demand | Spot |
|----------|-------|-----|---------|-----------|------|
| **c6i.2xlarge** | 8 | 16 GB | 30 GB EBS | ~$0.34/hr | ~$0.10-0.15/hr |

**Why c6i.2xlarge:**
- 8 vCPUs allows 8-10 parallel training processes
- 16 GB RAM is sufficient for multiple PyTorch instances
- Intel Ice Lake CPUs have good single-thread performance
- Spot pricing is very cheap (~$0.10/hr)

### Quick Start

```bash
# 1. Launch instance from custom AMI
aws ec2 run-instances \
  --image-id ami-0dd2d28e9f738aa69 \
  --instance-type c6i.2xlarge \
  --key-name ml-training \
  --security-group-ids sg-08e1d25f98f79c51e \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":30}}]' \
  --query 'Instances[0].InstanceId' --output text

# 2. Wait for instance and get IP
aws ec2 describe-instances --instance-ids <instance-id> \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text

# 3. SSH in
ssh -i ~/.ssh/ml-training.pem ubuntu@<ip>

# 4. Activate environment
source ~/ml-env/bin/activate
cd ~/training/scripts
```

### Parallel Training (10 processes on single instance)

Each process trains on a different random seed or data slice:

```bash
# Run 10 parallel training processes with different seeds
for seed in {0..9}; do
  python3 train_corners_cnn.py \
    --epochs 10 \
    --lr 0.002 \
    --batch_size 64 \
    --augment \
    --scheduler cosine_warm \
    --seed $((42 + seed)) \
    --data ~/training/data/synthetic \
    --output "cnn_seed${seed}.pth" \
    --results_json "results_seed${seed}.json" \
    > train_${seed}.log 2>&1 &
  echo "Started training process $seed (seed=$((42 + seed)))"
done

echo "All 10 processes started. Monitor with: tail -f train_*.log"
wait
echo "All processes complete"
```

### Parallel Data Generation

Generate synthetic training data using multiple cores:

```bash
# Generate 100K images using 8 parallel processes
for i in {0..7}; do
  offset=$((i * 12500))
  python3 generate_synthetic.py \
    --output ~/training/data/synthetic \
    --samples 12500 \
    --img_size 128,72 \
    --start_index $offset \
    > gen_${i}.log 2>&1 &
done
wait
```

**Note:** `--start_index` ensures non-overlapping sample IDs across processes.

### Collect Results

```bash
# View all results
cat results_*.json | jq -s 'sort_by(.best_pixel_error)'

# Download results locally
scp -i ~/.ssh/ml-training.pem ubuntu@<ip>:~/training/scripts/results_*.json ./results/

# Download best model
best=$(cat results_*.json | jq -rs 'sort_by(.best_pixel_error)[0].seed')
scp -i ~/.ssh/ml-training.pem ubuntu@<ip>:~/training/scripts/cnn_seed${best}.pth ./
```

### Creating a New AMI

After generating data or installing packages, save as AMI:

```bash
# Create AMI from running instance
aws ec2 create-image \
  --instance-id <instance-id> \
  --name "ml-training-100k-$(date +%Y%m%d)" \
  --description "Training AMI with 100K synthetic images"
```

### Cost Comparison

| Approach | Instance | Time | Cost |
|----------|----------|------|------|
| 10 sequential runs | c6i.2xlarge | ~5 hours | ~$0.50-1.70 |
| **10 parallel runs** | c6i.2xlarge | ~30 min | **~$0.05-0.17** |
| 10 parallel (t3.medium × 10) | 10 instances | ~30 min | ~$0.50 |

**Key insight:** Running 10 processes on one 8-vCPU instance is cheaper than 10 separate instances because:
1. No overhead per instance (launch time, IP allocation)
2. Shared memory for data loading
3. Single EBS volume instead of 10

## Wikipedia ZIM Archive

Offline Wikipedia archive for training data generation (renders pages without internet).

**File:** `wikipedia_en_top_maxi_2025-09.zim` (7.2 GB, top 100K articles with images)

**Locations:**
- Local: `/Volumes/DevBlueB/kiwix/wikipedia_en_top_maxi_2025-09.zim`
- AWS: `/mnt/data/home/ubuntu/wikipedia_en_top_maxi_2025-09.zim`

**AWS Volume:** `vol-0e39424e89b42bf75` (tagged `ml-training-data`, 50GB)
- Snapshot: `snap-08aefb347e9f1f7c0`
- Contains: Wikipedia ZIM + old synthetic training images
- Availability Zone: us-east-1f

## 1080p Wiki Training Data Generation (2025-12-14)

**Goal:** Generate 1M training images from Wikipedia pages with CSS 3D perspective transforms.

### Current State

**Local generation running:**
- 12 parallel workers on `/Volumes/SamsungBlue/ml-training/wiki_training/`
- Target: 1,000,000 images (100K pages × 10 samples each)
- Rate: ~1,200 samples/min with 12 workers
- Output: 1920×1080 JPEG images + JSON corner labels

**S3 Upload running:**
- Bucket: `s3://ml-training-wiki-homography`
- Sync running in background

### Scripts

**Data Generation:**
- `experiments/generate_wiki_training_css3d.py` - CSS 3D transforms in browser (crisp text at any angle)
- `experiments/generate_wiki_training.py` - OpenCV warpPerspective (faster but aliased text)

**Training:**
- `experiments/train_corners_1080p.py` - 1.58M parameter CNN for 1920×1080 input

### Commands

```bash
# Check generation progress
ps aux | grep -c "[g]enerate_wiki" && echo "workers" && \
find /Volumes/SamsungBlue/ml-training/wiki_training -name "*.jpg" | wc -l && echo "images"

# Start 12 parallel workers for 1M samples
for i in {0..11}; do
  start_id=$((i * 83340))
  pages=8333
  outdir="/Volumes/SamsungBlue/ml-training/wiki_training/worker_${i}"
  mkdir -p "$outdir/images" "$outdir/labels"
  nice -n 10 python3 generate_wiki_training_css3d.py \
    --pages $pages --transforms 9 --start_id $start_id --output "$outdir" \
    > "$outdir/progress.log" 2>&1 &
done

# Upload to S3
aws s3 sync /Volumes/SamsungBlue/ml-training/wiki_training s3://ml-training-wiki-homography/wiki_training/

# Train 1080p model
python3 train_corners_1080p.py --data /Volumes/SamsungBlue/ml-training/wiki_training --epochs 50

# Check S3 upload
aws s3 ls s3://ml-training-wiki-homography/wiki_training/ --recursive --summarize | tail -5
```

### CNN Architecture (1080p) - CURRENT CHOICE

**This is the chosen architecture for corner prediction training.**

```
Input: 1920×1080×1 grayscale
Conv 7×7, 32, stride 2  → 960×540×32      (1,600 params)
Conv 3×3, 64, stride 2  → 480×270×64      (18,496 params)
Conv 3×3, 128, stride 2 → 240×135×128     (73,856 params)
Conv 3×3, 256, stride 2 → 120×67×256      (295,168 params)
Conv 3×3, 512, stride 2 → 60×33×512       (1,180,160 params)
GlobalAveragePool       → 512
Dense                   → 8 corners        (4,104 params)
─────────────────────────────────────────────────────────
Total: 1.58M parameters
```

### Storage

- **Local:** `/Volumes/SamsungBlue/ml-training/wiki_training/` (~180GB for 1M samples)
- **S3:** `s3://ml-training-wiki-homography/wiki_training/` (~$4/month storage)

### AWS GPU Training Instance (Active)

- **Instance:** `i-0563f8c98bdc5d818` (g5.xlarge)
- **IP:** `98.91.248.103`
- **GPU:** NVIDIA A10G (24GB VRAM)
- **AMI:** Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Ubuntu 22.04)
- **Cost:** ~$1.00/hr on-demand, ~$0.30-0.40/hr spot
- **Data:** `~/data/wiki_training` (synced from S3)

**Training commands:**
```bash
# SSH in
ssh -i ~/.ssh/ml-training.pem ubuntu@98.91.248.103

# Check GPU
nvidia-smi

# Run training
python3 train_corners_1080p.py --data ~/data/wiki_training --epochs 50 --lr 0.002

# Monitor download progress
find ~/data -name "*.jpg" | wc -l
```

### AWS Free Tier Instance

- Instance: `i-0b895b27eb2d0c5da` (t2.micro)
- Note: Crashes running Chromium (OOM with 1GB RAM). Use for file transfers only, not generation.
- Mount data volume: `sudo mount /dev/xvdf1 /mnt/data`
