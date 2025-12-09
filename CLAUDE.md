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
└── util/
    └── MatTracker.java          # Tracks OpenCV Mat allocations for leak detection
```

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
./r2 --output training_data/homography

# Generate specific amount
./r2 --samples_per_page 30 --pages 100 --output training_data/homography

# Estimate mode - show projections without generating
./r2 --estimate --pages 1000
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

8. **Output Resolution**:
   - Default: 1920×1080 (1080p) for high-quality source data
   - **IMPORTANT**: Resize significantly for training (224×224, 320×240, etc.)
   - Training on 1080p directly is wasteful and slow

### Training Recommendations

- **Train on corners_normalized**, not raw homography values
- Homography matrix values can range from -200M to +200M (extremely variable)
- Corners are bounded to [-1, +1] after normalization
- Network output: 8 values (4 corners × 2 coordinates)

### Implementation Plan

1. [x] Create training data generator (`experiments/HomographyTrainingDataGenerator.java`)
   - Uses JavaFX WebView to render Wikipedia pages
   - Random perspective transforms simulating paper on table
   - Shadow effects for realism
   - Dual video output (content + geometry)
2. [x] Create Python training script (`experiments/train_homography.py`)
   - CNN with conv backbone + regression head
   - Combined loss: MSE + reprojection error
   - Exports weights for Java inference
3. [ ] Generate large training dataset (1000+ pages × 10 samples = 10,000+ pairs)
4. [ ] Train and tune the model
5. [ ] Export trained model for Java inference
6. [ ] Create `HomographyEstimatorProcessor` - runs inference in pipeline

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
