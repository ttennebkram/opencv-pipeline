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
