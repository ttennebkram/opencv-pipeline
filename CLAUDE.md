# OpenCV Pipeline Editor

A graphical node-based pipeline editor for OpenCV image processing operations.

**IMPORTANT: Always read README.md as well when starting a session.** The README contains user-facing documentation including command line options, feature descriptions, and platform support details that provide important context.

## IMPORTANT: Git Policy

**DO NOT create git commits or push without explicit user approval.**

- You may SUGGEST a commit when work is complete
- Wait for the user to test and confirm before committing
- Never commit code in a potentially broken state
- The user prefers to verify changes work before committing

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
mvn exec:exec@run -Dpipeline.args="file.json -a"
mvn exec:exec@run -Dpipeline.args="file.json -a --fullscreen_node_name Monitor"
mvn exec:exec@run -Dpipeline.args="file.json -a --camera_fps 15 --max_time 60"

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

## Raspberry Pi Camera Support

The webcam source supports Raspberry Pi cameras (CSI modules like OV5647) via `rpicam-still` polling.

### How it works

On Raspberry Pi 5, the camera uses libcamera which is not compatible with OpenCV's V4L2 backend. OpenCV can *open* the V4L2 device but cannot *read* frames from it. The solution:

1. `FXWebcamSource.open()` first checks if `rpicam-still` is available
2. If found, it starts `rpicam-still --timelapse 100` which writes JPEG frames to a temp file every 100ms
3. `captureFrame()` reads the latest JPEG from the temp file using `Imgcodecs.imread()`
4. Falls back to standard V4L2 VideoCapture for USB webcams

### Key files

- `FXWebcamSource.java` - `tryOpenWithRpicam()`, `captureFrameFromRpicam()`, `stopRpicamProcess()`

### Alternatives considered but not viable

- **GStreamer pipeline**: OpenCV (openpnp build) is not compiled with GStreamer support
- **rpicam-vid TCP/UDP streaming**: OpenCV couldn't read the streams reliably
- **Building custom OpenCV**: Would break cross-platform portability

### USB webcams

USB webcams work normally via V4L2 and don't need the rpicam workaround. The code automatically detects which method to use.

## Command Line Options

Key options for automated/scripted use:
- `-a`, `--auto_start`, `--auto_run` - Start pipeline immediately after loading
- `--fullscreen_node_name NAME` - Open fullscreen preview of named node (live, updates continuously)
- `--max_time SECONDS` - Exit after specified time (timer thread is daemon, so app exits cleanly)
- `--camera_index`, `--camera_resolution`, `--camera_fps`, `--camera_mirror` - Override all webcam sources

Options can appear before or after the pipeline filename.

## Known Issues

- Watch for file handle leaks - always close InputStreams, use try-with-resources
- FFT nodes have complex multi-channel output modes (grayscale vs 4-channel)
- Container nodes have their own sub-pipeline serialized inline

## Recent Bug Fixes Worth Noting

- **Node label serialization** (v2.4.0): Custom labels that match the node type name (e.g., naming a Monitor node "Monitor") now save/restore correctly. Previously the condition `!savedLabel.equals(type)` rejected these.
- **Fullscreen preview race conditions**: The fullscreen window uses an `AnimationTimer` to continuously update from `node.previewImage`, and properly requests focus to avoid beeps on first click.
- **Clean exit with --max_time**: The timer thread is set as daemon (`setDaemon(true)`) so the JVM exits when windows close.
- **Pipeline stop race condition**: Null checks added for `pipelineExecutor` in callbacks that may fire after pipeline stops.
