# OpenCV Pipeline Editor

A visual node-based editor for creating OpenCV image processing pipelines. Built with Java, JavaFX, and OpenCV.

Not just a drawing program - **it's live!** Each node you see is **an actual thread**, and each connection between them is a queue. It even supports sub-pipeline diagrams (containers).

![Screenshot](screenshots/pipeline.png)


## Quick Start

Download the JAR for your platform from the [releases page](https://github.com/ttennebkram/opencv-pipeline/releases):

| Platform | Download |
|----------|----------|
| macOS Apple Silicon (M1-M5) | [opencv-pipeline-mac-aarch64.jar](https://github.com/ttennebkram/opencv-pipeline/releases/latest/download/opencv-pipeline-mac-aarch64.jar) |
| macOS Intel (x86_64) | [opencv-pipeline-mac.jar](https://github.com/ttennebkram/opencv-pipeline/releases/latest/download/opencv-pipeline-mac.jar) |
| Linux x86_64 | [opencv-pipeline-linux.jar](https://github.com/ttennebkram/opencv-pipeline/releases/latest/download/opencv-pipeline-linux.jar) |
| Linux ARM64 (Raspberry Pi 4 & 5) | [opencv-pipeline-linux-aarch64.jar](https://github.com/ttennebkram/opencv-pipeline/releases/latest/download/opencv-pipeline-linux-aarch64.jar) |
| Windows x86_64 (Intel/AMD) | [opencv-pipeline-win.jar](https://github.com/ttennebkram/opencv-pipeline/releases/latest/download/opencv-pipeline-win.jar) |

Then run:

```bash
java -jar opencv-pipeline-<platform>.jar
```

Or to automatically load and start a pipeline:

```bash
java -jar opencv-pipeline-<platform>.jar pipeline.json --start
```

**Note:** No Windows ARM64 support. OpenCV doesn't provide Windows ARM64 native libraries.

Requires Java 17+.

## Features

- **Visual Pipeline Editor**: Drag-and-drop nodes on a canvas to build image processing pipelines
- **Multiple Source Types**: Webcam capture, image files, video files, blank canvas
- **60+ Processing Nodes**: Blur, edge detection, FFT filters, morphological operations, feature detection, and more
- **Real-time Preview**: See the output of each node as the pipeline runs
- **Threaded Execution**: Each node runs in its own thread with queue-based communication
- **Container Nodes**: Create reusable sub-pipelines that can be nested
- **Save/Load Pipelines**: Save your pipelines to JSON files and reload them later
- **Cross-Platform**: Runs on macOS, Linux, and Windows (platform-specific JARs)


## Requirements

- Java 17 or higher
- A webcam (optional, for webcam source nodes)

## Building from Source

```bash
# Build for current platform (macOS Apple Silicon by default)
mvn clean package

# Build for a specific platform
mvn clean package -Pmac           # macOS Intel
mvn clean package -Plinux         # Linux x86_64
mvn clean package -Plinux-aarch64 # Linux ARM64 (Raspberry Pi)
mvn clean package -Pwin           # Windows x86_64

# Build all platforms at once
./build-all-platforms.sh
```

The uber-jar is created at `target/opencv-pipeline.jar`.

For development (automatically reopens the last saved file):

```bash
mvn compile exec:exec
```

Or any of:

```bash
mvn compile exec:exec@start
mvn compile exec:exec@run -Dpipeline.args="--start"
mvn compile exec:exec@run -Dpipeline.args="pipeline.json"
mvn compile exec:exec@run -Dpipeline.args="pipeline.json --start"
```


To compile and run the app use either of:

```bash
mvn compile exec:exec
mvn compile exec:exec@start
```

## Usage

1. **Add Nodes**: Click on node types in the toolbar to add them to the canvas
2. **Connect Nodes**: Drag from an output port to an input port to create connections
3. **Configure Nodes**: Double-click or right-click nodes to open properties dialog.  Container nodes use double-click to open the sub-pipeline, which is usually what users want; so use right-click to set the file property.
4. **Run Pipeline**: Click "Start Pipeline" to begin processing
5. **View Output**: Each node shows a live thumbnail of its output.  Single-click on the thumbnail to see a larger version of that image in the Preview pane.

**macOS Note**: macOS enforces a "Java" application menu which is mostly useless. You may find it simpler to just ignore it and use the **File** menu inside the window instead, just like on Windows or Linux.

### Keyboard Shortcuts

- **Delete/Backspace**: Delete selected nodes or connections
- **Ctrl+S / Cmd+S**: Save pipeline
- **Ctrl+O / Cmd+O**: Open pipeline
- **Ctrl+N / Cmd+N**: New pipeline

### Save Warning

The "unsaved changes" warning triggers when you add, delete, move, or edit properties of nodes and connections. However, some runtime state (like preview selections and stats updates) are not tracked and won't prompt a save warning.

## Node Categories

### Source
- **Webcam** - Live camera capture
- **File** - Load image or video files
- **Blank** - Solid color canvas

### Color
- Grayscale, Invert, Gain, CLAHE
- Color In Range (HSV filtering)
- Bit Planes (Grayscale & Color)

### Blur
- Gaussian Blur, Median Blur, Box Blur
- Bilateral Filter, Mean Shift Filter

### Edge Detection
- Canny Edges, Sobel, Laplacian, Scharr

### Morphological
- Erode, Dilate, Morph Open, Morph Close
- Morphology Ex (advanced operations)

### Threshold
- Threshold, Adaptive Threshold

### FFT (Frequency Domain)
- FFT Low Pass, FFT High Pass
- FFT Low Pass 4, FFT High Pass 4 (with spectrum visualization)

### Detection
- Blob Detector, Connected Components
- Contours, Histogram
- Hough Circles, Hough Lines
- Harris Corners, Shi-Tomasi Corners
- SIFT Features, ORB Features

### Transform
- Warp Affine, Crop, Resize
- Filter2D (custom convolution kernels)
- Resize

### Drawing
- Rectangle, Circle, Ellipse
- Line, Arrow, Text

### Combine
- Add Weighted (blend two images)
- Match Template
- Clone (duplicate stream)

### Bitwise
- Bitwise AND, OR, XOR, NOT

### Flow
- Container (sub-pipeline)
- Monitor (display output)

## Position Coordinates

Drawing nodes (shapes, text) support negative position values for coordinates relative to image edges:
- Positive values: offset from left/top edge (e.g., `50` = 50 pixels from left)
- Negative values: offset from right/bottom edge (e.g., `-1` = rightmost/bottom pixel, `-50` = 50 pixels from right/bottom)

This allows positioning elements relative to image dimensions that may vary at runtime.

## Performance Note
- FFT takes over a second to filter a full resolution 12 megapixel image
- If using FFT filters, we suggest you use the Resize node first to shrink the image donw

## Architecture

The application uses a modular processor architecture:
- Each node type is implemented as a self-contained `FXProcessor` class
- Processors handle their own UI (properties dialog), serialization, and image processing
- New node types can be added by creating a new processor class with the `@FXProcessorInfo` annotation
- Runtime discovery automatically registers all processors

## ML Experiments

The `experiments` branch contains experimental deep learning code for training CNNs with GPU acceleration.

```bash
git checkout experiments
mvn exec:exec@ml -Dml.class=MLFacade
```

Features:
- **Automatic backend selection**: Uses Python+GPU (MPS/CUDA) if available, falls back to pure Java
- **Cross-platform inference**: Train anywhere, run inference in pure Java (~0.3ms/image)
- **Step-by-step Python bridge**: Java orchestrates training while Python/PyTorch handles GPU computation

See `CLAUDE.md` for detailed ML documentation.

## License

MIT License
