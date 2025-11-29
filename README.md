# OpenCV Pipeline Editor

A visual node-based editor for creating OpenCV image processing pipelines. Built with OpenCV, Java, SWT, and Eclipse GEF.

![Screenshot](screenshots/pipeline.png)


## Quick Start

Download the JAR for your platform from the [releases page](https://github.com/ttennebkram/opencv-pipeline/releases) and run:

```bash
# macOS (Apple Silicon)
java -XstartOnFirstThread -jar opencv-pipeline-macos-aarch64.jar

# Linux
java -jar opencv-pipeline-linux-x86_64.jar

# Windows
java -jar opencv-pipeline-windows-x86_64.jar
```

Requires Java 17+.

## Features

- **Visual Pipeline Editor**: Drag-and-drop nodes on a canvas to build image processing pipelines
- **Multiple Source Types**: Webcam capture, image files, video files
- **Processing Nodes**: Blur, edge detection, color conversion, morphological operations, and more
- **Real-time Preview**: See the output of selected nodes as the pipeline runs
- **Threaded Execution**: Each node runs in its own thread with queue-based communication
- **Save/Load Pipelines**: Save your pipelines to JSON files and reload them later

## Platform Support

Pre-built JARs are available for:
- **macOS** (Apple Silicon and Intel)
- **Linux** (x86_64 and ARM64/Raspberry Pi)
- **Windows** (x86_64)

**Why separate JARs per platform?** The GUI uses SWT (Standard Widget Toolkit), which provides native OS widgets but requires platform-specific native libraries. Each JAR bundles the correct SWT natives for that platform. OpenCV also uses JNI for high-performance image processing, but its natives are bundled in a way that works across platforms.

## Requirements

- Java 17 or higher
- Maven (for building from source)

## Building

Build for your current platform (defaults to macOS ARM64):

```bash
mvn clean package
```

Build for a specific platform:

```bash
mvn clean package -P linux-aarch64   # Raspberry Pi 64-bit
mvn clean package -P linux-x86_64    # Linux desktop
mvn clean package -P macos-aarch64   # Apple Silicon (default)
mvn clean package -P macos-x86_64    # Intel Mac
mvn clean package -P windows-x86_64  # Windows
```

Build all platforms at once:

```bash
./build-all-platforms.sh
```

Output JARs are created in `target/` with platform-specific names.

## Running

### macOS

```bash
# Apple Silicon
java -XstartOnFirstThread -jar opencv-pipeline-macos-aarch64.jar

# Intel Mac
java -XstartOnFirstThread -jar opencv-pipeline-macos-x86_64.jar
```

Note: The `-XstartOnFirstThread` flag is required on macOS for SWT applications.

### Linux

```bash
# x86_64
java -jar opencv-pipeline-linux-x86_64.jar

# ARM64 (Raspberry Pi 64-bit)
java -jar opencv-pipeline-linux-aarch64.jar
```

If you encounter display issues on Wayland, try: `GDK_BACKEND=x11 java -jar ...`

### Windows

```bash
java -jar opencv-pipeline-windows-x86_64.jar
```

### For development (macOS)

```bash
mvn exec:exec
```

## Usage

1. **Add Nodes**: Click on node names in the left panel to create new nodes
2. **Connect Nodes**: Drag from one node to another to create connections
3. **Configure Nodes**: Double-click a node to open its properties dialog
4. **Run Pipeline**: Click "Start Pipeline" to begin processing
5. **Preview Output**: Select a node to see its output in the preview panel

## Node Categories

### Source
- Webcam
- File (image/video)
- Blank (solid color image)

### Basic
- Grayscale
- Invert
- Threshold
- Adaptive Threshold
- Gain
- CLAHE
- Color In Range
- Bit Planes Grayscale
- Bit Planes Color

### Blur
- Gaussian Blur
- Median Blur
- Bilateral Blur
- Box Blur
- Mean Shift Blur

### Edge Detection
- Canny Edge
- Sobel
- Laplacian
- Scharr

### Morphological
- Erode
- Dilate
- Morph Open
- Morph Close

### Detection
- Blob Detector
- Connected Components
- Hough Circles
- Hough Lines
- Harris Corners
- Shi-Tomasi Corners
- Contours
- SIFT Features

### Transform
- Warp Affine
- FFT Filter

### Content
- Shapes (rectangle, circle, ellipse, line, arrow)
- Text

## Position Coordinates

Content nodes (Shapes, Text) support negative position values for coordinates relative to image edges:
- Positive values: offset from left/top edge (e.g., `50` = 50 pixels from left)
- Negative values: offset from right/bottom edge (e.g., `-1` = rightmost/bottom pixel, `-50` = 50 pixels from right/bottom)

This allows positioning elements relative to image dimensions that may vary at runtime.

## License

MIT License
