# OpenCV Pipeline Editor

A visual node-based editor for creating OpenCV image processing pipelines. Built with OpenCV, Java, SWT, and Eclipse GEF.

## Quick Start

Download the latest release and run:

```bash
# Download from releases
curl -LO https://github.com/ttennebkram/opencv-pipeline/releases/latest/download/opencv-pipeline.jar

# Run (macOS)
java -XstartOnFirstThread -jar opencv-pipeline.jar
```

Requires Java 17+ (ARM64 on macOS).

## Features

- **Visual Pipeline Editor**: Drag-and-drop nodes on a canvas to build image processing pipelines
- **Multiple Source Types**: Webcam capture, image files, video files
- **Processing Nodes**: Blur, edge detection, color conversion, morphological operations, and more
- **Real-time Preview**: See the output of selected nodes as the pipeline runs
- **Threaded Execution**: Each node runs in its own thread with queue-based communication
- **Save/Load Pipelines**: Save your pipelines to JSON files and reload them later

## Requirements

- Java 17 or higher (tested with Temurin 21 ARM64)
- Maven
- OpenCV 4.x (native library must be available)

## Building

Build the executable jar:

```bash
mvn clean package
```

This creates `target/opencv-pipeline.jar` with all dependencies bundled.

## Running

### From the jar file

```bash
java -XstartOnFirstThread -jar target/opencv-pipeline.jar
```

Note: The `-XstartOnFirstThread` flag is required on macOS for SWT applications.

### For development

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
