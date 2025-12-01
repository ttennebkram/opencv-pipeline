# OpenCV Pipeline Editor

A visual node-based editor for creating OpenCV image processing pipelines. Built with Java, JavaFX, and OpenCV.

Not just a drawing program - **it's live!** Each node you see is **an actual thread**, and each connection between them is a queue. It even supports sub-pipeline diagrams (containers).

![Screenshot](screenshots/pipeline.png)


## Quick Start

Download `opencv-pipeline.jar` from the [releases page](https://github.com/ttennebkram/opencv-pipeline/releases) and run:

```bash
java -jar opencv-pipeline.jar
```

That's it! This one JAR works on all **supported** platforms:
- **macOS** Apple Silicon (M1-M5)
- **macOS** Intel (x86_64)
- **Linux** x86_64
- **Linux** ARM64 (Raspberry Pi 4 & 5)
- **Windows** x86_64 (Intel/AMD)
- *No Windows ARM64*

Sorry, no Windows on ARM support. This project uses **OpenCV** which relies on **JNI** (Java Native Interface) for high-performance image processing, and OpenCV doesn't provide Windows ARM64 native libraries. We wanted to support it, but it's out of our control.

Requires Java 17+.

## Features

- **Visual Pipeline Editor**: Drag-and-drop nodes on a canvas to build image processing pipelines
- **Multiple Source Types**: Webcam capture, image files, video files, blank canvas
- **60+ Processing Nodes**: Blur, edge detection, FFT filters, morphological operations, feature detection, and more
- **Real-time Preview**: See the output of each node as the pipeline runs
- **Threaded Execution**: Each node runs in its own thread with queue-based communication
- **Container Nodes**: Create reusable sub-pipelines that can be nested
- **Save/Load Pipelines**: Save your pipelines to JSON files and reload them later
- **Cross-Platform**: Single JAR runs on macOS, Linux, and Windows


## Requirements

- Java 17 or higher
- A webcam (optional, for webcam source nodes)

## Building from Source

```bash
mvn clean package
```

The uber-jar is created at `target/opencv-pipeline.jar`.

For development:

```bash
mvn compile exec:exec
```

## Usage

1. **Add Nodes**: Click on node types in the toolbar to add them to the canvas
2. **Connect Nodes**: Drag from an output port to an input port to create connections
3. **Configure Nodes**: Double-click a node to open its properties dialog
4. **Run Pipeline**: Click "Start Pipeline" to begin processing
5. **View Output**: Each node shows a live thumbnail of its output

### Keyboard Shortcuts

- **Delete/Backspace**: Delete selected nodes or connections
- **Ctrl+S / Cmd+S**: Save pipeline
- **Ctrl+O / Cmd+O**: Open pipeline
- **Ctrl+N / Cmd+N**: New pipeline

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
- Canny Edge, Sobel, Laplacian, Scharr

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

## Architecture

The application uses a modular processor architecture:
- Each node type is implemented as a self-contained `FXProcessor` class
- Processors handle their own UI (properties dialog), serialization, and image processing
- New node types can be added by creating a new processor class with the `@FXProcessorInfo` annotation
- Runtime discovery automatically registers all processors

## License

MIT License
