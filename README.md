# OpenCV Pipeline Editor

A visual node-based editor for creating OpenCV image processing pipelines. Built with Java, SWT, and OpenCV.

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

### Color
- Grayscale
- HSV
- Threshold
- Adaptive Threshold
- Invert

### Morphology
- Erode
- Dilate
- Morph Open
- Morph Close

### Other
- Resize
- Gain

## License

MIT License
