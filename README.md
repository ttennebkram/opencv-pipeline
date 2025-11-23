# OpenCV Pipeline Editor

**THIS IS THE MAIN PROJECT** - A graphical node-based pipeline editor for OpenCV image processing operations.

## Running

```bash
cd /Users/mbennett/Dropbox/dev/opencv-pipeline-editor
mvn exec:exec
```

Requires ARM64 Java 21: `/Library/Java/JavaVirtualMachines/temurin-21-arm64.jdk`

## Features
- Node-based visual editor for image processing pipelines
- Drag and drop nodes
- Connect nodes with connectors
- Selection, deletion
- Persistence (save/load pipelines)

## Current Node Types (Implemented)
- ImageSource
- GaussianBlur
- Grayscale
- Invert
- Threshold
- Gain
- CannyEdge

## Node Types (TODO - Need Implementation)
- MedianBlur
- BilateralFilter
- Laplacian
- Sobel
- Erode
- Dilate

## Related Projects
- `/Users/mbennett/Dropbox/dev/webcam-filters/effects/opencv/` - Python OpenCV effects (source for porting)
- `/Users/mbennett/Dropbox/dev/gef-classic-github/` - GEF Classic (Draw2D library used for graphics)

## Porting Effects
Effects are being ported from the Python webcam-filters project. Each effect needs:
1. A node class extending `ProcessingNode`
2. Parameters UI (sliders, dropdowns)
3. OpenCV processing in the `process()` method
4. Entry in the `createNodeForType()` switch statement
