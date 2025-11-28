package com.ttennebkram.pipeline.fx;

import java.util.*;

/**
 * Registry of available node types for the JavaFX editor.
 * This is a simplified version of NodeRegistry that doesn't depend on SWT.
 */
public class FXNodeRegistry {

    public static class NodeType {
        public final String name;
        public final String displayName;
        public final String category;
        public final boolean isSource;
        public final boolean isDualInput;
        public final int outputCount;

        public NodeType(String name, String displayName, String category,
                        boolean isSource, boolean isDualInput, int outputCount) {
            this.name = name;
            this.displayName = displayName;
            this.category = category;
            this.isSource = isSource;
            this.isDualInput = isDualInput;
            this.outputCount = outputCount;
        }

        public NodeType(String name, String displayName, String category) {
            this(name, displayName, category, false, false, 1);
        }
    }

    private static final List<NodeType> nodeTypes = new ArrayList<>();
    private static final Map<String, List<NodeType>> byCategory = new LinkedHashMap<>();

    static {
        // Source nodes
        register("WebcamSource", "Webcam", "Source", true, false, 1);
        register("FileSource", "File", "Source", true, false, 1);
        register("BlankSource", "Blank", "Source", true, false, 1);

        // Basic processing
        register("Grayscale", "Grayscale", "Basic");
        register("Invert", "Invert", "Basic");
        register("Threshold", "Threshold", "Basic");
        register("AdaptiveThreshold", "Adaptive Threshold", "Basic");
        register("Gain", "Gain", "Basic");
        register("CLAHE", "CLAHE", "Basic");
        register("BitPlanesGrayscale", "Bit Planes Gray", "Basic");
        register("BitPlanesColor", "Bit Planes Color", "Basic");

        // Content/Drawing
        register("Rectangle", "Rectangle", "Content");
        register("Circle", "Circle", "Content");
        register("Ellipse", "Ellipse", "Content");
        register("Line", "Line", "Content");
        register("Arrow", "Arrow", "Content");
        register("Text", "Text", "Content");

        // Blur
        register("GaussianBlur", "Gaussian Blur", "Blur");
        register("MedianBlur", "Median Blur", "Blur");
        register("BilateralFilter", "Bilateral", "Blur");
        register("BoxBlur", "Box Blur", "Blur");
        register("MeanShift", "Mean Shift", "Blur");

        // Edge detection
        register("CannyEdge", "Canny", "Edges");
        register("Sobel", "Sobel", "Edges");
        register("Laplacian", "Laplacian", "Edges");
        register("Scharr", "Scharr", "Edges");

        // Filter
        register("ColorInRange", "Color In Range", "Filter");
        register("BitwiseNot", "Bitwise NOT", "Filter");
        register("Filter2D", "Filter 2D", "Filter");
        register("FFTLowPass", "FFT Low-Pass", "Filter");
        register("FFTHighPass", "FFT High-Pass", "Filter");
        register("FFTLowPass4", "FFT Low-Pass 4", "Filter", false, false, 4);
        register("FFTHighPass4", "FFT High-Pass 4", "Filter", false, false, 4);

        // Morphology
        register("Erode", "Erode", "Morphology");
        register("Dilate", "Dilate", "Morphology");
        register("MorphOpen", "Morph Open", "Morphology");
        register("MorphClose", "Morph Close", "Morphology");
        register("MorphologyEx", "Morphology Ex", "Morphology");

        // Transform
        register("WarpAffine", "Warp Affine", "Transform");
        register("Crop", "Crop", "Transform");

        // Detection
        register("BlobDetector", "Blob Detector", "Detection");
        register("ConnectedComponents", "Connected Components", "Detection");
        register("HoughCircles", "Hough Circles", "Detection");
        register("HoughLines", "Hough Lines", "Detection");
        register("HarrisCorners", "Harris Corners", "Detection");
        register("ShiTomasi", "Shi-Tomasi Corners", "Detection");
        register("Contours", "Contours", "Detection");
        register("SIFTFeatures", "SIFT Features", "Detection");
        register("ORBFeatures", "ORB Features", "Detection");
        register("MatchTemplate", "Match Template", "Detection", false, true, 1);

        // Dual Input
        register("AddClamp", "Add (Clamp)", "Dual Input", false, true, 1);
        register("SubtractClamp", "Subtract (Clamp)", "Dual Input", false, true, 1);
        register("AddWeighted", "Add Weighted", "Dual Input", false, true, 1);
        register("BitwiseAnd", "Bitwise AND", "Dual Input", false, true, 1);
        register("BitwiseOr", "Bitwise OR", "Dual Input", false, true, 1);
        register("BitwiseXor", "Bitwise XOR", "Dual Input", false, true, 1);

        // Visualization
        register("Histogram", "Histogram", "Visualization");

        // Utility
        register("Clone", "Clone", "Utility", false, false, 2);
        register("Monitor", "Monitor", "Utility");
        register("Container", "Container", "Utility");
    }

    private static void register(String name, String displayName, String category) {
        register(name, displayName, category, false, false, 1);
    }

    private static void register(String name, String displayName, String category,
                                  boolean isSource, boolean isDualInput, int outputCount) {
        NodeType type = new NodeType(name, displayName, category, isSource, isDualInput, outputCount);
        nodeTypes.add(type);
        byCategory.computeIfAbsent(category, k -> new ArrayList<>()).add(type);
    }

    /**
     * Get all categories in display order.
     */
    public static List<String> getCategories() {
        return new ArrayList<>(byCategory.keySet());
    }

    /**
     * Get all node types in a category.
     */
    public static List<NodeType> getNodesInCategory(String category) {
        return byCategory.getOrDefault(category, Collections.emptyList());
    }

    /**
     * Get a node type by name.
     */
    public static NodeType getNodeType(String name) {
        for (NodeType type : nodeTypes) {
            if (type.name.equals(name) || type.displayName.equals(name)) {
                return type;
            }
        }
        return null;
    }

    /**
     * Get all node types.
     */
    public static List<NodeType> getAllNodeTypes() {
        return Collections.unmodifiableList(nodeTypes);
    }
}
