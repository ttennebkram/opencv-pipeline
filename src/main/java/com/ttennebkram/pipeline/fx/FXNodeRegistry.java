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
        public final String buttonName;  // Shorter name for toolbar buttons (null = use displayName)
        public final String category;
        public final boolean isSource;
        public final boolean isDualInput;
        public final boolean isContainer;
        public final int outputCount;

        public NodeType(String name, String displayName, String buttonName, String category,
                        boolean isSource, boolean isDualInput, boolean isContainer, int outputCount) {
            this.name = name;
            this.displayName = displayName;
            this.buttonName = buttonName;
            this.category = category;
            this.isSource = isSource;
            this.isDualInput = isDualInput;
            this.isContainer = isContainer;
            this.outputCount = outputCount;
        }

        public NodeType(String name, String displayName, String category,
                        boolean isSource, boolean isDualInput, boolean isContainer, int outputCount) {
            this(name, displayName, null, category, isSource, isDualInput, isContainer, outputCount);
        }

        public NodeType(String name, String displayName, String category) {
            this(name, displayName, null, category, false, false, false, 1);
        }

        /**
         * Get the name to display on toolbar buttons.
         */
        public String getButtonName() {
            return buttonName != null ? buttonName : displayName;
        }
    }

    private static final List<NodeType> nodeTypes = new ArrayList<>();
    private static final Map<String, List<NodeType>> byCategory = new LinkedHashMap<>();

    static {
        // Source nodes
        register("WebcamSource", "Webcam Source", "Sources", true, false, 1);
        register("FileSource", "File Source", "Sources", true, false, 1);
        register("BlankSource", "Blank Source", "Sources", true, false, 1);

        // Basic processing
        register("Grayscale", "Grayscale/Color Convert", "Basic");
        register("Invert", "Invert", "Basic");
        register("Threshold", "Threshold", "Basic");
        register("AdaptiveThreshold", "Adaptive Threshold", "Basic");
        register("Gain", "Gain", "Basic");
        register("CLAHE", "CLAHE Contrast", "Basic");
        register("BitPlanesGrayscale", "Bit Planes Gray", "Basic");
        register("BitPlanesColor", "Bit Planes Color", "Basic");

        // Blur
        register("GaussianBlur", "Gaussian Blur", "Blur");
        register("MedianBlur", "Median Blur", "Blur");
        register("BilateralFilter", "Bilateral", "Blur");
        register("BoxBlur", "Box Blur", "Blur");
        register("MeanShift", "Mean Shift", "Blur");

        // Content/Drawing
        register("Rectangle", "Rectangle", "Content");
        register("Circle", "Circle", "Content");
        register("Ellipse", "Ellipse", "Content");
        register("Line", "Line", "Content");
        register("Arrow", "Arrow", "Content");
        register("Text", "Text", "Content");

        // Edge detection (button shows short name, node title shows full name)
        register("CannyEdge", "Canny Edges", "Canny", "Edges");
        register("Sobel", "Sobel Edges", "Sobel", "Edges");
        register("Laplacian", "Laplacian Edges", "Laplacian", "Edges");
        register("Scharr", "Scharr Edges", "Scharr", "Edges");

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
        register("Monitor", "Monitor/Passthrough", "Utility");
        register("Container", "Container/Sub-diagram", "Utility", false, false, true, 1);

        // Container I/O nodes (only shown in container editor)
        register("ContainerInput", "Input", "Container I/O", true, false, 1);
        register("ContainerOutput", "Output", "Container I/O");

        // Sort nodes within each category alphabetically by display name
        for (List<NodeType> nodes : byCategory.values()) {
            nodes.sort((a, b) -> a.displayName.compareToIgnoreCase(b.displayName));
        }
    }

    private static void register(String name, String displayName, String category) {
        register(name, displayName, null, category, false, false, false, 1);
    }

    private static void register(String name, String displayName, String buttonName, String category) {
        register(name, displayName, buttonName, category, false, false, false, 1);
    }

    private static void register(String name, String displayName, String category,
                                  boolean isSource, boolean isDualInput, int outputCount) {
        register(name, displayName, null, category, isSource, isDualInput, false, outputCount);
    }

    private static void register(String name, String displayName, String category,
                                  boolean isSource, boolean isDualInput, boolean isContainer, int outputCount) {
        register(name, displayName, null, category, isSource, isDualInput, isContainer, outputCount);
    }

    private static void register(String name, String displayName, String buttonName, String category,
                                  boolean isSource, boolean isDualInput, boolean isContainer, int outputCount) {
        NodeType type = new NodeType(name, displayName, buttonName, category, isSource, isDualInput, isContainer, outputCount);
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
     * Get categories excluding certain ones (e.g., exclude "Container I/O" from main editor).
     */
    public static List<String> getCategoriesExcluding(String... excludeCategories) {
        List<String> result = new ArrayList<>();
        java.util.Set<String> excluded = new java.util.HashSet<>(java.util.Arrays.asList(excludeCategories));
        for (String cat : byCategory.keySet()) {
            if (!excluded.contains(cat)) {
                result.add(cat);
            }
        }
        return result;
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
