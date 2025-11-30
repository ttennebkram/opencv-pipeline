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
        public final String description;  // Method signature / description
        public final boolean isSource;
        public final boolean isDualInput;
        public final boolean isContainer;
        public final int outputCount;

        public NodeType(String name, String displayName, String buttonName, String category, String description,
                        boolean isSource, boolean isDualInput, boolean isContainer, int outputCount) {
            this.name = name;
            this.displayName = displayName;
            this.buttonName = buttonName;
            this.category = category;
            this.description = description;
            this.isSource = isSource;
            this.isDualInput = isDualInput;
            this.isContainer = isContainer;
            this.outputCount = outputCount;
        }

        public NodeType(String name, String displayName, String category,
                        boolean isSource, boolean isDualInput, boolean isContainer, int outputCount) {
            this(name, displayName, null, category, null, isSource, isDualInput, isContainer, outputCount);
        }

        public NodeType(String name, String displayName, String category) {
            this(name, displayName, null, category, null, false, false, false, 1);
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
        register("WebcamSource", "Webcam Source", "Sources", "Webcam capture\nVideoCapture.read()", true, false, 1);
        register("FileSource", "File Source", "Sources", "Image/Video file source\nImgcodecs.imread() / VideoCapture", true, false, 1);
        register("BlankSource", "Blank Source", "Sources", "Solid color image generator", true, false, 1);

        // Basic processing
        register("Grayscale", "Grayscale/Color Convert", "Basic", "Color space conversion\nImgproc.cvtColor(src, dst, code)");
        register("Invert", "Invert", "Basic", "Invert colors\nCore.bitwise_not(src, dst)");
        register("Threshold", "Threshold", "Basic", "Binary threshold\nImgproc.threshold(src, dst, thresh, maxval, type)");
        register("AdaptiveThreshold", "Adaptive Threshold", "Basic", "Adaptive threshold\nImgproc.adaptiveThreshold(...)");
        register("Gain", "Gain", "Basic", "Brightness/Gain adjustment\nCore.multiply(src, gain, dst)");
        register("CLAHE", "CLAHE Contrast", "Basic", "Contrast Limited AHE\nImgproc.createCLAHE().apply(src, dst)");
        register("BitPlanesGrayscale", "Bit Planes Gray", "Basic", "Bit plane decomposition (grayscale)\nBit masking with gain");
        register("BitPlanesColor", "Bit Planes Color", "Basic", "Bit plane decomposition (color)\nBit masking per channel with gain");

        // Blur
        register("GaussianBlur", "Gaussian Blur", "Blur", "Gaussian blur\nImgproc.GaussianBlur(src, dst, ksize, sigma)");
        register("MedianBlur", "Median Blur", "Blur", "Median blur\nImgproc.medianBlur(src, dst, ksize)");
        register("BilateralFilter", "Bilateral Blur", "Blur", "Bilateral filter (edge-preserving)\nImgproc.bilateralFilter(src, dst, d, sigmaColor, sigmaSpace)");
        register("BoxBlur", "Box Blur", "Blur", "Box blur (average)\nImgproc.blur(src, dst, ksize)");
        register("MeanShift", "Mean Shift Blur", "Blur", "Mean shift filtering\nImgproc.pyrMeanShiftFiltering(src, dst, sp, sr)");

        // Content/Drawing
        register("Rectangle", "Rectangle", "Content", "Draw rectangle\nImgproc.rectangle(img, pt1, pt2, color, thickness)");
        register("Circle", "Circle", "Content", "Draw circle\nImgproc.circle(img, center, radius, color, thickness)");
        register("Ellipse", "Ellipse", "Content", "Draw ellipse\nImgproc.ellipse(img, center, axes, angle, ...)");
        register("Line", "Line", "Content", "Draw line\nImgproc.line(img, pt1, pt2, color, thickness)");
        register("Arrow", "Arrow", "Content", "Draw arrow\nImgproc.arrowedLine(img, pt1, pt2, color, thickness)");
        register("Text", "Text", "Content", "Draw text\nImgproc.putText(img, text, org, font, scale, color)");

        // Edge detection (button shows short name, node title shows full name)
        register("CannyEdge", "Canny Edges", "Canny", "Edges", "Canny edge detection\nImgproc.Canny(src, dst, threshold1, threshold2)");
        register("Sobel", "Sobel Edges", "Sobel", "Edges", "Sobel derivatives\nImgproc.Sobel(src, dst, ddepth, dx, dy, ksize)");
        register("Laplacian", "Laplacian Edges", "Laplacian", "Edges", "Laplacian operator\nImgproc.Laplacian(src, dst, ddepth)");
        register("Scharr", "Scharr Edges", "Scharr", "Edges", "Scharr derivatives\nImgproc.Scharr(src, dst, ddepth, dx, dy)");

        // Filter
        register("ColorInRange", "Color In Range", "Filter", "Color range mask\nCore.inRange(src, lowerb, upperb, dst)");
        register("BitwiseNot", "Bitwise NOT", "Filter", "Bitwise NOT\nCore.bitwise_not(src, dst)");
        register("Filter2D", "Filter 2D", "Filter", "2D convolution\nImgproc.filter2D(src, dst, ddepth, kernel)");
        register("FFTLowPass", "FFT Low-Pass", "Filter", "FFT Low-Pass Filter\nCore.dft() / Core.idft()");
        register("FFTHighPass", "FFT High-Pass", "Filter", "FFT High-Pass Filter\nCore.dft() / Core.idft()");
        register("FFTLowPass4", "FFT Low-Pass 4", "Filter", "FFT Low-Pass (4 outputs)\nCore.dft() / Core.idft()", false, false, 4);
        register("FFTHighPass4", "FFT High-Pass 4", "Filter", "FFT High-Pass (4 outputs)\nCore.dft() / Core.idft()", false, false, 4);

        // Morphology
        register("Erode", "Erode", "Morphology", "Erosion\nImgproc.erode(src, dst, kernel, iterations)");
        register("Dilate", "Dilate", "Morphology", "Dilation\nImgproc.dilate(src, dst, kernel, iterations)");
        register("MorphOpen", "Morph Open", "Morphology", "Morphological opening (erode+dilate)\nImgproc.morphologyEx(src, dst, MORPH_OPEN, kernel)");
        register("MorphClose", "Morph Close", "Morphology", "Morphological closing (dilate+erode)\nImgproc.morphologyEx(src, dst, MORPH_CLOSE, kernel)");
        register("MorphologyEx", "Morphology Ex", "Morphology", "Extended morphology operations\nImgproc.morphologyEx(src, dst, op, kernel)");

        // Transform
        register("WarpAffine", "Warp Affine", "Transform", "Affine transformation\nImgproc.warpAffine(src, dst, M, dsize)");
        register("Crop", "Crop", "Transform", "Crop region of interest\nMat.submat(roi)");
        register("Resize", "Resize", "Transform", "Resize image\nImgproc.resize(src, dst, dsize)");

        // Detection
        register("BlobDetector", "Blob Detector", "Detection", "Blob detection\nSimpleBlobDetector.detect(image, keypoints)");
        register("ConnectedComponents", "Connected Components", "Detection", "Connected component labeling\nImgproc.connectedComponentsWithStats(...)");
        register("HoughCircles", "Hough Circles", "Detection", "Hough circle detection\nImgproc.HoughCircles(src, circles, method, dp, minDist, ...)");
        register("HoughLines", "Hough Lines", "Detection", "Hough line detection\nImgproc.HoughLinesP(src, lines, rho, theta, threshold, ...)");
        register("HarrisCorners", "Harris Corners", "Detection", "Harris corner detection\nImgproc.cornerHarris(src, dst, blockSize, ksize, k)");
        register("ShiTomasi", "Shi-Tomasi Corners", "Detection", "Shi-Tomasi corner detection\nImgproc.goodFeaturesToTrack(src, corners, maxCorners, ...)");
        register("Contours", "Contours", "Detection", "Contour detection\nImgproc.findContours(src, contours, hierarchy, mode, method)");
        register("SIFTFeatures", "SIFT Features", "Detection", "SIFT feature detection\nSIFT.detectAndCompute(image, mask, keypoints, descriptors)");
        register("ORBFeatures", "ORB Features", "Detection", "ORB feature detection\nORB.detectAndCompute(image, mask, keypoints, descriptors)");
        register("MatchTemplate", "Match Template", "Detection", "Template matching\nImgproc.matchTemplate(image, templ, result, method)", false, true, 1);

        // Dual Input
        register("AddClamp", "Add w/Clamp", "Dual Input", "Add images with clamping\nCore.add(src1, src2, dst)", false, true, 1);
        register("SubtractClamp", "Subtract (Clamp)", "Dual Input", "Subtract images with clamping\nCore.subtract(src1, src2, dst)", false, true, 1);
        register("AddWeighted", "Add Weighted", "Dual Input", "Weighted addition (blend)\nCore.addWeighted(src1, alpha, src2, beta, gamma, dst)", false, true, 1);
        register("BitwiseAnd", "Bitwise AND", "Dual Input", "Bitwise AND\nCore.bitwise_and(src1, src2, dst)", false, true, 1);
        register("BitwiseOr", "Bitwise OR", "Dual Input", "Bitwise OR\nCore.bitwise_or(src1, src2, dst)", false, true, 1);
        register("BitwiseXor", "Bitwise XOR", "Dual Input", "Bitwise XOR\nCore.bitwise_xor(src1, src2, dst)", false, true, 1);

        // Visualization
        register("Histogram", "Histogram", "Visualization", "Histogram visualization\nImgproc.calcHist() / custom render");

        // Utility
        register("Clone", "Clone", "Utility", "Clone to multiple outputs\nMat.clone()", false, false, 2);
        register("Monitor", "Monitor/Passthrough", "Utility", "Monitor (passthrough)\nPasses input unchanged for preview");
        register("Container", "Container/Sub-diagram", "Utility", "Container sub-diagram\nEncapsulates a pipeline", false, false, true, 1);

        // Container I/O nodes (only shown in container editor)
        register("ContainerInput", "Input", "Container I/O", "Container input\nReceives data from parent pipeline", true, false, 1);
        register("ContainerOutput", "Output", "Container I/O", "Container output\nSends data to parent pipeline");

        // Sort nodes within each category alphabetically by display name
        for (List<NodeType> nodes : byCategory.values()) {
            nodes.sort((a, b) -> a.displayName.compareToIgnoreCase(b.displayName));
        }
    }

    private static void register(String name, String displayName, String category) {
        register(name, displayName, null, category, null, false, false, false, 1);
    }

    private static void register(String name, String displayName, String category, String description) {
        register(name, displayName, null, category, description, false, false, false, 1);
    }

    // Note: For buttonName with no description, use the 5-arg version with null description
    private static void register(String name, String displayName, String buttonName, String category, String description) {
        register(name, displayName, buttonName, category, description, false, false, false, 1);
    }

    private static void register(String name, String displayName, String category,
                                  boolean isSource, boolean isDualInput, int outputCount) {
        register(name, displayName, null, category, null, isSource, isDualInput, false, outputCount);
    }

    private static void register(String name, String displayName, String category, String description,
                                  boolean isSource, boolean isDualInput, int outputCount) {
        register(name, displayName, null, category, description, isSource, isDualInput, false, outputCount);
    }

    private static void register(String name, String displayName, String category,
                                  boolean isSource, boolean isDualInput, boolean isContainer, int outputCount) {
        register(name, displayName, null, category, null, isSource, isDualInput, isContainer, outputCount);
    }

    private static void register(String name, String displayName, String category, String description,
                                  boolean isSource, boolean isDualInput, boolean isContainer, int outputCount) {
        register(name, displayName, null, category, description, isSource, isDualInput, isContainer, outputCount);
    }

    private static void register(String name, String displayName, String buttonName, String category, String description,
                                  boolean isSource, boolean isDualInput, boolean isContainer, int outputCount) {
        NodeType type = new NodeType(name, displayName, buttonName, category, description, isSource, isDualInput, isContainer, outputCount);
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
