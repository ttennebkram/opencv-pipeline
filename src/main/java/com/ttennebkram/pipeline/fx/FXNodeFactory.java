package com.ttennebkram.pipeline.fx;

import java.lang.reflect.Constructor;
import java.util.HashMap;
import java.util.Map;

/**
 * Factory for creating pipeline nodes in the JavaFX environment.
 * This factory creates "headless" processing nodes that don't depend on SWT,
 * or wraps existing node classes with JavaFX-compatible interfaces.
 *
 * The approach is:
 * 1. FXNode handles visual representation (position, selection, connections)
 * 2. This factory creates the actual processing node
 * 3. The two are linked via a unique ID
 */
public class FXNodeFactory {

    // Map of node type names to their processing node classes
    private static final Map<String, Class<?>> nodeClasses = new HashMap<>();

    // Map to track FXNode -> processing node associations
    private static final Map<Integer, Object> processingNodes = new HashMap<>();

    private static boolean initialized = false;

    /**
     * Initialize the node class mappings lazily.
     * This allows the factory to work even when node classes aren't compiled.
     */
    private static synchronized void ensureInitialized() {
        if (initialized) return;
        initialized = true;
        registerNodeMappings();
    }

    private static void registerNodeMappings() {
        // Source nodes
        registerNode("WebcamSource", "com.ttennebkram.pipeline.nodes.WebcamSourceNode");
        registerNode("FileSource", "com.ttennebkram.pipeline.nodes.FileSourceNode");
        registerNode("BlankSource", "com.ttennebkram.pipeline.nodes.BlankSourceNode");

        // Basic processing
        registerNode("Grayscale", "com.ttennebkram.pipeline.nodes.GrayscaleNode");
        registerNode("Invert", "com.ttennebkram.pipeline.nodes.InvertNode");
        registerNode("Threshold", "com.ttennebkram.pipeline.nodes.ThresholdNode");
        registerNode("AdaptiveThreshold", "com.ttennebkram.pipeline.nodes.AdaptiveThresholdNode");
        registerNode("Gain", "com.ttennebkram.pipeline.nodes.GainNode");
        registerNode("CLAHE", "com.ttennebkram.pipeline.nodes.CLAHENode");
        registerNode("BitPlanesGrayscale", "com.ttennebkram.pipeline.nodes.BitPlanesGrayscaleNode");
        registerNode("BitPlanesColor", "com.ttennebkram.pipeline.nodes.BitPlanesColorNode");

        // Content/Drawing
        registerNode("Rectangle", "com.ttennebkram.pipeline.nodes.RectangleNode");
        registerNode("Circle", "com.ttennebkram.pipeline.nodes.CircleNode");
        registerNode("Ellipse", "com.ttennebkram.pipeline.nodes.EllipseNode");
        registerNode("Line", "com.ttennebkram.pipeline.nodes.LineNode");
        registerNode("Arrow", "com.ttennebkram.pipeline.nodes.ArrowNode");
        registerNode("Text", "com.ttennebkram.pipeline.nodes.TextNode");

        // Blur
        registerNode("GaussianBlur", "com.ttennebkram.pipeline.nodes.GaussianBlurNode");
        registerNode("MedianBlur", "com.ttennebkram.pipeline.nodes.MedianBlurNode");
        registerNode("BilateralFilter", "com.ttennebkram.pipeline.nodes.BilateralFilterNode");
        registerNode("BoxBlur", "com.ttennebkram.pipeline.nodes.BoxBlurNode");
        registerNode("MeanShift", "com.ttennebkram.pipeline.nodes.MeanShiftFilterNode");

        // Edge detection
        registerNode("CannyEdge", "com.ttennebkram.pipeline.nodes.CannyEdgeNode");
        registerNode("Sobel", "com.ttennebkram.pipeline.nodes.SobelNode");
        registerNode("Laplacian", "com.ttennebkram.pipeline.nodes.LaplacianNode");
        registerNode("Scharr", "com.ttennebkram.pipeline.nodes.ScharrNode");

        // Filter
        registerNode("ColorInRange", "com.ttennebkram.pipeline.nodes.ColorInRangeNode");
        registerNode("BitwiseNot", "com.ttennebkram.pipeline.nodes.BitwiseNotNode");
        registerNode("Filter2D", "com.ttennebkram.pipeline.nodes.Filter2DNode");
        registerNode("FFTLowPass", "com.ttennebkram.pipeline.nodes.FFTLowPassFilterNode");
        registerNode("FFTHighPass", "com.ttennebkram.pipeline.nodes.FFTHighPassFilterNode");
        registerNode("FFTLowPass4", "com.ttennebkram.pipeline.nodes.FFTLowPass4Node");
        registerNode("FFTHighPass4", "com.ttennebkram.pipeline.nodes.FFTHighPass4Node");

        // Morphology
        registerNode("Erode", "com.ttennebkram.pipeline.nodes.ErodeNode");
        registerNode("Dilate", "com.ttennebkram.pipeline.nodes.DilateNode");
        registerNode("MorphOpen", "com.ttennebkram.pipeline.nodes.MorphOpenNode");
        registerNode("MorphClose", "com.ttennebkram.pipeline.nodes.MorphCloseNode");
        registerNode("MorphologyEx", "com.ttennebkram.pipeline.nodes.MorphologyExNode");

        // Transform
        registerNode("WarpAffine", "com.ttennebkram.pipeline.nodes.WarpAffineNode");
        registerNode("Crop", "com.ttennebkram.pipeline.nodes.CropNode");

        // Detection
        registerNode("BlobDetector", "com.ttennebkram.pipeline.nodes.BlobDetectorNode");
        registerNode("ConnectedComponents", "com.ttennebkram.pipeline.nodes.ConnectedComponentsNode");
        registerNode("HoughCircles", "com.ttennebkram.pipeline.nodes.HoughCirclesNode");
        registerNode("HoughLines", "com.ttennebkram.pipeline.nodes.HoughLinesNode");
        registerNode("HarrisCorners", "com.ttennebkram.pipeline.nodes.HarrisCornersNode");
        registerNode("ShiTomasi", "com.ttennebkram.pipeline.nodes.ShiTomasiCornersNode");
        registerNode("Contours", "com.ttennebkram.pipeline.nodes.ContoursNode");
        registerNode("SIFTFeatures", "com.ttennebkram.pipeline.nodes.SIFTFeaturesNode");
        registerNode("ORBFeatures", "com.ttennebkram.pipeline.nodes.ORBFeaturesNode");
        registerNode("MatchTemplate", "com.ttennebkram.pipeline.nodes.MatchTemplateNode");

        // Dual Input
        registerNode("AddClamp", "com.ttennebkram.pipeline.nodes.AddClampNode");
        registerNode("SubtractClamp", "com.ttennebkram.pipeline.nodes.SubtractClampNode");
        registerNode("AddWeighted", "com.ttennebkram.pipeline.nodes.AddWeightedNode");
        registerNode("BitwiseAnd", "com.ttennebkram.pipeline.nodes.BitwiseAndNode");
        registerNode("BitwiseOr", "com.ttennebkram.pipeline.nodes.BitwiseOrNode");
        registerNode("BitwiseXor", "com.ttennebkram.pipeline.nodes.BitwiseXorNode");

        // Visualization
        registerNode("Histogram", "com.ttennebkram.pipeline.nodes.HistogramNode");

        // Utility
        registerNode("Clone", "com.ttennebkram.pipeline.nodes.CloneNode");
        registerNode("Monitor", "com.ttennebkram.pipeline.nodes.MonitorNode");
        registerNode("Container", "com.ttennebkram.pipeline.nodes.ContainerNode");
    }

    private static void registerNode(String typeName, String className) {
        try {
            Class<?> clazz = Class.forName(className);
            nodeClasses.put(typeName, clazz);
        } catch (ClassNotFoundException e) {
            // Class not found - may be expected if node isn't compiled yet
            System.err.println("Warning: Node class not found: " + className);
        }
    }

    /**
     * Check if a node type has an associated processing class.
     */
    public static boolean hasProcessingClass(String nodeType) {
        ensureInitialized();
        return nodeClasses.containsKey(nodeType);
    }

    /**
     * Get the processing class for a node type.
     */
    public static Class<?> getProcessingClass(String nodeType) {
        ensureInitialized();
        return nodeClasses.get(nodeType);
    }

    /**
     * Create an FXNode with appropriate settings based on node type.
     */
    public static FXNode createFXNode(String nodeType, double x, double y) {
        FXNodeRegistry.NodeType typeInfo = FXNodeRegistry.getNodeType(nodeType);
        FXNode node;

        if (typeInfo != null) {
            if (typeInfo.isSource) {
                node = FXNode.createSourceNode(typeInfo.displayName, typeInfo.name, x, y);
            } else if (typeInfo.isDualInput) {
                node = FXNode.createDualInputNode(typeInfo.displayName, typeInfo.name, x, y);
            } else if (typeInfo.outputCount > 1) {
                node = FXNode.createMultiOutputNode(typeInfo.displayName, typeInfo.name, x, y, typeInfo.outputCount);
            } else {
                node = new FXNode(typeInfo.displayName, typeInfo.name, x, y);
            }
            // Set container flag and color
            node.isContainer = typeInfo.isContainer;
            node.canBeDisabled = typeInfo.canBeDisabled;
            if (typeInfo.isContainer) {
                node.backgroundColor = NodeRenderer.COLOR_CONTAINER_NODE;
            }
        } else {
            node = new FXNode(nodeType, nodeType, x, y);
        }

        // Set boundary node properties for ContainerInput and ContainerOutput
        if ("ContainerInput".equals(nodeType)) {
            node.isBoundaryNode = true;
            node.hasInput = false;  // Source node - no input
            node.outputCount = 1;
            node.backgroundColor = NodeRenderer.COLOR_CONTAINER_NODE;
            node.height = NodeRenderer.NODE_HEIGHT;
        } else if ("ContainerOutput".equals(nodeType)) {
            node.isBoundaryNode = true;
            node.hasInput = true;   // Has input
            node.outputCount = 0;   // No outputs - this is a sink node
            node.backgroundColor = NodeRenderer.COLOR_CONTAINER_NODE;
            node.height = NodeRenderer.NODE_HEIGHT;
        }

        return node;
    }

    /**
     * Associate a processing node with an FXNode.
     */
    public static void setProcessingNode(FXNode fxNode, Object processingNode) {
        processingNodes.put(fxNode.id, processingNode);
    }

    /**
     * Get the processing node associated with an FXNode.
     */
    public static Object getProcessingNode(FXNode fxNode) {
        return processingNodes.get(fxNode.id);
    }

    /**
     * Remove the processing node association.
     */
    public static void removeProcessingNode(FXNode fxNode) {
        processingNodes.remove(fxNode.id);
    }

    /**
     * Get all registered node type names.
     */
    public static java.util.Set<String> getRegisteredTypes() {
        ensureInitialized();
        return nodeClasses.keySet();
    }
}
