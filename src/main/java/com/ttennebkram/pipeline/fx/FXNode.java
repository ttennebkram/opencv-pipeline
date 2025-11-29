package com.ttennebkram.pipeline.fx;

import javafx.scene.image.Image;
import javafx.scene.paint.Color;

/**
 * Lightweight node data class for JavaFX rendering.
 * This is independent of SWT and holds only the data needed for display and interaction.
 */
public class FXNode {
    // Position and size
    public double x;
    public double y;
    public double width;
    public double height;

    // Display properties
    public String label;
    public String nodeType;
    public Color backgroundColor;
    public boolean selected;
    public boolean enabled;

    // Connection configuration
    public boolean hasInput;
    public boolean hasDualInput;
    public boolean isContainer;
    public boolean isBoundaryNode;  // Container I/O boundary nodes (Input/Output)
    public int outputCount;

    // Thumbnail image for displaying node output
    public Image thumbnail;

    // Counter for input/output frames processed
    public int inputCount = 0;
    public int outputCount1 = 0;  // Output 1 counter
    public int outputCount2 = 0;  // Output 2 counter (for multi-output nodes)
    public int outputCount3 = 0;  // Output 3 counter
    public int outputCount4 = 0;  // Output 4 counter

    // Thread priority and work stats (for backpressure display)
    public int threadPriority = 5;  // Default to NORM_PRIORITY
    public long workUnitsCompleted = 0;
    public double effectiveFps = 0;  // For source nodes only

    // Webcam-specific properties - default set dynamically when creating WebcamSource nodes
    public int cameraIndex = -1; // -1 means "auto-detect highest camera"

    // FileSource-specific properties
    public String filePath = "";
    public double fps = -1.0;  // -1 means "automatic" (1fps for images, normal for video)

    // Container-specific properties - internal nodes and connections
    public java.util.List<FXNode> innerNodes = new java.util.ArrayList<>();
    public java.util.List<FXConnection> innerConnections = new java.util.ArrayList<>();
    public String pipelineFilePath = "";  // Path to external pipeline JSON file for containers

    // Unique identifier for serialization
    public int id;
    private static int nextId = 1;

    /**
     * Generate a new unique ID for a node.
     * Used when loading inner nodes to avoid ID collisions with outer nodes.
     */
    public static int generateNewId() {
        return nextId++;
    }

    /**
     * Reassign a new unique ID to this node.
     */
    public void reassignId() {
        this.id = nextId++;
    }

    public FXNode(String label, String nodeType, double x, double y) {
        this.id = nextId++;
        this.label = label;
        this.nodeType = nodeType;
        this.x = x;
        this.y = y;
        this.width = NodeRenderer.NODE_WIDTH;
        this.height = NodeRenderer.NODE_HEIGHT;
        this.backgroundColor = Color.rgb(200, 220, 255);
        this.selected = false;
        this.enabled = true;
        this.hasInput = true;
        this.hasDualInput = false;
        this.isContainer = false;
        this.isBoundaryNode = false;
        this.outputCount = 1;
    }

    /**
     * Create a source node (wider, no input).
     */
    public static FXNode createSourceNode(String label, String nodeType, double x, double y) {
        FXNode node = new FXNode(label, nodeType, x, y);
        node.width = NodeRenderer.NODE_WIDTH;  // Same width as processing nodes
        node.height = NodeRenderer.SOURCE_NODE_HEIGHT;
        node.hasInput = false;
        node.backgroundColor = Color.rgb(200, 255, 200); // Green tint for sources
        return node;
    }

    /**
     * Create a dual-input node.
     */
    public static FXNode createDualInputNode(String label, String nodeType, double x, double y) {
        FXNode node = new FXNode(label, nodeType, x, y);
        node.hasDualInput = true;
        node.backgroundColor = Color.rgb(255, 220, 200); // Orange tint for dual-input
        return node;
    }

    /**
     * Create a multi-output node.
     */
    public static FXNode createMultiOutputNode(String label, String nodeType, double x, double y, int outputs) {
        FXNode node = new FXNode(label, nodeType, x, y);
        node.outputCount = outputs;
        node.backgroundColor = Color.rgb(220, 200, 255); // Purple tint for multi-output
        return node;
    }

    // Checkbox constants (must match NodeRenderer)
    private static final int CHECKBOX_SIZE = 12;
    private static final int CHECKBOX_MARGIN = 5;
    private static final int HELP_ICON_SIZE = 14;
    private static final int HELP_ICON_MARGIN = 5;

    /**
     * Check if a point is inside this node.
     */
    public boolean contains(double px, double py) {
        return px >= x && px <= x + width && py >= y && py <= y + height;
    }

    /**
     * Check if a point is on the enabled checkbox.
     */
    public boolean isOnCheckbox(double px, double py) {
        // First check if point is within node bounds
        if (!contains(px, py)) {
            return false;
        }
        double cbX = x + CHECKBOX_MARGIN;
        double cbY = y + CHECKBOX_MARGIN;
        return px >= cbX && px <= cbX + CHECKBOX_SIZE &&
               py >= cbY && py <= cbY + CHECKBOX_SIZE;
    }

    /**
     * Check if a point is on the help icon.
     */
    public boolean isOnHelpIcon(double px, double py) {
        // First check if point is within node bounds
        if (!contains(px, py)) {
            return false;
        }
        double iconX = x + width - HELP_ICON_SIZE - HELP_ICON_MARGIN;
        double iconY = y + HELP_ICON_MARGIN;
        return px >= iconX && px <= iconX + HELP_ICON_SIZE &&
               py >= iconY && py <= iconY + HELP_ICON_SIZE;
    }

    /**
     * Get the input connection point position.
     */
    public double[] getInputPoint(int index) {
        if (!hasInput) return null;
        double iy = y + height / 2;
        if (index == 1 && hasDualInput) {
            iy += 20;
        }
        return new double[]{x, iy};
    }

    /**
     * Get the output connection point position.
     */
    public double[] getOutputPoint(int index) {
        if (index < 0 || index >= outputCount) return null;
        double spacing = height / (outputCount + 1);
        return new double[]{x + width, y + spacing * (index + 1)};
    }

    /**
     * Check if a point is near an input connection point.
     */
    public int getInputPointAt(double px, double py, double tolerance) {
        if (!hasInput) return -1;
        double[] pt = getInputPoint(0);
        if (Math.abs(px - pt[0]) <= tolerance && Math.abs(py - pt[1]) <= tolerance) {
            return 0;
        }
        if (hasDualInput) {
            pt = getInputPoint(1);
            if (Math.abs(px - pt[0]) <= tolerance && Math.abs(py - pt[1]) <= tolerance) {
                return 1;
            }
        }
        return -1;
    }

    /**
     * Check if a point is near an output connection point.
     */
    public int getOutputPointAt(double px, double py, double tolerance) {
        for (int i = 0; i < outputCount; i++) {
            double[] pt = getOutputPoint(i);
            if (Math.abs(px - pt[0]) <= tolerance && Math.abs(py - pt[1]) <= tolerance) {
                return i;
            }
        }
        return -1;
    }
}
