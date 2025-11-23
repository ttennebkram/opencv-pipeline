package com.example.pipeline.nodes;

import org.eclipse.swt.widgets.Display;
import org.eclipse.swt.widgets.Shell;
import org.opencv.core.Mat;

/**
 * Placeholder node for unknown/unrecognized node types.
 * Passes input through unchanged.
 */
public class UnknownNode extends ProcessingNode {
    private String originalType;

    public UnknownNode(Display display, Shell shell, int x, int y, String originalType) {
        super(display, shell, "Unknown: " + originalType, x, y);
        this.originalType = originalType;
    }

    // For serialization
    public String getOriginalType() { return originalType; }
    public void setOriginalType(String v) { originalType = v; }

    @Override
    public Mat process(Mat input) {
        // Pass through unchanged
        return input;
    }

    @Override
    public String getDescription() {
        return "Unknown node type: " + originalType + "\nPasses input through unchanged";
    }

    @Override
    public void showPropertiesDialog() {
        // No properties to edit
    }
}
