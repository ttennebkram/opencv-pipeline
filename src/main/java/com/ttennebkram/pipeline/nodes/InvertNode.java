package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.registry.NodeInfo;
import org.eclipse.swt.SWT;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Mat;

/**
 * Invert effect node.
 */
@NodeInfo(
    name = "Invert",
    category = "Basic",
    aliases = {}
)
public class InvertNode extends ProcessingNode {
    public InvertNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Invert", x, y);
    }

    @Override
    public Mat process(Mat input) {
        if (!enabled || input == null || input.empty()) {
            return input;
        }
        Mat output = new Mat();
        org.opencv.core.Core.bitwise_not(input, output);
        return output;
    }

    @Override
    public String getDescription() {
        return "Invert Colors\ncv2.bitwise_not(src)";
    }

    @Override
    public String getDisplayName() {
        return "Invert";
    }

    @Override
    public String getCategory() {
        return "Basic";
    }

    @Override
    protected Runnable addPropertiesContent(Shell dialog, int columns) {
        // Method signature
        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText(getDescription());
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));

        // Separator
        Label sep = new Label(dialog, SWT.SEPARATOR | SWT.HORIZONTAL);
        sep.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        new Label(dialog, SWT.NONE).setText("Inverts all pixel values (negative image).\nNo parameters to configure.");

        return null; // No extra save action needed
    }

    @Override
    public void serializeProperties(JsonObject json) {
        // No properties to serialize
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        // No properties to deserialize
    }
}
