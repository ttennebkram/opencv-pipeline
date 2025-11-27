package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.registry.NodeInfo;
import org.eclipse.swt.SWT;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Mat;

/**
 * Monitor node - passes input through unchanged.
 * Useful for monitoring intermediate outputs or tapping into connections.
 */
@NodeInfo(
    name = "Monitor",
    category = "Utility",
    aliases = {"Passthrough", "Tap"}
)
public class MonitorNode extends ProcessingNode {

    public MonitorNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Monitor", x, y);
    }

    @Override
    public Mat process(Mat input) {
        if (input == null || input.empty()) {
            return input;
        }
        // Just clone and return - no processing
        return input.clone();
    }

    @Override
    public String getDescription() {
        return "Monitor\nPasses input unchanged for preview";
    }

    @Override
    public String getDisplayName() {
        return "Monitor";
    }

    @Override
    public String getCategory() {
        return "Utility";
    }

    @Override
    protected int getPropertiesDialogColumns() {
        return 1;
    }

    @Override
    protected Runnable addPropertiesContent(Shell dialog, int columns) {
        Label infoLabel = new Label(dialog, SWT.NONE);
        infoLabel.setText("This node passes input through unchanged.\nUseful for previewing outputs.");
        GridData gd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        infoLabel.setLayoutData(gd);

        return null;
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
