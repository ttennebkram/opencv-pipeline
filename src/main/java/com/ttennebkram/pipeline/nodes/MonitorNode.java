package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.registry.NodeInfo;
import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
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
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Monitor Properties");
        dialog.setLayout(new GridLayout(1, false));

        Label infoLabel = new Label(dialog, SWT.NONE);
        infoLabel.setText("This node passes input through unchanged.\nUseful for previewing outputs.");
        GridData gd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        infoLabel.setLayoutData(gd);

        Button okBtn = new Button(dialog, SWT.PUSH);
        okBtn.setText("OK");
        dialog.setDefaultButton(okBtn);
        okBtn.setLayoutData(new GridData(SWT.CENTER, SWT.CENTER, false, false));
        okBtn.addListener(SWT.Selection, e -> dialog.dispose());

        dialog.pack();
        Point cursor = shell.getDisplay().getCursorLocation();
        dialog.setLocation(cursor.x, cursor.y);
        dialog.open();
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
