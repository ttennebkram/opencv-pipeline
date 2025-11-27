package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.registry.NodeInfo;
import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Core;
import org.opencv.core.Mat;

/**
 * Bitwise NOT node - performs bitwise NOT on an image.
 * Single input, produces inverted bit output.
 */
@NodeInfo(name = "BitwiseNot", category = "Filter", aliases = {"Bitwise NOT"})
public class BitwiseNotNode extends ProcessingNode {

    public BitwiseNotNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Bitwise NOT", x, y);
    }

    @Override
    public Mat process(Mat input) {
        if (!enabled || input == null || input.empty()) {
            return input;
        }

        Mat output = new Mat();
        Core.bitwise_not(input, output);
        return output;
    }

    @Override
    public String getDescription() {
        return "Bitwise NOT\ncv2.bitwise_not(img)";
    }

    @Override
    public String getDisplayName() {
        return "Bitwise NOT";
    }

    @Override
    public String getCategory() {
        return "Filter";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("NOT Properties");
        dialog.setLayout(new GridLayout(1, false));

        // Node name field
        Text nameText = addNameField(dialog, 1);

        // Description
        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText(getDescription() + "\n\nInverts all bits in the input image.");
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sigLabel.setLayoutData(sigGd);

        // OK button
        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(1, true));
        GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
        buttonComp.setLayoutData(gd);

        Button okBtn = new Button(buttonComp, SWT.PUSH);
        okBtn.setText("OK");
        dialog.setDefaultButton(okBtn);
        okBtn.addListener(SWT.Selection, e -> {
            saveNameField(nameText);
            dialog.dispose();
            notifyChanged();
        });

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
