package com.example.pipeline.nodes;

import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Mat;

/**
 * Invert effect node.
 */
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
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Invert Properties");
        dialog.setLayout(new GridLayout(1, false));

        // Method signature
        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText(getDescription());
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));

        // Separator
        Label sep = new Label(dialog, SWT.SEPARATOR | SWT.HORIZONTAL);
        sep.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        new Label(dialog, SWT.NONE).setText("Inverts all pixel values (negative image).\nNo parameters to configure.");

        Button okBtn = new Button(dialog, SWT.PUSH);
        okBtn.setText("OK");
        okBtn.setLayoutData(new GridData(SWT.CENTER, SWT.CENTER, true, false));
        okBtn.addListener(SWT.Selection, e -> dialog.dispose());

        dialog.pack();
        // Position dialog near cursor
        Point cursor = shell.getDisplay().getCursorLocation();
        dialog.setLocation(cursor.x, cursor.y);
        dialog.open();
    }
}
