package com.example.pipeline.nodes;

import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

/**
 * Bilateral Filter node.
 */
public class BilateralFilterNode extends ProcessingNode {
    private int diameter = 9;
    private int sigmaColor = 75;
    private int sigmaSpace = 75;

    public BilateralFilterNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Bilateral Filter", x, y);
    }

    // Getters/setters for serialization
    public int getDiameter() { return diameter; }
    public void setDiameter(int v) { diameter = v; }
    public int getSigmaColor() { return sigmaColor; }
    public void setSigmaColor(int v) { sigmaColor = v; }
    public int getSigmaSpace() { return sigmaSpace; }
    public void setSigmaSpace(int v) { sigmaSpace = v; }

    @Override
    public Mat process(Mat input) {
        if (!enabled || input == null || input.empty()) {
            return input;
        }
        Mat output = new Mat();
        Imgproc.bilateralFilter(input, output, diameter, sigmaColor, sigmaSpace);
        return output;
    }

    @Override
    public String getDescription() {
        return "Bilateral Filter\ncv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)";
    }

    @Override
    public String getDisplayName() {
        return "Bilateral Blur";
    }

    @Override
    public String getCategory() {
        return "Blur";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Bilateral Filter Properties");
        dialog.setLayout(new GridLayout(3, false));

        // Method signature
        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText(getDescription());
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sigGd.horizontalSpan = 3;
        sigLabel.setLayoutData(sigGd);

        // Separator
        Label sep = new Label(dialog, SWT.SEPARATOR | SWT.HORIZONTAL);
        GridData sepGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sepGd.horizontalSpan = 3;
        sep.setLayoutData(sepGd);

        // Diameter
        new Label(dialog, SWT.NONE).setText("Diameter:");
        Scale dScale = new Scale(dialog, SWT.HORIZONTAL);
        dScale.setMinimum(1);
        dScale.setMaximum(25);
        dScale.setSelection(diameter);
        dScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label dLabel = new Label(dialog, SWT.NONE);
        dLabel.setText(String.valueOf(diameter));
        dScale.addListener(SWT.Selection, e -> dLabel.setText(String.valueOf(dScale.getSelection())));

        // Sigma Color
        new Label(dialog, SWT.NONE).setText("Sigma Color:");
        Scale scScale = new Scale(dialog, SWT.HORIZONTAL);
        scScale.setMinimum(1);
        scScale.setMaximum(200);
        scScale.setSelection(sigmaColor);
        scScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label scLabel = new Label(dialog, SWT.NONE);
        scLabel.setText(String.valueOf(sigmaColor));
        scScale.addListener(SWT.Selection, e -> scLabel.setText(String.valueOf(scScale.getSelection())));

        // Sigma Space
        new Label(dialog, SWT.NONE).setText("Sigma Space:");
        Scale ssScale = new Scale(dialog, SWT.HORIZONTAL);
        ssScale.setMinimum(1);
        ssScale.setMaximum(200);
        ssScale.setSelection(sigmaSpace);
        ssScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label ssLabel = new Label(dialog, SWT.NONE);
        ssLabel.setText(String.valueOf(sigmaSpace));
        ssScale.addListener(SWT.Selection, e -> ssLabel.setText(String.valueOf(ssScale.getSelection())));

        // Buttons
        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(2, true));
        GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
        gd.horizontalSpan = 3;
        buttonComp.setLayoutData(gd);

        Button okBtn = new Button(buttonComp, SWT.PUSH);
        okBtn.setText("OK");
        okBtn.addListener(SWT.Selection, e -> {
            diameter = dScale.getSelection();
            sigmaColor = scScale.getSelection();
            sigmaSpace = ssScale.getSelection();
            dialog.dispose();
            notifyChanged();
        });

        Button cancelBtn = new Button(buttonComp, SWT.PUSH);
        cancelBtn.setText("Cancel");
        cancelBtn.addListener(SWT.Selection, e -> dialog.dispose());

        dialog.pack();
        Point cursor = shell.getDisplay().getCursorLocation();
        dialog.setLocation(cursor.x, cursor.y);
        dialog.open();
    }
}
