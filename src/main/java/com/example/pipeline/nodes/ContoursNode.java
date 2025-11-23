package com.example.pipeline.nodes;

import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

/**
 * Contours detection node.
 */
public class ContoursNode extends ProcessingNode {
    private int thresholdValue = 127;
    private int retrievalMode = 0; // 0=EXTERNAL, 1=LIST, 2=CCOMP, 3=TREE
    private int thickness = 2;
    private int colorR = 0, colorG = 255, colorB = 0;

    private static final String[] RETRIEVAL_MODES = {"External", "List", "Two-level", "Tree"};
    private static final int[] RETRIEVAL_VALUES = {
        Imgproc.RETR_EXTERNAL, Imgproc.RETR_LIST, Imgproc.RETR_CCOMP, Imgproc.RETR_TREE
    };

    public ContoursNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Contours", x, y);
    }

    // Getters/setters for serialization
    public int getThresholdValue() { return thresholdValue; }
    public void setThresholdValue(int v) { thresholdValue = v; }
    public int getRetrievalMode() { return retrievalMode; }
    public void setRetrievalMode(int v) { retrievalMode = v; }
    public int getThickness() { return thickness; }
    public void setThickness(int v) { thickness = v; }
    public int getColorR() { return colorR; }
    public void setColorR(int v) { colorR = v; }
    public int getColorG() { return colorG; }
    public void setColorG(int v) { colorG = v; }
    public int getColorB() { return colorB; }
    public void setColorB(int v) { colorB = v; }

    @Override
    public Mat process(Mat input) {
        if (!enabled || input == null || input.empty()) return input;

        // Convert to grayscale
        Mat gray = new Mat();
        if (input.channels() == 3) {
            Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
        } else {
            gray = input.clone();
        }

        // Apply threshold
        Mat binary = new Mat();
        Imgproc.threshold(gray, binary, thresholdValue, 255, Imgproc.THRESH_BINARY);

        // Find contours
        java.util.List<org.opencv.core.MatOfPoint> contours = new java.util.ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(binary, contours, hierarchy,
            RETRIEVAL_VALUES[retrievalMode], Imgproc.CHAIN_APPROX_SIMPLE);

        // Create output image (color)
        Mat result = new Mat();
        if (input.channels() == 1) {
            Imgproc.cvtColor(input, result, Imgproc.COLOR_GRAY2BGR);
        } else {
            result = input.clone();
        }

        // Draw contours
        Scalar color = new Scalar(colorB, colorG, colorR);
        Imgproc.drawContours(result, contours, -1, color, thickness);

        return result;
    }

    @Override
    public String getDescription() {
        return "Contour Detection\ncv2.findContours(image, mode, method)";
    }

    @Override
    public String getDisplayName() {
        return "Contours";
    }

    @Override
    public String getCategory() {
        return "Detection";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Contours Properties");
        dialog.setLayout(new GridLayout(3, false));

        // Method signature
        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText(getDescription());
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sigGd.horizontalSpan = 3;
        sigLabel.setLayoutData(sigGd);

        // Threshold
        new Label(dialog, SWT.NONE).setText("Threshold:");
        Scale threshScale = new Scale(dialog, SWT.HORIZONTAL);
        threshScale.setMinimum(0);
        threshScale.setMaximum(255);
        threshScale.setSelection(thresholdValue);
        threshScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label threshLabel = new Label(dialog, SWT.NONE);
        threshLabel.setText(String.valueOf(thresholdValue));
        threshScale.addListener(SWT.Selection, e -> threshLabel.setText(String.valueOf(threshScale.getSelection())));

        // Retrieval Mode
        new Label(dialog, SWT.NONE).setText("Retrieval Mode:");
        Combo modeCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        modeCombo.setItems(RETRIEVAL_MODES);
        modeCombo.select(retrievalMode);
        GridData comboGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
        comboGd.horizontalSpan = 2;
        modeCombo.setLayoutData(comboGd);

        // Thickness
        new Label(dialog, SWT.NONE).setText("Line Thickness:");
        Scale thickScale = new Scale(dialog, SWT.HORIZONTAL);
        thickScale.setMinimum(1);
        thickScale.setMaximum(10);
        thickScale.setSelection(thickness);
        thickScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label thickLabel = new Label(dialog, SWT.NONE);
        thickLabel.setText(String.valueOf(thickness));
        thickScale.addListener(SWT.Selection, e -> thickLabel.setText(String.valueOf(thickScale.getSelection())));

        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(2, true));
        GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
        gd.horizontalSpan = 3;
        buttonComp.setLayoutData(gd);

        Button okBtn = new Button(buttonComp, SWT.PUSH);
        okBtn.setText("OK");
        okBtn.addListener(SWT.Selection, e -> {
            thresholdValue = threshScale.getSelection();
            retrievalMode = modeCombo.getSelectionIndex();
            thickness = thickScale.getSelection();
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
