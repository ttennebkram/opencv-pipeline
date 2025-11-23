package com.example.pipeline.nodes;

import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

/**
 * Shi-Tomasi Corners detection node.
 */
public class ShiTomasiCornersNode extends ProcessingNode {
    private int maxCorners = 100;
    private int qualityLevel = 1; // 0.01 * 100
    private int minDistance = 10;
    private int blockSize = 3;
    private int markerSize = 5;
    private int colorR = 0, colorG = 255, colorB = 0;

    public ShiTomasiCornersNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Shi-Tomasi", x, y);
    }

    // Getters/setters for serialization
    public int getMaxCorners() { return maxCorners; }
    public void setMaxCorners(int v) { maxCorners = v; }
    public int getQualityLevel() { return qualityLevel; }
    public void setQualityLevel(int v) { qualityLevel = v; }
    public int getMinDistance() { return minDistance; }
    public void setMinDistance(int v) { minDistance = v; }
    public int getBlockSize() { return blockSize; }
    public void setBlockSize(int v) { blockSize = v; }
    public int getMarkerSize() { return markerSize; }
    public void setMarkerSize(int v) { markerSize = v; }
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

        // Create output image (color)
        Mat result = new Mat();
        if (input.channels() == 1) {
            Imgproc.cvtColor(input, result, Imgproc.COLOR_GRAY2BGR);
        } else {
            result = input.clone();
        }

        // Detect corners using goodFeaturesToTrack
        MatOfPoint corners = new MatOfPoint();
        double quality = qualityLevel / 100.0;
        Imgproc.goodFeaturesToTrack(gray, corners, maxCorners, quality, minDistance, new Mat(), blockSize, false, 0.04);

        // Draw corners
        Scalar color = new Scalar(colorB, colorG, colorR);
        org.opencv.core.Point[] cornerArray = corners.toArray();
        for (org.opencv.core.Point corner : cornerArray) {
            Imgproc.circle(result, corner, markerSize, color, -1);
        }

        return result;
    }

    @Override
    public String getDescription() {
        return "Shi-Tomasi Corner Detection\ncv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance)";
    }

    @Override
    public String getDisplayName() {
        return "Shi-Tomasi";
    }

    @Override
    public String getCategory() {
        return "Detection";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Shi-Tomasi Corners Properties");
        dialog.setLayout(new GridLayout(3, false));

        // Method signature
        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText(getDescription());
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sigGd.horizontalSpan = 3;
        sigLabel.setLayoutData(sigGd);

        // Max Corners
        new Label(dialog, SWT.NONE).setText("Max Corners:");
        Scale maxScale = new Scale(dialog, SWT.HORIZONTAL);
        maxScale.setMinimum(1);
        maxScale.setMaximum(500);
        maxScale.setSelection(maxCorners);
        maxScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label maxLabel = new Label(dialog, SWT.NONE);
        maxLabel.setText(String.valueOf(maxCorners));
        maxScale.addListener(SWT.Selection, e -> maxLabel.setText(String.valueOf(maxScale.getSelection())));

        // Quality Level
        new Label(dialog, SWT.NONE).setText("Quality Level %:");
        Scale qualScale = new Scale(dialog, SWT.HORIZONTAL);
        qualScale.setMinimum(1);
        qualScale.setMaximum(100);
        qualScale.setSelection(qualityLevel);
        qualScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label qualLabel = new Label(dialog, SWT.NONE);
        qualLabel.setText(String.valueOf(qualityLevel));
        qualScale.addListener(SWT.Selection, e -> qualLabel.setText(String.valueOf(qualScale.getSelection())));

        // Min Distance
        new Label(dialog, SWT.NONE).setText("Min Distance:");
        Scale distScale = new Scale(dialog, SWT.HORIZONTAL);
        distScale.setMinimum(1);
        distScale.setMaximum(100);
        distScale.setSelection(minDistance);
        distScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label distLabel = new Label(dialog, SWT.NONE);
        distLabel.setText(String.valueOf(minDistance));
        distScale.addListener(SWT.Selection, e -> distLabel.setText(String.valueOf(distScale.getSelection())));

        // Block Size
        new Label(dialog, SWT.NONE).setText("Block Size:");
        Scale blockScale = new Scale(dialog, SWT.HORIZONTAL);
        blockScale.setMinimum(3);
        blockScale.setMaximum(15);
        blockScale.setSelection(blockSize);
        blockScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label blockLabel = new Label(dialog, SWT.NONE);
        blockLabel.setText(String.valueOf(blockSize));
        blockScale.addListener(SWT.Selection, e -> blockLabel.setText(String.valueOf(blockScale.getSelection())));

        // Marker Size
        new Label(dialog, SWT.NONE).setText("Marker Size:");
        Scale markerScale = new Scale(dialog, SWT.HORIZONTAL);
        markerScale.setMinimum(1);
        markerScale.setMaximum(15);
        markerScale.setSelection(markerSize);
        markerScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label markerLabel = new Label(dialog, SWT.NONE);
        markerLabel.setText(String.valueOf(markerSize));
        markerScale.addListener(SWT.Selection, e -> markerLabel.setText(String.valueOf(markerScale.getSelection())));

        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(2, true));
        GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
        gd.horizontalSpan = 3;
        buttonComp.setLayoutData(gd);

        Button okBtn = new Button(buttonComp, SWT.PUSH);
        okBtn.setText("OK");
        okBtn.addListener(SWT.Selection, e -> {
            maxCorners = maxScale.getSelection();
            qualityLevel = qualScale.getSelection();
            minDistance = distScale.getSelection();
            blockSize = blockScale.getSelection();
            markerSize = markerScale.getSelection();
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
