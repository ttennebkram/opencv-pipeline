package com.example.pipeline.nodes;

import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

/**
 * Harris Corners detection node.
 */
public class HarrisCornersNode extends ProcessingNode {
    private int blockSize = 2;
    private int ksize = 3;
    private int thresholdPercent = 1; // 0.01 * 100
    private int markerSize = 5;
    private int colorR = 255, colorG = 0, colorB = 0;

    public HarrisCornersNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Harris Corners", x, y);
    }

    // Getters/setters for serialization
    public int getBlockSize() { return blockSize; }
    public void setBlockSize(int v) { blockSize = v; }
    public int getKsize() { return ksize; }
    public void setKsize(int v) { ksize = v; }
    public int getThresholdPercent() { return thresholdPercent; }
    public void setThresholdPercent(int v) { thresholdPercent = v; }
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

        // Convert to float
        Mat grayFloat = new Mat();
        gray.convertTo(grayFloat, CvType.CV_32F);

        // Apply Harris corner detection
        Mat harris = new Mat();
        Imgproc.cornerHarris(grayFloat, harris, blockSize, ksize, 0.04);

        // Normalize and convert to byte for thresholding
        Mat harrisNorm = new Mat();
        Core.normalize(harris, harrisNorm, 0, 255, Core.NORM_MINMAX);
        Mat harrisNormScaled = new Mat();
        harrisNorm.convertTo(harrisNormScaled, CvType.CV_8U);

        // Create output image (color)
        Mat result = new Mat();
        if (input.channels() == 1) {
            Imgproc.cvtColor(input, result, Imgproc.COLOR_GRAY2BGR);
        } else {
            result = input.clone();
        }

        // Find and draw corners
        Scalar color = new Scalar(colorB, colorG, colorR);
        double threshold = thresholdPercent * 2.55; // Convert percent to 0-255 range

        for (int i = 0; i < harrisNormScaled.rows(); i++) {
            for (int j = 0; j < harrisNormScaled.cols(); j++) {
                if (harrisNormScaled.get(i, j)[0] > threshold) {
                    Imgproc.circle(result, new org.opencv.core.Point(j, i), markerSize, color, -1);
                }
            }
        }

        return result;
    }

    @Override
    public String getDescription() {
        return "Harris Corner Detection\ncv2.cornerHarris(src, blockSize, ksize, k)";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Harris Corners Properties");
        dialog.setLayout(new GridLayout(3, false));

        // Method signature
        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText(getDescription());
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sigGd.horizontalSpan = 3;
        sigLabel.setLayoutData(sigGd);

        // Block Size
        new Label(dialog, SWT.NONE).setText("Block Size:");
        Scale blockScale = new Scale(dialog, SWT.HORIZONTAL);
        blockScale.setMinimum(2);
        blockScale.setMaximum(10);
        blockScale.setSelection(blockSize);
        blockScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label blockLabel = new Label(dialog, SWT.NONE);
        blockLabel.setText(String.valueOf(blockSize));
        blockScale.addListener(SWT.Selection, e -> blockLabel.setText(String.valueOf(blockScale.getSelection())));

        // Aperture Size (ksize)
        new Label(dialog, SWT.NONE).setText("Aperture Size:");
        Combo ksizeCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        ksizeCombo.setItems(new String[]{"3", "5", "7"});
        ksizeCombo.select(ksize == 3 ? 0 : (ksize == 5 ? 1 : 2));
        GridData comboGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
        comboGd.horizontalSpan = 2;
        ksizeCombo.setLayoutData(comboGd);

        // Threshold
        new Label(dialog, SWT.NONE).setText("Threshold %:");
        Scale threshScale = new Scale(dialog, SWT.HORIZONTAL);
        threshScale.setMinimum(1);
        threshScale.setMaximum(100);
        threshScale.setSelection(thresholdPercent);
        threshScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label threshLabel = new Label(dialog, SWT.NONE);
        threshLabel.setText(String.valueOf(thresholdPercent));
        threshScale.addListener(SWT.Selection, e -> threshLabel.setText(String.valueOf(threshScale.getSelection())));

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
            blockSize = blockScale.getSelection();
            int idx = ksizeCombo.getSelectionIndex();
            ksize = idx == 0 ? 3 : (idx == 1 ? 5 : 7);
            thresholdPercent = threshScale.getSelection();
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
