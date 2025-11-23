package com.example.pipeline.nodes;

import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.SimpleBlobDetector;
import org.opencv.features2d.SimpleBlobDetector_Params;
import org.opencv.imgproc.Imgproc;

/**
 * Blob Detection node.
 */
public class BlobDetectorNode extends ProcessingNode {
    private int minThreshold = 10;
    private int maxThreshold = 200;
    private boolean filterByArea = true;
    private int minArea = 100;
    private int maxArea = 5000;
    private boolean filterByCircularity = false;
    private int minCircularity = 10; // 0.1 * 100
    private int colorR = 255, colorG = 0, colorB = 0;

    public BlobDetectorNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Blob Detector", x, y);
    }

    // Getters/setters for serialization
    public int getMinThreshold() { return minThreshold; }
    public void setMinThreshold(int v) { minThreshold = v; }
    public int getMaxThreshold() { return maxThreshold; }
    public void setMaxThreshold(int v) { maxThreshold = v; }
    public boolean isFilterByArea() { return filterByArea; }
    public void setFilterByArea(boolean v) { filterByArea = v; }
    public int getMinArea() { return minArea; }
    public void setMinArea(int v) { minArea = v; }
    public int getMaxArea() { return maxArea; }
    public void setMaxArea(int v) { maxArea = v; }
    public boolean isFilterByCircularity() { return filterByCircularity; }
    public void setFilterByCircularity(boolean v) { filterByCircularity = v; }
    public int getMinCircularity() { return minCircularity; }
    public void setMinCircularity(int v) { minCircularity = v; }
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

        // Set up SimpleBlobDetector parameters
        SimpleBlobDetector_Params params = new SimpleBlobDetector_Params();
        params.set_minThreshold(minThreshold);
        params.set_maxThreshold(maxThreshold);
        params.set_filterByArea(filterByArea);
        params.set_minArea(minArea);
        params.set_maxArea(maxArea);
        params.set_filterByCircularity(filterByCircularity);
        params.set_minCircularity((float)(minCircularity / 100.0));

        // Create detector and detect blobs
        SimpleBlobDetector detector = SimpleBlobDetector.create(params);
        MatOfKeyPoint keypoints = new MatOfKeyPoint();
        detector.detect(gray, keypoints);

        // Draw keypoints
        Scalar color = new Scalar(colorB, colorG, colorR);
        Features2d.drawKeypoints(result, keypoints, result, color,
            Features2d.DrawMatchesFlags_DRAW_RICH_KEYPOINTS);

        return result;
    }

    @Override
    public String getDescription() {
        return "Blob Detection\ncv2.SimpleBlobDetector_create(params)";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Blob Detector Properties");
        dialog.setLayout(new GridLayout(3, false));

        // Method signature
        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText(getDescription());
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sigGd.horizontalSpan = 3;
        sigLabel.setLayoutData(sigGd);

        // Min Threshold
        new Label(dialog, SWT.NONE).setText("Min Threshold:");
        Scale minThreshScale = new Scale(dialog, SWT.HORIZONTAL);
        minThreshScale.setMinimum(0);
        minThreshScale.setMaximum(255);
        minThreshScale.setSelection(minThreshold);
        minThreshScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label minThreshLabel = new Label(dialog, SWT.NONE);
        minThreshLabel.setText(String.valueOf(minThreshold));
        minThreshScale.addListener(SWT.Selection, e -> minThreshLabel.setText(String.valueOf(minThreshScale.getSelection())));

        // Max Threshold
        new Label(dialog, SWT.NONE).setText("Max Threshold:");
        Scale maxThreshScale = new Scale(dialog, SWT.HORIZONTAL);
        maxThreshScale.setMinimum(0);
        maxThreshScale.setMaximum(255);
        maxThreshScale.setSelection(maxThreshold);
        maxThreshScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label maxThreshLabel = new Label(dialog, SWT.NONE);
        maxThreshLabel.setText(String.valueOf(maxThreshold));
        maxThreshScale.addListener(SWT.Selection, e -> maxThreshLabel.setText(String.valueOf(maxThreshScale.getSelection())));

        // Filter by Area checkbox
        Button areaCheck = new Button(dialog, SWT.CHECK);
        areaCheck.setText("Filter by Area");
        areaCheck.setSelection(filterByArea);
        GridData areaGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
        areaGd.horizontalSpan = 3;
        areaCheck.setLayoutData(areaGd);

        // Min Area
        new Label(dialog, SWT.NONE).setText("Min Area:");
        Scale minAreaScale = new Scale(dialog, SWT.HORIZONTAL);
        minAreaScale.setMinimum(1);
        minAreaScale.setMaximum(10000);
        minAreaScale.setSelection(minArea);
        minAreaScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label minAreaLabel = new Label(dialog, SWT.NONE);
        minAreaLabel.setText(String.valueOf(minArea));
        minAreaScale.addListener(SWT.Selection, e -> minAreaLabel.setText(String.valueOf(minAreaScale.getSelection())));

        // Max Area
        new Label(dialog, SWT.NONE).setText("Max Area:");
        Scale maxAreaScale = new Scale(dialog, SWT.HORIZONTAL);
        maxAreaScale.setMinimum(1);
        maxAreaScale.setMaximum(50000);
        maxAreaScale.setSelection(maxArea);
        maxAreaScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label maxAreaLabel = new Label(dialog, SWT.NONE);
        maxAreaLabel.setText(String.valueOf(maxArea));
        maxAreaScale.addListener(SWT.Selection, e -> maxAreaLabel.setText(String.valueOf(maxAreaScale.getSelection())));

        // Filter by Circularity checkbox
        Button circCheck = new Button(dialog, SWT.CHECK);
        circCheck.setText("Filter by Circularity");
        circCheck.setSelection(filterByCircularity);
        GridData circGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
        circGd.horizontalSpan = 3;
        circCheck.setLayoutData(circGd);

        // Min Circularity
        new Label(dialog, SWT.NONE).setText("Min Circularity %:");
        Scale circScale = new Scale(dialog, SWT.HORIZONTAL);
        circScale.setMinimum(1);
        circScale.setMaximum(100);
        circScale.setSelection(minCircularity);
        circScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label circLabel = new Label(dialog, SWT.NONE);
        circLabel.setText(String.valueOf(minCircularity));
        circScale.addListener(SWT.Selection, e -> circLabel.setText(String.valueOf(circScale.getSelection())));

        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(2, true));
        GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
        gd.horizontalSpan = 3;
        buttonComp.setLayoutData(gd);

        Button okBtn = new Button(buttonComp, SWT.PUSH);
        okBtn.setText("OK");
        okBtn.addListener(SWT.Selection, e -> {
            minThreshold = minThreshScale.getSelection();
            maxThreshold = maxThreshScale.getSelection();
            filterByArea = areaCheck.getSelection();
            minArea = minAreaScale.getSelection();
            maxArea = maxAreaScale.getSelection();
            filterByCircularity = circCheck.getSelection();
            minCircularity = circScale.getSelection();
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
