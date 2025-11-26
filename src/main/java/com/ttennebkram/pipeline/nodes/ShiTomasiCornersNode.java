package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.registry.NodeInfo;
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
@NodeInfo(
    name = "ShiTomasi",
    category = "Detection",
    aliases = {"Shi-Tomasi", "Shi-Tomasi Corners"}
)
public class ShiTomasiCornersNode extends ProcessingNode {
    private int maxCorners = 100;
    private int qualityLevel = 1; // 0.01 * 100
    private int minDistance = 10;
    private int blockSize = 3;
    private boolean useHarrisDetector = false;
    private int kPercent = 4; // 0.04 * 100
    private int markerSize = 5;
    private int colorR = 0, colorG = 255, colorB = 0;
    private boolean drawFeatures = true;

    public ShiTomasiCornersNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Shi-Tomasi Corners", x, y);
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
    public boolean isUseHarrisDetector() { return useHarrisDetector; }
    public void setUseHarrisDetector(boolean v) { useHarrisDetector = v; }
    public int getKPercent() { return kPercent; }
    public void setKPercent(int v) { kPercent = v; }
    public int getMarkerSize() { return markerSize; }
    public void setMarkerSize(int v) { markerSize = v; }
    public int getColorR() { return colorR; }
    public void setColorR(int v) { colorR = v; }
    public int getColorG() { return colorG; }
    public void setColorG(int v) { colorG = v; }
    public int getColorB() { return colorB; }
    public void setColorB(int v) { colorB = v; }
    public boolean isDrawFeatures() { return drawFeatures; }
    public void setDrawFeatures(boolean v) { drawFeatures = v; }

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
        double k = kPercent / 100.0;
        Imgproc.goodFeaturesToTrack(gray, corners, maxCorners, quality, minDistance, new Mat(), blockSize, useHarrisDetector, k);

        // Draw corners
        if (drawFeatures) {
            Scalar color = new Scalar(colorB, colorG, colorR);
            org.opencv.core.Point[] cornerArray = corners.toArray();
            for (org.opencv.core.Point corner : cornerArray) {
                Imgproc.circle(result, corner, markerSize, color, -1);
            }
        }

        return result;
    }

    @Override
    public String getDescription() {
        return "Shi-Tomasi Corner Detection\ncv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance)";
    }

    @Override
    public String getDisplayName() {
        return "Shi-Tomasi Corners";
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

        // Draw Features checkbox
        Button drawFeaturesBtn = new Button(dialog, SWT.CHECK);
        drawFeaturesBtn.setText("Draw Features");
        drawFeaturesBtn.setSelection(drawFeatures);
        GridData drawGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        drawGd.horizontalSpan = 3;
        drawFeaturesBtn.setLayoutData(drawGd);

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

        // Use Harris Detector checkbox
        new Label(dialog, SWT.NONE).setText("Use Harris:");
        Button harrisCheck = new Button(dialog, SWT.CHECK);
        harrisCheck.setSelection(useHarrisDetector);
        GridData harrisGd = new GridData(SWT.LEFT, SWT.CENTER, false, false);
        harrisGd.horizontalSpan = 2;
        harrisCheck.setLayoutData(harrisGd);

        // k parameter (Harris free parameter)
        new Label(dialog, SWT.NONE).setText("k (%):");
        Scale kScale = new Scale(dialog, SWT.HORIZONTAL);
        kScale.setMinimum(1);
        kScale.setMaximum(10);
        kScale.setSelection(kPercent);
        kScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label kLabel = new Label(dialog, SWT.NONE);
        kLabel.setText(String.valueOf(kPercent));
        kScale.addListener(SWT.Selection, e -> kLabel.setText(String.valueOf(kScale.getSelection())));

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
        dialog.setDefaultButton(okBtn);
        okBtn.addListener(SWT.Selection, e -> {
            drawFeatures = drawFeaturesBtn.getSelection();
            maxCorners = maxScale.getSelection();
            qualityLevel = qualScale.getSelection();
            minDistance = distScale.getSelection();
            blockSize = blockScale.getSelection();
            useHarrisDetector = harrisCheck.getSelection();
            kPercent = kScale.getSelection();
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

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("maxCorners", maxCorners);
        json.addProperty("qualityLevel", qualityLevel);
        json.addProperty("minDistance", minDistance);
        json.addProperty("blockSize", blockSize);
        json.addProperty("useHarrisDetector", useHarrisDetector);
        json.addProperty("kPercent", kPercent);
        json.addProperty("markerSize", markerSize);
        json.addProperty("colorR", colorR);
        json.addProperty("colorG", colorG);
        json.addProperty("colorB", colorB);
        json.addProperty("drawFeatures", drawFeatures);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        if (json.has("maxCorners")) maxCorners = json.get("maxCorners").getAsInt();
        if (json.has("qualityLevel")) qualityLevel = json.get("qualityLevel").getAsInt();
        if (json.has("minDistance")) minDistance = json.get("minDistance").getAsInt();
        if (json.has("blockSize")) blockSize = json.get("blockSize").getAsInt();
        if (json.has("useHarrisDetector")) useHarrisDetector = json.get("useHarrisDetector").getAsBoolean();
        if (json.has("kPercent")) kPercent = json.get("kPercent").getAsInt();
        if (json.has("markerSize")) markerSize = json.get("markerSize").getAsInt();
        if (json.has("colorR")) colorR = json.get("colorR").getAsInt();
        if (json.has("colorG")) colorG = json.get("colorG").getAsInt();
        if (json.has("colorB")) colorB = json.get("colorB").getAsInt();
        if (json.has("drawFeatures")) drawFeatures = json.get("drawFeatures").getAsBoolean();
    }
}
