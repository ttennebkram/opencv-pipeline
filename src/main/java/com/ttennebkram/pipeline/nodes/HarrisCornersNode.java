package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.registry.NodeInfo;
import org.eclipse.swt.SWT;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

/**
 * Harris Corners detection node.
 */
@NodeInfo(
    name = "HarrisCorners",
    category = "Detection",
    aliases = {"Harris Corners"}
)
public class HarrisCornersNode extends ProcessingNode {
    private boolean showOriginal = true;
    private boolean drawFeatures = true;
    private int blockSize = 2;
    private int ksize = 3;
    private int kPercent = 4; // 0.04 * 100 - Harris free parameter
    private int thresholdPercent = 1; // 0.01 * 100
    private int markerSize = 5;
    private int colorR = 255, colorG = 0, colorB = 0;

    public HarrisCornersNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Harris Corners", x, y);
    }

    // Getters/setters for serialization
    public boolean getShowOriginal() { return showOriginal; }
    public void setShowOriginal(boolean v) { showOriginal = v; }
    public boolean isDrawFeatures() { return drawFeatures; }
    public void setDrawFeatures(boolean v) { drawFeatures = v; }
    public int getBlockSize() { return blockSize; }
    public void setBlockSize(int v) { blockSize = v; }
    public int getKsize() { return ksize; }
    public void setKsize(int v) { ksize = v; }
    public int getKPercent() { return kPercent; }
    public void setKPercent(int v) { kPercent = v; }
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

        // Apply Harris corner detection with configurable k parameter
        double k = kPercent / 100.0;
        Mat harris = new Mat();
        Imgproc.cornerHarris(grayFloat, harris, blockSize, ksize, k);

        // Normalize and convert to byte for thresholding
        Mat harrisNorm = new Mat();
        Core.normalize(harris, harrisNorm, 0, 255, Core.NORM_MINMAX);
        Mat harrisNormScaled = new Mat();
        harrisNorm.convertTo(harrisNormScaled, CvType.CV_8U);

        // Create output image
        Mat result;
        if (showOriginal) {
            if (input.channels() == 1) {
                result = new Mat();
                Imgproc.cvtColor(input, result, Imgproc.COLOR_GRAY2BGR);
            } else {
                result = input.clone();
            }
        } else {
            result = Mat.zeros(input.size(), CvType.CV_8UC3);
        }

        // Find and draw corners
        if (drawFeatures) {
            Scalar color = new Scalar(colorB, colorG, colorR);
            double threshold = thresholdPercent * 2.55; // Convert percent to 0-255 range

            for (int i = 0; i < harrisNormScaled.rows(); i++) {
                for (int j = 0; j < harrisNormScaled.cols(); j++) {
                    if (harrisNormScaled.get(i, j)[0] > threshold) {
                        Imgproc.circle(result, new org.opencv.core.Point(j, i), markerSize, color, -1);
                    }
                }
            }
        }

        // Cleanup
        gray.release();
        grayFloat.release();
        harris.release();
        harrisNorm.release();
        harrisNormScaled.release();

        return result;
    }

    @Override
    public String getDescription() {
        return "Harris Corner Detection\ncv2.cornerHarris(src, blockSize, ksize, k)";
    }

    @Override
    public String getDisplayName() {
        return "Harris Corners";
    }

    @Override
    public String getCategory() {
        return "Detection";
    }

    @Override
    protected int getPropertiesDialogColumns() {
        return 3;
    }

    @Override
    protected Runnable addPropertiesContent(Shell dialog, int columns) {
        // Method signature
        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText(getDescription());
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sigGd.horizontalSpan = columns;
        sigLabel.setLayoutData(sigGd);

        // Separator
        Label sep1 = new Label(dialog, SWT.SEPARATOR | SWT.HORIZONTAL);
        GridData sepGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sepGd.horizontalSpan = columns;
        sep1.setLayoutData(sepGd);

        // Show Original checkbox
        Button showOrigBtn = new Button(dialog, SWT.CHECK);
        showOrigBtn.setText("Show Original Background");
        showOrigBtn.setSelection(showOriginal);
        GridData showGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        showGd.horizontalSpan = columns;
        showOrigBtn.setLayoutData(showGd);

        // Draw Features checkbox
        Button drawFeaturesBtn = new Button(dialog, SWT.CHECK);
        drawFeaturesBtn.setText("Draw Features");
        drawFeaturesBtn.setSelection(drawFeatures);
        GridData drawGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        drawGd.horizontalSpan = columns;
        drawFeaturesBtn.setLayoutData(drawGd);

        // Block Size
        new Label(dialog, SWT.NONE).setText("Block Size:");
        Scale blockScale = new Scale(dialog, SWT.HORIZONTAL);
        blockScale.setMinimum(2);
        blockScale.setMaximum(10);
        int blockSliderPos = Math.min(Math.max(blockSize, 2), 10);
        blockScale.setSelection(blockSliderPos);
        blockScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label blockLabel = new Label(dialog, SWT.NONE);
        blockLabel.setText(String.valueOf(blockSize));
        blockScale.addListener(SWT.Selection, e -> blockLabel.setText(String.valueOf(blockScale.getSelection())));

        // Aperture Size (ksize)
        new Label(dialog, SWT.NONE).setText("Kernel/Aperture Size:");
        Combo ksizeCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        ksizeCombo.setItems(new String[]{"3", "5", "7"});
        ksizeCombo.select(ksize == 3 ? 0 : (ksize == 5 ? 1 : 2));
        GridData comboGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
        comboGd.horizontalSpan = 2;
        ksizeCombo.setLayoutData(comboGd);

        // Harris k parameter
        new Label(dialog, SWT.NONE).setText("Harris k (%):");
        Scale kScale = new Scale(dialog, SWT.HORIZONTAL);
        kScale.setMinimum(1);
        kScale.setMaximum(10);
        int kSliderPos = Math.min(Math.max(kPercent, 1), 10);
        kScale.setSelection(kSliderPos);
        kScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label kLabel = new Label(dialog, SWT.NONE);
        kLabel.setText(String.valueOf(kPercent));
        kScale.addListener(SWT.Selection, e -> kLabel.setText(String.valueOf(kScale.getSelection())));

        // Threshold
        new Label(dialog, SWT.NONE).setText("Threshold %:");
        Scale threshScale = new Scale(dialog, SWT.HORIZONTAL);
        threshScale.setMinimum(1);
        threshScale.setMaximum(100);
        int threshSliderPos = Math.min(Math.max(thresholdPercent, 1), 100);
        threshScale.setSelection(threshSliderPos);
        threshScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label threshLabel = new Label(dialog, SWT.NONE);
        threshLabel.setText(String.valueOf(thresholdPercent));
        threshScale.addListener(SWT.Selection, e -> threshLabel.setText(String.valueOf(threshScale.getSelection())));

        // Marker Size
        new Label(dialog, SWT.NONE).setText("Marker Size:");
        Scale markerScale = new Scale(dialog, SWT.HORIZONTAL);
        markerScale.setMinimum(1);
        markerScale.setMaximum(15);
        int markerSliderPos = Math.min(Math.max(markerSize, 1), 15);
        markerScale.setSelection(markerSliderPos);
        markerScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label markerLabel = new Label(dialog, SWT.NONE);
        markerLabel.setText(String.valueOf(markerSize));
        markerScale.addListener(SWT.Selection, e -> markerLabel.setText(String.valueOf(markerScale.getSelection())));

        return () -> {
            showOriginal = showOrigBtn.getSelection();
            drawFeatures = drawFeaturesBtn.getSelection();
            blockSize = blockScale.getSelection();
            int idx = ksizeCombo.getSelectionIndex();
            ksize = idx == 0 ? 3 : (idx == 1 ? 5 : 7);
            kPercent = kScale.getSelection();
            thresholdPercent = threshScale.getSelection();
            markerSize = markerScale.getSelection();
        };
    }

    @Override
    public void serializeProperties(JsonObject json) {
        super.serializeProperties(json);
        json.addProperty("showOriginal", showOriginal);
        json.addProperty("drawFeatures", drawFeatures);
        json.addProperty("blockSize", blockSize);
        json.addProperty("ksize", ksize);
        json.addProperty("kPercent", kPercent);
        json.addProperty("thresholdPercent", thresholdPercent);
        json.addProperty("markerSize", markerSize);
        json.addProperty("colorR", colorR);
        json.addProperty("colorG", colorG);
        json.addProperty("colorB", colorB);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        super.deserializeProperties(json);
        if (json.has("showOriginal")) showOriginal = json.get("showOriginal").getAsBoolean();
        if (json.has("drawFeatures")) drawFeatures = json.get("drawFeatures").getAsBoolean();
        if (json.has("blockSize")) blockSize = json.get("blockSize").getAsInt();
        if (json.has("ksize")) ksize = json.get("ksize").getAsInt();
        if (json.has("kPercent")) kPercent = json.get("kPercent").getAsInt();
        if (json.has("thresholdPercent")) thresholdPercent = json.get("thresholdPercent").getAsInt();
        if (json.has("markerSize")) markerSize = json.get("markerSize").getAsInt();
        if (json.has("colorR")) colorR = json.get("colorR").getAsInt();
        if (json.has("colorG")) colorG = json.get("colorG").getAsInt();
        if (json.has("colorB")) colorB = json.get("colorB").getAsInt();
    }
}
