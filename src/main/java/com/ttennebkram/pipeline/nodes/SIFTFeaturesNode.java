package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.registry.NodeInfo;
import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.SIFT;
import org.opencv.imgproc.Imgproc;

/**
 * SIFT (Scale-Invariant Feature Transform) feature detection node.
 */
@NodeInfo(name = "SIFTFeatures", category = "Detection", aliases = {"SIFT Features"})
public class SIFTFeaturesNode extends ProcessingNode {
    private int nFeatures = 500;
    private int nOctaveLayers = 3;
    private double contrastThreshold = 0.04;
    private double edgeThreshold = 10.0;
    private double sigma = 1.6;
    private boolean showRichKeypoints = true;
    private int colorIndex = 0; // 0=green, 1=red, 2=blue, 3=yellow, 4=white

    private static final String[] COLOR_NAMES = {"Green", "Red", "Blue", "Yellow", "White"};
    private static final Scalar[] COLORS = {
        new Scalar(0, 255, 0),    // Green
        new Scalar(0, 0, 255),    // Red
        new Scalar(255, 0, 0),    // Blue
        new Scalar(0, 255, 255),  // Yellow
        new Scalar(255, 255, 255) // White
    };

    public SIFTFeaturesNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "SIFT Features", x, y);
    }

    // Getters/setters for serialization
    public int getNFeatures() { return nFeatures; }
    public void setNFeatures(int v) { nFeatures = v; }
    public int getNOctaveLayers() { return nOctaveLayers; }
    public void setNOctaveLayers(int v) { nOctaveLayers = v; }
    public double getContrastThreshold() { return contrastThreshold; }
    public void setContrastThreshold(double v) { contrastThreshold = v; }
    public double getEdgeThreshold() { return edgeThreshold; }
    public void setEdgeThreshold(double v) { edgeThreshold = v; }
    public double getSigma() { return sigma; }
    public void setSigma(double v) { sigma = v; }
    public boolean isShowRichKeypoints() { return showRichKeypoints; }
    public void setShowRichKeypoints(boolean v) { showRichKeypoints = v; }
    public int getColorIndex() { return colorIndex; }
    public void setColorIndex(int v) { colorIndex = v; }

    @Override
    public Mat process(Mat input) {
        if (!enabled || input == null || input.empty()) {
            return input;
        }

        Mat gray = null;
        MatOfKeyPoint keypoints = null;
        Mat output = null;

        try {
            // Convert to grayscale for detection
            gray = new Mat();
            if (input.channels() == 3) {
                Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
            } else {
                gray = input.clone();
            }

            // Create SIFT detector
            SIFT sift = SIFT.create(nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);

            // Detect keypoints
            keypoints = new MatOfKeyPoint();
            sift.detect(gray, keypoints);

            // Draw keypoints
            output = new Mat();
            int flags = showRichKeypoints ? Features2d.DrawMatchesFlags_DRAW_RICH_KEYPOINTS : Features2d.DrawMatchesFlags_DEFAULT;
            Scalar color = COLORS[colorIndex < COLORS.length ? colorIndex : 0];
            Features2d.drawKeypoints(input, keypoints, output, color, flags);

            return output;
        } finally {
            // Release intermediate Mats (but not output which is returned)
            if (gray != null) gray.release();
            if (keypoints != null) keypoints.release();
        }
    }

    @Override
    public String getDescription() {
        return "SIFT: Scale-Invariant Feature Transform\ncv2.SIFT_create(nfeatures, ...)";
    }

    @Override
    public String getDisplayName() {
        return "SIFT Features";
    }

    @Override
    public String getCategory() {
        return "Detection";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("SIFT Features Properties");
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

        // Max Features
        new Label(dialog, SWT.NONE).setText("Max Features:");
        Scale featuresScale = new Scale(dialog, SWT.HORIZONTAL);
        featuresScale.setMinimum(10);
        featuresScale.setMaximum(2000);
        featuresScale.setSelection(nFeatures);
        featuresScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label featuresLabel = new Label(dialog, SWT.NONE);
        featuresLabel.setText(String.valueOf(nFeatures));
        featuresScale.addListener(SWT.Selection, e -> featuresLabel.setText(String.valueOf(featuresScale.getSelection())));

        // Contrast Threshold (0.01-0.20, scaled by 1000)
        new Label(dialog, SWT.NONE).setText("Contrast Threshold:");
        Scale contrastScale = new Scale(dialog, SWT.HORIZONTAL);
        contrastScale.setMinimum(10);
        contrastScale.setMaximum(200);
        contrastScale.setSelection((int)(contrastThreshold * 1000));
        contrastScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label contrastLabel = new Label(dialog, SWT.NONE);
        contrastLabel.setText(String.format("%.3f", contrastThreshold));
        contrastScale.addListener(SWT.Selection, e -> {
            double val = contrastScale.getSelection() / 1000.0;
            contrastLabel.setText(String.format("%.3f", val));
        });

        // Edge Threshold (1-50)
        new Label(dialog, SWT.NONE).setText("Edge Threshold:");
        Scale edgeScale = new Scale(dialog, SWT.HORIZONTAL);
        edgeScale.setMinimum(10);
        edgeScale.setMaximum(500);
        edgeScale.setSelection((int)(edgeThreshold * 10));
        edgeScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label edgeLabel = new Label(dialog, SWT.NONE);
        edgeLabel.setText(String.format("%.1f", edgeThreshold));
        edgeScale.addListener(SWT.Selection, e -> {
            double val = edgeScale.getSelection() / 10.0;
            edgeLabel.setText(String.format("%.1f", val));
        });

        // Show Rich Keypoints
        new Label(dialog, SWT.NONE).setText("Show Size & Orientation:");
        Button richCheck = new Button(dialog, SWT.CHECK);
        richCheck.setSelection(showRichKeypoints);
        GridData richGd = new GridData();
        richGd.horizontalSpan = 2;
        richCheck.setLayoutData(richGd);

        // Keypoint Color
        new Label(dialog, SWT.NONE).setText("Keypoint Color:");
        Combo colorCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        colorCombo.setItems(COLOR_NAMES);
        colorCombo.select(colorIndex < COLOR_NAMES.length ? colorIndex : 0);
        GridData colorGd = new GridData();
        colorGd.horizontalSpan = 2;
        colorCombo.setLayoutData(colorGd);

        // Buttons
        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(2, true));
        GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
        gd.horizontalSpan = 3;
        buttonComp.setLayoutData(gd);

        Button okBtn = new Button(buttonComp, SWT.PUSH);
        okBtn.setText("OK");
        dialog.setDefaultButton(okBtn);
        okBtn.addListener(SWT.Selection, e -> {
            nFeatures = featuresScale.getSelection();
            contrastThreshold = contrastScale.getSelection() / 1000.0;
            edgeThreshold = edgeScale.getSelection() / 10.0;
            showRichKeypoints = richCheck.getSelection();
            colorIndex = colorCombo.getSelectionIndex();
            dialog.dispose();
            notifyChanged();
        });

        Button cancelBtn = new Button(buttonComp, SWT.PUSH);
        cancelBtn.setText("Cancel");
        cancelBtn.addListener(SWT.Selection, e -> dialog.dispose());

        dialog.pack();
        // Position dialog near cursor
        Point cursor = shell.getDisplay().getCursorLocation();
        dialog.setLocation(cursor.x, cursor.y);
        dialog.open();
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("nFeatures", nFeatures);
        json.addProperty("nOctaveLayers", nOctaveLayers);
        json.addProperty("contrastThreshold", contrastThreshold);
        json.addProperty("edgeThreshold", edgeThreshold);
        json.addProperty("sigma", sigma);
        json.addProperty("showRichKeypoints", showRichKeypoints);
        json.addProperty("colorIndex", colorIndex);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        if (json.has("nFeatures")) nFeatures = json.get("nFeatures").getAsInt();
        if (json.has("nOctaveLayers")) nOctaveLayers = json.get("nOctaveLayers").getAsInt();
        if (json.has("contrastThreshold")) contrastThreshold = json.get("contrastThreshold").getAsDouble();
        if (json.has("edgeThreshold")) edgeThreshold = json.get("edgeThreshold").getAsDouble();
        if (json.has("sigma")) sigma = json.get("sigma").getAsDouble();
        if (json.has("showRichKeypoints")) showRichKeypoints = json.get("showRichKeypoints").getAsBoolean();
        if (json.has("colorIndex")) colorIndex = json.get("colorIndex").getAsInt();
    }
}
