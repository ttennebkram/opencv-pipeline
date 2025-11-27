package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.registry.NodeInfo;
import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

/**
 * Hough Lines detection node.
 */
@NodeInfo(
    name = "HoughLines",
    category = "Detection",
    aliases = {"Hough Lines"}
)
public class HoughLinesNode extends ProcessingNode {
    private int threshold = 50;
    private int minLineLength = 50;
    private int maxLineGap = 10;
    private int cannyThresh1 = 50;
    private int cannyThresh2 = 150;
    private int thickness = 2;
    private int colorR = 255, colorG = 0, colorB = 0;

    public HoughLinesNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Hough Lines", x, y);
    }

    // Getters/setters for serialization
    public int getThreshold() { return threshold; }
    public void setThreshold(int v) { threshold = v; }
    public int getMinLineLength() { return minLineLength; }
    public void setMinLineLength(int v) { minLineLength = v; }
    public int getMaxLineGap() { return maxLineGap; }
    public void setMaxLineGap(int v) { maxLineGap = v; }
    public int getCannyThresh1() { return cannyThresh1; }
    public void setCannyThresh1(int v) { cannyThresh1 = v; }
    public int getCannyThresh2() { return cannyThresh2; }
    public void setCannyThresh2(int v) { cannyThresh2 = v; }
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

        Mat gray = null;
        Mat edges = null;
        Mat lines = null;
        Mat result = null;

        try {
            // Convert to grayscale for detection
            gray = new Mat();
            if (input.channels() == 3) {
                Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
            } else {
                gray = input.clone();
            }

            // Apply Canny edge detection
            edges = new Mat();
            Imgproc.Canny(gray, edges, cannyThresh1, cannyThresh2);

            // Create output image (color)
            result = new Mat();
            if (input.channels() == 1) {
                Imgproc.cvtColor(input, result, Imgproc.COLOR_GRAY2BGR);
            } else {
                result = input.clone();
            }

            // Detect lines using probabilistic Hough transform
            lines = new Mat();
            Imgproc.HoughLinesP(edges, lines, 1, Math.PI / 180, threshold, minLineLength, maxLineGap);

            // Draw lines
            Scalar color = new Scalar(colorB, colorG, colorR);
            for (int i = 0; i < lines.rows(); i++) {
                double[] l = lines.get(i, 0);
                Imgproc.line(result,
                    new org.opencv.core.Point(l[0], l[1]),
                    new org.opencv.core.Point(l[2], l[3]),
                    color, thickness);
            }

            return result;
        } finally {
            // Release intermediate Mats (but not result which is returned)
            if (gray != null) gray.release();
            if (edges != null) edges.release();
            if (lines != null) lines.release();
        }
    }

    @Override
    public String getDescription() {
        return "Hough Line Detection\ncv2.HoughLinesP(image, rho, theta, threshold)";
    }

    @Override
    public String getDisplayName() {
        return "Hough Lines";
    }

    @Override
    public String getCategory() {
        return "Detection";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Hough Lines Properties");
        dialog.setLayout(new GridLayout(3, false));

        // Method signature
        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText(getDescription());
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sigGd.horizontalSpan = 3;
        sigLabel.setLayoutData(sigGd);

        // Canny Threshold 1
        new Label(dialog, SWT.NONE).setText("Canny Thresh 1:");
        Scale c1Scale = new Scale(dialog, SWT.HORIZONTAL);
        c1Scale.setMinimum(0);
        c1Scale.setMaximum(255);
        // Clamp slider position to valid range, but keep actual value
        int c1SliderPos = Math.min(Math.max(cannyThresh1, 0), 255);
        c1Scale.setSelection(c1SliderPos);
        c1Scale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label c1Label = new Label(dialog, SWT.NONE);
        c1Label.setText(String.valueOf(cannyThresh1)); // Show real value
        c1Scale.addListener(SWT.Selection, e -> c1Label.setText(String.valueOf(c1Scale.getSelection())));

        // Canny Threshold 2
        new Label(dialog, SWT.NONE).setText("Canny Thresh 2:");
        Scale c2Scale = new Scale(dialog, SWT.HORIZONTAL);
        c2Scale.setMinimum(0);
        c2Scale.setMaximum(255);
        // Clamp slider position to valid range, but keep actual value
        int c2SliderPos = Math.min(Math.max(cannyThresh2, 0), 255);
        c2Scale.setSelection(c2SliderPos);
        c2Scale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label c2Label = new Label(dialog, SWT.NONE);
        c2Label.setText(String.valueOf(cannyThresh2)); // Show real value
        c2Scale.addListener(SWT.Selection, e -> c2Label.setText(String.valueOf(c2Scale.getSelection())));

        // Threshold
        new Label(dialog, SWT.NONE).setText("Hough Threshold:");
        Scale threshScale = new Scale(dialog, SWT.HORIZONTAL);
        threshScale.setMinimum(1);
        threshScale.setMaximum(200);
        // Clamp slider position to valid range, but keep actual value
        int threshSliderPos = Math.min(Math.max(threshold, 1), 200);
        threshScale.setSelection(threshSliderPos);
        threshScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label threshLabel = new Label(dialog, SWT.NONE);
        threshLabel.setText(String.valueOf(threshold)); // Show real value
        threshScale.addListener(SWT.Selection, e -> threshLabel.setText(String.valueOf(threshScale.getSelection())));

        // Min Line Length
        new Label(dialog, SWT.NONE).setText("Min Line Length:");
        Scale minLenScale = new Scale(dialog, SWT.HORIZONTAL);
        minLenScale.setMinimum(1);
        minLenScale.setMaximum(200);
        // Clamp slider position to valid range, but keep actual value
        int minLenSliderPos = Math.min(Math.max(minLineLength, 1), 200);
        minLenScale.setSelection(minLenSliderPos);
        minLenScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label minLenLabel = new Label(dialog, SWT.NONE);
        minLenLabel.setText(String.valueOf(minLineLength)); // Show real value
        minLenScale.addListener(SWT.Selection, e -> minLenLabel.setText(String.valueOf(minLenScale.getSelection())));

        // Max Line Gap
        new Label(dialog, SWT.NONE).setText("Max Line Gap:");
        Scale gapScale = new Scale(dialog, SWT.HORIZONTAL);
        gapScale.setMinimum(1);
        gapScale.setMaximum(100);
        // Clamp slider position to valid range, but keep actual value
        int gapSliderPos = Math.min(Math.max(maxLineGap, 1), 100);
        gapScale.setSelection(gapSliderPos);
        gapScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label gapLabel = new Label(dialog, SWT.NONE);
        gapLabel.setText(String.valueOf(maxLineGap)); // Show real value
        gapScale.addListener(SWT.Selection, e -> gapLabel.setText(String.valueOf(gapScale.getSelection())));

        // Thickness
        new Label(dialog, SWT.NONE).setText("Line Thickness:");
        Scale thickScale = new Scale(dialog, SWT.HORIZONTAL);
        thickScale.setMinimum(1);
        thickScale.setMaximum(10);
        // Clamp slider position to valid range, but keep actual value
        int thickSliderPos = Math.min(Math.max(thickness, 1), 10);
        thickScale.setSelection(thickSliderPos);
        thickScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label thickLabel = new Label(dialog, SWT.NONE);
        thickLabel.setText(String.valueOf(thickness)); // Show real value
        thickScale.addListener(SWT.Selection, e -> thickLabel.setText(String.valueOf(thickScale.getSelection())));

        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(2, true));
        GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
        gd.horizontalSpan = 3;
        buttonComp.setLayoutData(gd);

        Button okBtn = new Button(buttonComp, SWT.PUSH);
        okBtn.setText("OK");
        dialog.setDefaultButton(okBtn);
        okBtn.addListener(SWT.Selection, e -> {
            cannyThresh1 = c1Scale.getSelection();
            cannyThresh2 = c2Scale.getSelection();
            threshold = threshScale.getSelection();
            minLineLength = minLenScale.getSelection();
            maxLineGap = gapScale.getSelection();
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

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("threshold", threshold);
        json.addProperty("minLineLength", minLineLength);
        json.addProperty("maxLineGap", maxLineGap);
        json.addProperty("cannyThresh1", cannyThresh1);
        json.addProperty("cannyThresh2", cannyThresh2);
        json.addProperty("thickness", thickness);
        json.addProperty("colorR", colorR);
        json.addProperty("colorG", colorG);
        json.addProperty("colorB", colorB);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        if (json.has("threshold")) threshold = json.get("threshold").getAsInt();
        if (json.has("minLineLength")) minLineLength = json.get("minLineLength").getAsInt();
        if (json.has("maxLineGap")) maxLineGap = json.get("maxLineGap").getAsInt();
        if (json.has("cannyThresh1")) cannyThresh1 = json.get("cannyThresh1").getAsInt();
        if (json.has("cannyThresh2")) cannyThresh2 = json.get("cannyThresh2").getAsInt();
        if (json.has("thickness")) thickness = json.get("thickness").getAsInt();
        if (json.has("colorR")) colorR = json.get("colorR").getAsInt();
        if (json.has("colorG")) colorG = json.get("colorG").getAsInt();
        if (json.has("colorB")) colorB = json.get("colorB").getAsInt();
    }
}
