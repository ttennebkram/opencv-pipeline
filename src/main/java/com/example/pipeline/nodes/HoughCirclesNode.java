package com.example.pipeline.nodes;

import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Hough Circles detection node.
 */
public class HoughCirclesNode extends ProcessingNode {
    private int minDist = 50;
    private int param1 = 100;  // Canny high threshold
    private int param2 = 30;   // Accumulator threshold
    private int minRadius = 10;
    private int maxRadius = 100;
    private int thickness = 2;
    private boolean drawCenter = true;
    private int colorR = 0, colorG = 255, colorB = 0;

    public HoughCirclesNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Hough Circles", x, y);
    }

    // Getters/setters for serialization
    public int getMinDist() { return minDist; }
    public void setMinDist(int v) { minDist = v; }
    public int getParam1() { return param1; }
    public void setParam1(int v) { param1 = v; }
    public int getParam2() { return param2; }
    public void setParam2(int v) { param2 = v; }
    public int getMinRadius() { return minRadius; }
    public void setMinRadius(int v) { minRadius = v; }
    public int getMaxRadius() { return maxRadius; }
    public void setMaxRadius(int v) { maxRadius = v; }
    public int getThickness() { return thickness; }
    public void setThickness(int v) { thickness = v; }
    public boolean isDrawCenter() { return drawCenter; }
    public void setDrawCenter(boolean v) { drawCenter = v; }
    public int getColorR() { return colorR; }
    public void setColorR(int v) { colorR = v; }
    public int getColorG() { return colorG; }
    public void setColorG(int v) { colorG = v; }
    public int getColorB() { return colorB; }
    public void setColorB(int v) { colorB = v; }

    @Override
    public Mat process(Mat input) {
        if (!enabled || input == null || input.empty()) return input;

        // Convert to grayscale for detection
        Mat gray = new Mat();
        if (input.channels() == 3) {
            Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
        } else {
            gray = input.clone();
        }

        // Apply Gaussian blur to reduce noise
        Imgproc.GaussianBlur(gray, gray, new Size(9, 9), 2);

        // Create output image (color)
        Mat result = new Mat();
        if (input.channels() == 1) {
            Imgproc.cvtColor(input, result, Imgproc.COLOR_GRAY2BGR);
        } else {
            result = input.clone();
        }

        // Detect circles
        Mat circles = new Mat();
        Imgproc.HoughCircles(gray, circles, Imgproc.HOUGH_GRADIENT, 1, minDist,
            param1, param2, minRadius, maxRadius);

        // Draw circles
        Scalar color = new Scalar(colorB, colorG, colorR);
        for (int i = 0; i < circles.cols(); i++) {
            double[] c = circles.get(0, i);
            org.opencv.core.Point center = new org.opencv.core.Point(c[0], c[1]);
            int radius = (int) Math.round(c[2]);

            // Draw circle outline
            Imgproc.circle(result, center, radius, color, thickness);

            // Draw center point
            if (drawCenter) {
                Imgproc.circle(result, center, 2, color, 3);
            }
        }

        return result;
    }

    @Override
    public String getDescription() {
        return "Hough Circle Detection\ncv2.HoughCircles(image, method, dp, minDist)";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Hough Circles Properties");
        dialog.setLayout(new GridLayout(3, false));

        // Method signature
        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText(getDescription());
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sigGd.horizontalSpan = 3;
        sigLabel.setLayoutData(sigGd);

        // Min Distance
        new Label(dialog, SWT.NONE).setText("Min Distance:");
        Scale distScale = new Scale(dialog, SWT.HORIZONTAL);
        distScale.setMinimum(1);
        distScale.setMaximum(200);
        distScale.setSelection(minDist);
        distScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label distLabel = new Label(dialog, SWT.NONE);
        distLabel.setText(String.valueOf(minDist));
        distScale.addListener(SWT.Selection, e -> distLabel.setText(String.valueOf(distScale.getSelection())));

        // Param1 (Canny threshold)
        new Label(dialog, SWT.NONE).setText("Canny Threshold:");
        Scale p1Scale = new Scale(dialog, SWT.HORIZONTAL);
        p1Scale.setMinimum(1);
        p1Scale.setMaximum(300);
        p1Scale.setSelection(param1);
        p1Scale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label p1Label = new Label(dialog, SWT.NONE);
        p1Label.setText(String.valueOf(param1));
        p1Scale.addListener(SWT.Selection, e -> p1Label.setText(String.valueOf(p1Scale.getSelection())));

        // Param2 (Accumulator threshold)
        new Label(dialog, SWT.NONE).setText("Accum Threshold:");
        Scale p2Scale = new Scale(dialog, SWT.HORIZONTAL);
        p2Scale.setMinimum(1);
        p2Scale.setMaximum(100);
        p2Scale.setSelection(param2);
        p2Scale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label p2Label = new Label(dialog, SWT.NONE);
        p2Label.setText(String.valueOf(param2));
        p2Scale.addListener(SWT.Selection, e -> p2Label.setText(String.valueOf(p2Scale.getSelection())));

        // Min Radius
        new Label(dialog, SWT.NONE).setText("Min Radius:");
        Scale minRScale = new Scale(dialog, SWT.HORIZONTAL);
        minRScale.setMinimum(0);
        minRScale.setMaximum(200);
        minRScale.setSelection(minRadius);
        minRScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label minRLabel = new Label(dialog, SWT.NONE);
        minRLabel.setText(String.valueOf(minRadius));
        minRScale.addListener(SWT.Selection, e -> minRLabel.setText(String.valueOf(minRScale.getSelection())));

        // Max Radius
        new Label(dialog, SWT.NONE).setText("Max Radius:");
        Scale maxRScale = new Scale(dialog, SWT.HORIZONTAL);
        maxRScale.setMinimum(0);
        maxRScale.setMaximum(500);
        maxRScale.setSelection(maxRadius);
        maxRScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label maxRLabel = new Label(dialog, SWT.NONE);
        maxRLabel.setText(String.valueOf(maxRadius));
        maxRScale.addListener(SWT.Selection, e -> maxRLabel.setText(String.valueOf(maxRScale.getSelection())));

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

        // Draw center checkbox
        Button centerCheck = new Button(dialog, SWT.CHECK);
        centerCheck.setText("Draw Center Point");
        centerCheck.setSelection(drawCenter);
        GridData checkGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
        checkGd.horizontalSpan = 3;
        centerCheck.setLayoutData(checkGd);

        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(2, true));
        GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
        gd.horizontalSpan = 3;
        buttonComp.setLayoutData(gd);

        Button okBtn = new Button(buttonComp, SWT.PUSH);
        okBtn.setText("OK");
        okBtn.addListener(SWT.Selection, e -> {
            minDist = distScale.getSelection();
            param1 = p1Scale.getSelection();
            param2 = p2Scale.getSelection();
            minRadius = minRScale.getSelection();
            maxRadius = maxRScale.getSelection();
            thickness = thickScale.getSelection();
            drawCenter = centerCheck.getSelection();
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
