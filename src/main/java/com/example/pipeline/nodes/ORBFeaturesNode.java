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
import org.opencv.features2d.ORB;
import org.opencv.imgproc.Imgproc;

/**
 * ORB Features detection node.
 */
public class ORBFeaturesNode extends ProcessingNode {
    private int nFeatures = 500;
    private int fastThreshold = 20;
    private int nLevels = 8;
    private boolean showRich = true;
    private int colorR = 0, colorG = 255, colorB = 0;

    public ORBFeaturesNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "ORB Features", x, y);
    }

    // Getters/setters for serialization
    public int getNFeatures() { return nFeatures; }
    public void setNFeatures(int v) { nFeatures = v; }
    public int getFastThreshold() { return fastThreshold; }
    public void setFastThreshold(int v) { fastThreshold = v; }
    public int getNLevels() { return nLevels; }
    public void setNLevels(int v) { nLevels = v; }
    public boolean isShowRich() { return showRich; }
    public void setShowRich(boolean v) { showRich = v; }
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

        // Create output image (color)
        Mat result = new Mat();
        if (input.channels() == 1) {
            Imgproc.cvtColor(input, result, Imgproc.COLOR_GRAY2BGR);
        } else {
            result = input.clone();
        }

        // Create ORB detector
        ORB orb = ORB.create(
            nFeatures, 1.2f, nLevels, 31, 0, 2, ORB.HARRIS_SCORE, 31, fastThreshold);

        // Detect keypoints
        MatOfKeyPoint keypoints = new MatOfKeyPoint();
        orb.detect(gray, keypoints);

        // Draw keypoints
        Scalar color = new Scalar(colorB, colorG, colorR);
        int flags = showRich ? Features2d.DrawMatchesFlags_DRAW_RICH_KEYPOINTS : 0;
        Features2d.drawKeypoints(result, keypoints, result, color, flags);

        return result;
    }

    @Override
    public String getDescription() {
        return "ORB: Oriented FAST and Rotated BRIEF\ncv2.ORB_create(nfeatures, scaleFactor, nlevels)";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("ORB Features Properties");
        dialog.setLayout(new GridLayout(3, false));

        // Method signature
        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText(getDescription());
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sigGd.horizontalSpan = 3;
        sigLabel.setLayoutData(sigGd);

        // Max Features
        new Label(dialog, SWT.NONE).setText("Max Features:");
        Scale featScale = new Scale(dialog, SWT.HORIZONTAL);
        featScale.setMinimum(10);
        featScale.setMaximum(5000);
        featScale.setSelection(nFeatures);
        featScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label featLabel = new Label(dialog, SWT.NONE);
        featLabel.setText(String.valueOf(nFeatures));
        featScale.addListener(SWT.Selection, e -> featLabel.setText(String.valueOf(featScale.getSelection())));

        // FAST Threshold
        new Label(dialog, SWT.NONE).setText("FAST Threshold:");
        Scale fastScale = new Scale(dialog, SWT.HORIZONTAL);
        fastScale.setMinimum(1);
        fastScale.setMaximum(100);
        fastScale.setSelection(fastThreshold);
        fastScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label fastLabel = new Label(dialog, SWT.NONE);
        fastLabel.setText(String.valueOf(fastThreshold));
        fastScale.addListener(SWT.Selection, e -> fastLabel.setText(String.valueOf(fastScale.getSelection())));

        // Pyramid Levels
        new Label(dialog, SWT.NONE).setText("Pyramid Levels:");
        Scale levelsScale = new Scale(dialog, SWT.HORIZONTAL);
        levelsScale.setMinimum(1);
        levelsScale.setMaximum(16);
        levelsScale.setSelection(nLevels);
        levelsScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label levelsLabel = new Label(dialog, SWT.NONE);
        levelsLabel.setText(String.valueOf(nLevels));
        levelsScale.addListener(SWT.Selection, e -> levelsLabel.setText(String.valueOf(levelsScale.getSelection())));

        // Show Rich checkbox
        Button richCheck = new Button(dialog, SWT.CHECK);
        richCheck.setText("Show Size & Orientation");
        richCheck.setSelection(showRich);
        GridData richGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
        richGd.horizontalSpan = 3;
        richCheck.setLayoutData(richGd);

        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(2, true));
        GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
        gd.horizontalSpan = 3;
        buttonComp.setLayoutData(gd);

        Button okBtn = new Button(buttonComp, SWT.PUSH);
        okBtn.setText("OK");
        okBtn.addListener(SWT.Selection, e -> {
            nFeatures = featScale.getSelection();
            fastThreshold = fastScale.getSelection();
            nLevels = levelsScale.getSelection();
            showRich = richCheck.getSelection();
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
