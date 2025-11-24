package com.example.pipeline.nodes;

import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

/**
 * Scharr Edge Detection node.
 */
public class ScharrNode extends ProcessingNode {
    private static final String[] DIRECTIONS = {"X", "Y", "Both"};
    private int directionIndex = 2; // Default to Both
    private int scalePercent = 100; // 1.0 * 100
    private int delta = 0;

    public ScharrNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Scharr Edges", x, y);
    }

    // Getters/setters for serialization
    public int getDirectionIndex() { return directionIndex; }
    public void setDirectionIndex(int v) { directionIndex = v; }
    public int getScalePercent() { return scalePercent; }
    public void setScalePercent(int v) { scalePercent = v; }
    public int getDelta() { return delta; }
    public void setDelta(int v) { delta = v; }

    @Override
    public Mat process(Mat input) {
        if (!enabled || input == null || input.empty()) {
            return input;
        }

        // Convert to grayscale if needed
        Mat gray;
        if (input.channels() == 3) {
            gray = new Mat();
            Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
        } else {
            gray = input;
        }

        double scale = scalePercent / 100.0;

        Mat result;
        if (directionIndex == 0) { // X only
            Mat scharrX = new Mat();
            Imgproc.Scharr(gray, scharrX, CvType.CV_64F, 1, 0, scale, delta);
            result = new Mat();
            Core.convertScaleAbs(scharrX, result);
            scharrX.release();
        } else if (directionIndex == 1) { // Y only
            Mat scharrY = new Mat();
            Imgproc.Scharr(gray, scharrY, CvType.CV_64F, 0, 1, scale, delta);
            result = new Mat();
            Core.convertScaleAbs(scharrY, result);
            scharrY.release();
        } else { // Both
            Mat scharrX = new Mat();
            Mat scharrY = new Mat();
            Imgproc.Scharr(gray, scharrX, CvType.CV_64F, 1, 0, scale, delta);
            Imgproc.Scharr(gray, scharrY, CvType.CV_64F, 0, 1, scale, delta);

            Mat absX = new Mat();
            Mat absY = new Mat();
            Core.convertScaleAbs(scharrX, absX);
            Core.convertScaleAbs(scharrY, absY);

            result = new Mat();
            Core.addWeighted(absX, 0.5, absY, 0.5, 0, result);

            scharrX.release();
            scharrY.release();
            absX.release();
            absY.release();
        }

        // Convert back to BGR for display
        Mat output = new Mat();
        Imgproc.cvtColor(result, output, Imgproc.COLOR_GRAY2BGR);

        if (gray != input) {
            gray.release();
        }
        result.release();

        return output;
    }

    @Override
    public String getDescription() {
        return "Scharr Edge Detection\ncv2.Scharr(src, ddepth, dx, dy)";
    }

    @Override
    public String getDisplayName() {
        return "Scharr Edges";
    }

    @Override
    public String getCategory() {
        return "Edge Detection";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Scharr Edge Properties");
        dialog.setLayout(new GridLayout(2, false));

        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText(getDescription());
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sigGd.horizontalSpan = 2;
        sigLabel.setLayoutData(sigGd);

        Label sep = new Label(dialog, SWT.SEPARATOR | SWT.HORIZONTAL);
        GridData sepGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sepGd.horizontalSpan = 2;
        sep.setLayoutData(sepGd);

        new Label(dialog, SWT.NONE).setText("Direction:");
        Combo dirCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        dirCombo.setItems(DIRECTIONS);
        dirCombo.select(directionIndex);

        // Scale
        new Label(dialog, SWT.NONE).setText("Scale (%):");
        Scale scaleScale = new Scale(dialog, SWT.HORIZONTAL);
        scaleScale.setMinimum(10);
        scaleScale.setMaximum(500);
        // Clamp slider position to valid range, but keep actual value
        int scaleSliderPos = Math.min(Math.max(scalePercent, 10), 500);
        scaleScale.setSelection(scaleSliderPos);
        scaleScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        // Delta
        new Label(dialog, SWT.NONE).setText("Delta:");
        Scale deltaScale = new Scale(dialog, SWT.HORIZONTAL);
        deltaScale.setMinimum(0);
        deltaScale.setMaximum(255);
        // Clamp slider position to valid range, but keep actual value
        int deltaSliderPos = Math.min(Math.max(delta, 0), 255);
        deltaScale.setSelection(deltaSliderPos);
        deltaScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(2, true));
        GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
        gd.horizontalSpan = 2;
        buttonComp.setLayoutData(gd);

        Button okBtn = new Button(buttonComp, SWT.PUSH);
        okBtn.setText("OK");
        okBtn.addListener(SWT.Selection, e -> {
            directionIndex = dirCombo.getSelectionIndex();
            scalePercent = scaleScale.getSelection();
            delta = deltaScale.getSelection();
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
