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
 * Sobel Edge Detection node.
 */
public class SobelNode extends ProcessingNode {
    private static final String[] KERNEL_SIZES = {"1", "3", "5", "7"};
    private int dx = 1;
    private int dy = 0;
    private int kernelSizeIndex = 1; // Default to 3

    public SobelNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Sobel", x, y);
    }

    // Getters/setters for serialization
    public int getDx() { return dx; }
    public void setDx(int v) { dx = v; }
    public int getDy() { return dy; }
    public void setDy(int v) { dy = v; }
    public int getKernelSizeIndex() { return kernelSizeIndex; }
    public void setKernelSizeIndex(int v) { kernelSizeIndex = v; }

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

        int ksize = Integer.parseInt(KERNEL_SIZES[kernelSizeIndex]);

        // Compute gradients
        Mat gradX = new Mat();
        Mat gradY = new Mat();

        if (dx > 0) {
            Imgproc.Sobel(gray, gradX, CvType.CV_64F, dx, 0, ksize);
        }
        if (dy > 0) {
            Imgproc.Sobel(gray, gradY, CvType.CV_64F, 0, dy, ksize);
        }

        Mat result;
        if (dx > 0 && dy > 0) {
            // Combine both gradients
            Mat absX = new Mat();
            Mat absY = new Mat();
            Core.convertScaleAbs(gradX, absX);
            Core.convertScaleAbs(gradY, absY);
            result = new Mat();
            Core.addWeighted(absX, 0.5, absY, 0.5, 0, result);
            absX.release();
            absY.release();
        } else if (dx > 0) {
            result = new Mat();
            Core.convertScaleAbs(gradX, result);
        } else {
            result = new Mat();
            Core.convertScaleAbs(gradY, result);
        }

        // Convert back to BGR for display
        Mat output = new Mat();
        Imgproc.cvtColor(result, output, Imgproc.COLOR_GRAY2BGR);

        // Clean up
        if (gray != input) {
            gray.release();
        }
        if (!gradX.empty()) gradX.release();
        if (!gradY.empty()) gradY.release();
        result.release();

        return output;
    }

    @Override
    public String getDescription() {
        return "Sobel Edge Detection\ncv2.Sobel(src, ddepth, dx, dy, ksize)";
    }

    @Override
    public String getDisplayName() {
        return "Sobel Edges";
    }

    @Override
    public String getCategory() {
        return "Edge Detection";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Sobel Properties");
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

        // dx
        new Label(dialog, SWT.NONE).setText("dx (X derivative):");
        Scale dxScale = new Scale(dialog, SWT.HORIZONTAL);
        dxScale.setMinimum(0);
        dxScale.setMaximum(2);
        dxScale.setSelection(dx);
        dxScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label dxLabel = new Label(dialog, SWT.NONE);
        dxLabel.setText(String.valueOf(dx));
        dxScale.addListener(SWT.Selection, e -> dxLabel.setText(String.valueOf(dxScale.getSelection())));

        // dy
        new Label(dialog, SWT.NONE).setText("dy (Y derivative):");
        Scale dyScale = new Scale(dialog, SWT.HORIZONTAL);
        dyScale.setMinimum(0);
        dyScale.setMaximum(2);
        dyScale.setSelection(dy);
        dyScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label dyLabel = new Label(dialog, SWT.NONE);
        dyLabel.setText(String.valueOf(dy));
        dyScale.addListener(SWT.Selection, e -> dyLabel.setText(String.valueOf(dyScale.getSelection())));

        // Kernel Size
        new Label(dialog, SWT.NONE).setText("Kernel Size:");
        Combo ksizeCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        ksizeCombo.setItems(KERNEL_SIZES);
        ksizeCombo.select(kernelSizeIndex);
        GridData comboGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
        comboGd.horizontalSpan = 2;
        ksizeCombo.setLayoutData(comboGd);

        // Buttons
        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(2, true));
        GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
        gd.horizontalSpan = 3;
        buttonComp.setLayoutData(gd);

        Button okBtn = new Button(buttonComp, SWT.PUSH);
        okBtn.setText("OK");
        okBtn.addListener(SWT.Selection, e -> {
            dx = dxScale.getSelection();
            dy = dyScale.getSelection();
            // Ensure at least one derivative is non-zero
            if (dx == 0 && dy == 0) {
                dx = 1;
            }
            kernelSizeIndex = ksizeCombo.getSelectionIndex();
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
