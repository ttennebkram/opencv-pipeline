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
 * Laplacian Edge Detection node.
 */
public class LaplacianNode extends ProcessingNode {
    private static final String[] KERNEL_SIZES = {"1", "3", "5", "7"};
    private int kernelSizeIndex = 1; // Default to 3
    private int scalePercent = 100; // Scale 0.1-5.0 stored as 10-500 percent
    private int delta = 0; // 0-255
    private boolean useAbsolute = true;

    public LaplacianNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Laplacian Edges", x, y);
    }

    // Getters/setters for serialization
    public int getKernelSizeIndex() { return kernelSizeIndex; }
    public void setKernelSizeIndex(int v) { kernelSizeIndex = v; }
    public int getScalePercent() { return scalePercent; }
    public void setScalePercent(int v) { scalePercent = v; }
    public int getDelta() { return delta; }
    public void setDelta(int v) { delta = v; }
    public boolean isUseAbsolute() { return useAbsolute; }
    public void setUseAbsolute(boolean v) { useAbsolute = v; }

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

        // Get kernel size
        int ksize = Integer.parseInt(KERNEL_SIZES[kernelSizeIndex]);
        double scale = scalePercent / 100.0;

        // Apply Laplacian
        Mat laplacian = new Mat();
        Imgproc.Laplacian(gray, laplacian, CvType.CV_64F, ksize, scale, delta);

        // Convert to absolute and 8-bit
        Mat absLaplacian = new Mat();
        if (useAbsolute) {
            Core.convertScaleAbs(laplacian, absLaplacian);
        } else {
            laplacian.convertTo(absLaplacian, CvType.CV_8U);
        }

        // Convert back to BGR for display
        Mat output = new Mat();
        Imgproc.cvtColor(absLaplacian, output, Imgproc.COLOR_GRAY2BGR);

        // Clean up
        if (gray != input) {
            gray.release();
        }
        laplacian.release();
        absLaplacian.release();

        return output;
    }

    @Override
    public String getDescription() {
        return "Laplacian Edge Detection\ncv2.Laplacian(src, ddepth)";
    }

    @Override
    public String getDisplayName() {
        return "Laplacian Edges";
    }

    @Override
    public String getCategory() {
        return "Edge Detection";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Laplacian Properties");
        dialog.setLayout(new GridLayout(2, false));

        // Method signature
        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText(getDescription());
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sigGd.horizontalSpan = 2;
        sigLabel.setLayoutData(sigGd);

        // Separator
        Label sep = new Label(dialog, SWT.SEPARATOR | SWT.HORIZONTAL);
        GridData sepGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sepGd.horizontalSpan = 2;
        sep.setLayoutData(sepGd);

        // Kernel Size dropdown
        new Label(dialog, SWT.NONE).setText("Kernel Size:");
        Combo ksizeCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        ksizeCombo.setItems(KERNEL_SIZES);
        ksizeCombo.select(kernelSizeIndex);

        // Scale slider (10-500 percent, displayed as 0.1-5.0)
        new Label(dialog, SWT.NONE).setText("Scale:");
        Composite scaleComp = new Composite(dialog, SWT.NONE);
        scaleComp.setLayout(new GridLayout(2, false));
        Scale scaleSlider = new Scale(scaleComp, SWT.HORIZONTAL);
        scaleSlider.setMinimum(10);
        scaleSlider.setMaximum(500);
        // Clamp slider position to valid range, but keep actual value
        int scaleSliderPos = Math.min(Math.max(scalePercent, 10), 500);
        scaleSlider.setSelection(scaleSliderPos);
        scaleSlider.setLayoutData(new GridData(150, SWT.DEFAULT));
        Label scaleLabel = new Label(scaleComp, SWT.NONE);
        scaleLabel.setText(String.format("%.1f", scalePercent / 100.0)); // Show real value
        scaleSlider.addListener(SWT.Selection, e -> scaleLabel.setText(String.format("%.1f", scaleSlider.getSelection() / 100.0)));

        // Delta slider (0-255)
        new Label(dialog, SWT.NONE).setText("Delta:");
        Composite deltaComp = new Composite(dialog, SWT.NONE);
        deltaComp.setLayout(new GridLayout(2, false));
        Scale deltaSlider = new Scale(deltaComp, SWT.HORIZONTAL);
        deltaSlider.setMinimum(0);
        deltaSlider.setMaximum(255);
        // Clamp slider position to valid range, but keep actual value
        int deltaSliderPos = Math.min(Math.max(delta, 0), 255);
        deltaSlider.setSelection(deltaSliderPos);
        deltaSlider.setLayoutData(new GridData(150, SWT.DEFAULT));
        Label deltaLabel = new Label(deltaComp, SWT.NONE);
        deltaLabel.setText(String.valueOf(delta)); // Show real value
        deltaSlider.addListener(SWT.Selection, e -> deltaLabel.setText(String.valueOf(deltaSlider.getSelection())));

        // Use Absolute checkbox
        Button absCheck = new Button(dialog, SWT.CHECK);
        absCheck.setText("Use Absolute Value");
        absCheck.setSelection(useAbsolute);
        GridData absGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        absGd.horizontalSpan = 2;
        absCheck.setLayoutData(absGd);

        // Buttons
        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(2, true));
        GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
        gd.horizontalSpan = 2;
        buttonComp.setLayoutData(gd);

        Button okBtn = new Button(buttonComp, SWT.PUSH);
        okBtn.setText("OK");
        okBtn.addListener(SWT.Selection, e -> {
            kernelSizeIndex = ksizeCombo.getSelectionIndex();
            scalePercent = scaleSlider.getSelection();
            delta = deltaSlider.getSelection();
            useAbsolute = absCheck.getSelection();
            dialog.dispose();
            notifyChanged();
        });

        Button cancelBtn = new Button(buttonComp, SWT.PUSH);
        cancelBtn.setText("Cancel");
        cancelBtn.addListener(SWT.Selection, e -> dialog.dispose());

        dialog.pack();
        // Ensure minimum width for title
        Point size = dialog.getSize();
        if (size.x < 250) {
            dialog.setSize(250, size.y);
        }
        Point cursor = shell.getDisplay().getCursorLocation();
        dialog.setLocation(cursor.x, cursor.y);
        dialog.open();
    }
}
