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

    public LaplacianNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Laplacian", x, y);
    }

    // Getters/setters for serialization
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

        // Get kernel size
        int ksize = Integer.parseInt(KERNEL_SIZES[kernelSizeIndex]);

        // Apply Laplacian
        Mat laplacian = new Mat();
        Imgproc.Laplacian(gray, laplacian, CvType.CV_64F, ksize, 1, 0);

        // Convert to absolute and 8-bit
        Mat absLaplacian = new Mat();
        Core.convertScaleAbs(laplacian, absLaplacian);

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
