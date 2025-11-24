package com.example.pipeline.nodes;

import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.layout.RowLayout;
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
    private int dx = 1; // 0, 1, or 3
    private int dy = 0; // 0, 1, or 3
    private int kernelSizeIndex = 1; // Default to 3

    public SobelNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Sobel Edges", x, y);
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

        // Apply Sobel with dx and dy values
        Mat sobel = new Mat();
        Imgproc.Sobel(gray, sobel, CvType.CV_64F, dx, dy, ksize);

        Mat result = new Mat();
        Core.convertScaleAbs(sobel, result);
        sobel.release();

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
        dialog.setText("Sobel Edge Properties");
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

        // dx radio buttons
        new Label(dialog, SWT.NONE).setText("dx:");
        Composite dxComp = new Composite(dialog, SWT.NONE);
        RowLayout dxLayout = new RowLayout(SWT.HORIZONTAL);
        dxLayout.spacing = 10;
        dxComp.setLayout(dxLayout);

        Button dx0 = new Button(dxComp, SWT.RADIO);
        dx0.setText("0");
        dx0.setSelection(dx == 0);

        Button dx1 = new Button(dxComp, SWT.RADIO);
        dx1.setText("1");
        dx1.setSelection(dx == 1);

        Button dx3 = new Button(dxComp, SWT.RADIO);
        dx3.setText("3");
        dx3.setSelection(dx == 3);

        // dy radio buttons
        new Label(dialog, SWT.NONE).setText("dy:");
        Composite dyComp = new Composite(dialog, SWT.NONE);
        RowLayout dyLayout = new RowLayout(SWT.HORIZONTAL);
        dyLayout.spacing = 10;
        dyComp.setLayout(dyLayout);

        Button dy0 = new Button(dyComp, SWT.RADIO);
        dy0.setText("0");
        dy0.setSelection(dy == 0);

        Button dy1 = new Button(dyComp, SWT.RADIO);
        dy1.setText("1");
        dy1.setSelection(dy == 1);

        Button dy3 = new Button(dyComp, SWT.RADIO);
        dy3.setText("3");
        dy3.setSelection(dy == 3);

        // Kernel Size
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
            if (dx0.getSelection()) {
                dx = 0;
            } else if (dx1.getSelection()) {
                dx = 1;
            } else {
                dx = 3;
            }
            if (dy0.getSelection()) {
                dy = 0;
            } else if (dy1.getSelection()) {
                dy = 1;
            } else {
                dy = 3;
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
