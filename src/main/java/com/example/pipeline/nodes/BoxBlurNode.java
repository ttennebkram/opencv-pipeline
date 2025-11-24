package com.example.pipeline.nodes;

import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Box Blur (averaging) node - applies a simple normalized box filter.
 */
public class BoxBlurNode extends ProcessingNode {
    private int kernelSizeX = 5;
    private int kernelSizeY = 5;

    public BoxBlurNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Box Blur", x, y);
    }

    // Getters/setters for serialization
    public int getKernelSizeX() { return kernelSizeX; }
    public void setKernelSizeX(int v) { kernelSizeX = v; }
    public int getKernelSizeY() { return kernelSizeY; }
    public void setKernelSizeY(int v) { kernelSizeY = v; }

    @Override
    public Mat process(Mat input) {
        if (!enabled || input == null || input.empty()) {
            return input;
        }

        Mat output = new Mat();
        Imgproc.blur(input, output, new Size(kernelSizeX, kernelSizeY));
        return output;
    }

    @Override
    public String getDescription() {
        return "Box Blur\ncv2.blur(src, ksize)";
    }

    @Override
    public String getDisplayName() {
        return "Box Blur";
    }

    @Override
    public String getCategory() {
        return "Blur";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Box Blur Properties");
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

        // Kernel Size X
        new Label(dialog, SWT.NONE).setText("Kernel Size X:");
        Scale kxScale = new Scale(dialog, SWT.HORIZONTAL);
        kxScale.setMinimum(1);
        kxScale.setMaximum(31);
        kxScale.setSelection(kernelSizeX);
        kxScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label kxLabel = new Label(dialog, SWT.NONE);
        kxLabel.setText(String.valueOf(kernelSizeX));
        kxScale.addListener(SWT.Selection, e -> kxLabel.setText(String.valueOf(kxScale.getSelection())));

        // Kernel Size Y
        new Label(dialog, SWT.NONE).setText("Kernel Size Y:");
        Scale kyScale = new Scale(dialog, SWT.HORIZONTAL);
        kyScale.setMinimum(1);
        kyScale.setMaximum(31);
        kyScale.setSelection(kernelSizeY);
        kyScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label kyLabel = new Label(dialog, SWT.NONE);
        kyLabel.setText(String.valueOf(kernelSizeY));
        kyScale.addListener(SWT.Selection, e -> kyLabel.setText(String.valueOf(kyScale.getSelection())));

        // Buttons
        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(2, true));
        GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
        gd.horizontalSpan = 3;
        buttonComp.setLayoutData(gd);

        Button okBtn = new Button(buttonComp, SWT.PUSH);
        okBtn.setText("OK");
        okBtn.addListener(SWT.Selection, e -> {
            kernelSizeX = kxScale.getSelection();
            kernelSizeY = kyScale.getSelection();
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
}
