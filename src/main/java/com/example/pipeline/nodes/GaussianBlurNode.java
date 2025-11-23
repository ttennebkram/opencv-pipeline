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
 * Gaussian Blur effect node.
 */
public class GaussianBlurNode extends ProcessingNode {
    private int kernelSizeX = 7;
    private int kernelSizeY = 7;
    private double sigmaX = 0.0;

    public GaussianBlurNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Gaussian Blur", x, y);
    }

    // Getters/setters for serialization
    public int getKernelSizeX() { return kernelSizeX; }
    public void setKernelSizeX(int v) { kernelSizeX = v; }
    public int getKernelSizeY() { return kernelSizeY; }
    public void setKernelSizeY(int v) { kernelSizeY = v; }
    public double getSigmaX() { return sigmaX; }
    public void setSigmaX(double v) { sigmaX = v; }

    @Override
    public Mat process(Mat input) {
        if (!enabled || input == null || input.empty()) {
            return input;
        }
        // Ensure odd kernel sizes
        int kx = (kernelSizeX % 2 == 0) ? kernelSizeX + 1 : kernelSizeX;
        int ky = (kernelSizeY % 2 == 0) ? kernelSizeY + 1 : kernelSizeY;

        Mat output = new Mat();
        Imgproc.GaussianBlur(input, output, new Size(kx, ky), sigmaX);
        return output;
    }

    @Override
    public String getDescription() {
        return "Gaussian Blur\ncv2.GaussianBlur(src, ksize, sigmaX)";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Gaussian Blur Properties");
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

        // Sigma X
        new Label(dialog, SWT.NONE).setText("Sigma X:");
        Scale sigmaScale = new Scale(dialog, SWT.HORIZONTAL);
        sigmaScale.setMinimum(0);
        sigmaScale.setMaximum(100);
        sigmaScale.setSelection((int)(sigmaX * 10));
        sigmaScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label sigmaLabel = new Label(dialog, SWT.NONE);
        sigmaLabel.setText(sigmaX == 0 ? "0 (auto)" : String.format("%.1f", sigmaX));
        sigmaScale.addListener(SWT.Selection, e -> {
            double val = sigmaScale.getSelection() / 10.0;
            sigmaLabel.setText(val == 0 ? "0 (auto)" : String.format("%.1f", val));
        });

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
            sigmaX = sigmaScale.getSelection() / 10.0;
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
