package com.example.pipeline.nodes;

import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

/**
 * Median Blur node.
 */
public class MedianBlurNode extends ProcessingNode {
    private int kernelSize = 5;

    public MedianBlurNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Median Blur", x, y);
    }

    // Getters/setters for serialization
    public int getKernelSize() { return kernelSize; }
    public void setKernelSize(int v) { kernelSize = v; }

    @Override
    public Mat process(Mat input) {
        if (!enabled || input == null || input.empty()) {
            return input;
        }
        // Ensure odd kernel size
        int ksize = (kernelSize % 2 == 0) ? kernelSize + 1 : kernelSize;

        Mat output = new Mat();
        Imgproc.medianBlur(input, output, ksize);
        return output;
    }

    @Override
    public String getDescription() {
        return "Median Blur\ncv2.medianBlur(src, ksize)";
    }

    @Override
    public String getDisplayName() {
        return "Median Blur";
    }

    @Override
    public String getCategory() {
        return "Blur";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Median Blur Properties");
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

        // Kernel Size
        new Label(dialog, SWT.NONE).setText("Kernel Size:");
        Scale kScale = new Scale(dialog, SWT.HORIZONTAL);
        kScale.setMinimum(1);
        kScale.setMaximum(31);
        kScale.setSelection(kernelSize);
        kScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label kLabel = new Label(dialog, SWT.NONE);
        kLabel.setText(String.valueOf(kernelSize));
        kScale.addListener(SWT.Selection, e -> kLabel.setText(String.valueOf(kScale.getSelection())));

        // Buttons
        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(2, true));
        GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
        gd.horizontalSpan = 3;
        buttonComp.setLayoutData(gd);

        Button okBtn = new Button(buttonComp, SWT.PUSH);
        okBtn.setText("OK");
        okBtn.addListener(SWT.Selection, e -> {
            kernelSize = kScale.getSelection();
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
