package com.example.pipeline.nodes;

import org.eclipse.swt.SWT;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.*;

/**
 * Crop node - crops a region of interest from the image.
 */
public class CropNode extends ProcessingNode {
    private int cropX = 0, cropY = 0;  // Top-left corner
    private int cropWidth = 100, cropHeight = 100; // Size of crop region

    public CropNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Crop", x, y);
    }

    // Getters/setters for serialization
    public int getCropX() { return cropX; }
    public void setCropX(int v) { cropX = v; }
    public int getCropY() { return cropY; }
    public void setCropY(int v) { cropY = v; }
    public int getCropWidth() { return cropWidth; }
    public void setCropWidth(int v) { cropWidth = v; }
    public int getCropHeight() { return cropHeight; }
    public void setCropHeight(int v) { cropHeight = v; }

    // Helper to convert relative coords to absolute
    private int toAbsoluteX(int val, int imgWidth) {
        return val < 0 ? imgWidth + val : val;
    }

    private int toAbsoluteY(int val, int imgHeight) {
        return val < 0 ? imgHeight + val : val;
    }

    @Override
    public Mat process(Mat input) {
        int imgWidth = input.width();
        int imgHeight = input.height();

        // Convert coordinates (negative values are relative to right/bottom)
        int absX = toAbsoluteX(cropX, imgWidth);
        int absY = toAbsoluteY(cropY, imgHeight);

        // Clamp to valid range
        absX = Math.max(0, Math.min(absX, imgWidth - 1));
        absY = Math.max(0, Math.min(absY, imgHeight - 1));

        // Calculate actual width/height within bounds
        int actualWidth = Math.min(cropWidth, imgWidth - absX);
        int actualHeight = Math.min(cropHeight, imgHeight - absY);

        if (actualWidth <= 0 || actualHeight <= 0) {
            return input.clone(); // Return original if crop region is invalid
        }

        // Create ROI and return cropped image
        Rect roi = new Rect(absX, absY, actualWidth, actualHeight);
        return new Mat(input, roi).clone();
    }

    @Override
    public String getDescription() {
        return "Crop Image\nimg[y:y+h, x:x+w]";
    }

    @Override
    public String getDisplayName() {
        return "Crop";
    }

    @Override
    public String getCategory() {
        return "Transform";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Crop Properties");
        dialog.setLayout(new GridLayout(2, false));

        // Description
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

        // X
        new Label(dialog, SWT.NONE).setText("X:");
        Spinner xSpinner = new Spinner(dialog, SWT.BORDER);
        xSpinner.setMinimum(-4096);
        xSpinner.setMaximum(4096);
        xSpinner.setSelection(cropX);
        xSpinner.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Y
        new Label(dialog, SWT.NONE).setText("Y:");
        Spinner ySpinner = new Spinner(dialog, SWT.BORDER);
        ySpinner.setMinimum(-4096);
        ySpinner.setMaximum(4096);
        ySpinner.setSelection(cropY);
        ySpinner.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Width
        new Label(dialog, SWT.NONE).setText("Width:");
        Spinner widthSpinner = new Spinner(dialog, SWT.BORDER);
        widthSpinner.setMinimum(1);
        widthSpinner.setMaximum(4096);
        widthSpinner.setSelection(cropWidth);
        widthSpinner.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Height
        new Label(dialog, SWT.NONE).setText("Height:");
        Spinner heightSpinner = new Spinner(dialog, SWT.BORDER);
        heightSpinner.setMinimum(1);
        heightSpinner.setMaximum(4096);
        heightSpinner.setSelection(cropHeight);
        heightSpinner.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Buttons
        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(2, true));
        GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
        gd.horizontalSpan = 2;
        buttonComp.setLayoutData(gd);

        Button okBtn = new Button(buttonComp, SWT.PUSH);
        okBtn.setText("OK");
        okBtn.addListener(SWT.Selection, e -> {
            cropX = xSpinner.getSelection();
            cropY = ySpinner.getSelection();
            cropWidth = widthSpinner.getSelection();
            cropHeight = heightSpinner.getSelection();
            dialog.dispose();
            notifyChanged();
        });

        Button cancelBtn = new Button(buttonComp, SWT.PUSH);
        cancelBtn.setText("Cancel");
        cancelBtn.addListener(SWT.Selection, e -> dialog.dispose());

        dialog.pack();
        org.eclipse.swt.graphics.Point cursorLoc = shell.getDisplay().getCursorLocation();
        dialog.setLocation(cursorLoc.x, cursorLoc.y);
        dialog.open();
    }
}
