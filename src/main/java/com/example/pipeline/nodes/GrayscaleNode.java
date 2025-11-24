package com.example.pipeline.nodes;

import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

/**
 * Grayscale / Color Conversion node.
 */
public class GrayscaleNode extends ProcessingNode {
    private static final String[] CONVERSION_NAMES = {
        "BGR to Grayscale", "BGR to RGB", "BGR to HSV", "BGR to HLS",
        "BGR to LAB", "BGR to LUV", "BGR to YCrCb", "BGR to XYZ"
    };
    private static final int[] CONVERSION_CODES = {
        Imgproc.COLOR_BGR2GRAY, Imgproc.COLOR_BGR2RGB, Imgproc.COLOR_BGR2HSV,
        Imgproc.COLOR_BGR2HLS, Imgproc.COLOR_BGR2Lab, Imgproc.COLOR_BGR2Luv,
        Imgproc.COLOR_BGR2YCrCb, Imgproc.COLOR_BGR2XYZ
    };
    private int conversionIndex = 0;

    public GrayscaleNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Grayscale/Color Convert", x, y);
    }

    // Getters/setters for serialization
    public int getConversionIndex() { return conversionIndex; }
    public void setConversionIndex(int v) { conversionIndex = v; }

    @Override
    public Mat process(Mat input) {
        if (!enabled || input == null || input.empty()) {
            return input;
        }
        Mat output = new Mat();
        Imgproc.cvtColor(input, output, CONVERSION_CODES[conversionIndex]);

        // Convert grayscale back to BGR for display
        if (output.channels() == 1) {
            Mat bgr = new Mat();
            Imgproc.cvtColor(output, bgr, Imgproc.COLOR_GRAY2BGR);
            output.release();
            output = bgr;
        }
        return output;
    }

    @Override
    public String getDescription() {
        return "Grayscale or Color Conversion\ncv2.cvtColor(src, code)";
    }

    @Override
    public String getDisplayName() {
        return "Grayscale/Color Convert";
    }

    @Override
    public String getCategory() {
        return "Basic";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Color Conversion Properties");
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

        new Label(dialog, SWT.NONE).setText("Conversion:");
        Combo combo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        combo.setItems(CONVERSION_NAMES);
        combo.select(conversionIndex);

        // Buttons
        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(2, true));
        GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
        gd.horizontalSpan = 2;
        buttonComp.setLayoutData(gd);

        Button okBtn = new Button(buttonComp, SWT.PUSH);
        okBtn.setText("OK");
        okBtn.addListener(SWT.Selection, e -> {
            conversionIndex = combo.getSelectionIndex();
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
