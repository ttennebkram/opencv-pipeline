package com.example.pipeline.nodes;

import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Mat;

/**
 * Gain effect node.
 */
public class GainNode extends ProcessingNode {
    private double gain = 1.0;

    public GainNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Gain", x, y);
    }

    // Getters/setters for serialization
    public double getGain() { return gain; }
    public void setGain(double v) { gain = v; }

    @Override
    public Mat process(Mat input) {
        if (!enabled || input == null || input.empty()) {
            return input;
        }
        Mat output = new Mat();
        input.convertTo(output, -1, gain, 0);
        return output;
    }

    @Override
    public String getDescription() {
        return "Brightness/Gain Adjustment\ncv2.multiply(src, gain)";
    }

    @Override
    public String getDisplayName() {
        return "Gain";
    }

    @Override
    public String getCategory() {
        return "Basic";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Gain Properties");
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

        new Label(dialog, SWT.NONE).setText("Gain (0.1x - 10x):");
        Scale gainScale = new Scale(dialog, SWT.HORIZONTAL);
        gainScale.setMinimum(1);
        gainScale.setMaximum(100);
        // Use logarithmic mapping: scale value = log10(gain) * 50 + 50
        gainScale.setSelection((int)(Math.log10(gain) * 50 + 50));
        gainScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label gainLabel = new Label(dialog, SWT.NONE);
        gainLabel.setText(String.format("%.2fx", gain));
        gainScale.addListener(SWT.Selection, e -> {
            double logVal = (gainScale.getSelection() - 50) / 50.0;
            double g = Math.pow(10, logVal);
            gainLabel.setText(String.format("%.2fx", g));
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
            double logVal = (gainScale.getSelection() - 50) / 50.0;
            gain = Math.pow(10, logVal);
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
