package com.example.pipeline.nodes;

import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.*;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

/**
 * Threshold effect node.
 */
public class ThresholdNode extends ProcessingNode {
    private static final String[] TYPE_NAMES = {
        "BINARY", "BINARY_INV", "TRUNC", "TOZERO", "TOZERO_INV"
    };
    private static final int[] TYPE_CODES = {
        Imgproc.THRESH_BINARY, Imgproc.THRESH_BINARY_INV, Imgproc.THRESH_TRUNC,
        Imgproc.THRESH_TOZERO, Imgproc.THRESH_TOZERO_INV
    };
    private static final String[] MODIFIER_NAMES = {"None", "OTSU", "TRIANGLE"};
    private static final int[] MODIFIER_CODES = {0, Imgproc.THRESH_OTSU, Imgproc.THRESH_TRIANGLE};

    private int threshValue = 127;
    private int maxValue = 255;
    private int typeIndex = 0;
    private int modifierIndex = 0;
    private double returnedThreshold = 0; // Store the returned threshold value

    public ThresholdNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Threshold", x, y);
        // Increase height to accommodate the return value display
        this.height = NODE_HEIGHT + 15;
    }

    // Getters/setters for serialization
    public int getThreshValue() { return threshValue; }
    public void setThreshValue(int v) { threshValue = v; }
    public int getMaxValue() { return maxValue; }
    public void setMaxValue(int v) { maxValue = v; }
    public int getTypeIndex() { return typeIndex; }
    public void setTypeIndex(int v) { typeIndex = v; }
    public int getModifierIndex() { return modifierIndex; }
    public void setModifierIndex(int v) { modifierIndex = v; }
    public double getReturnedThreshold() { return returnedThreshold; }
    public void setReturnedThreshold(double v) { returnedThreshold = v; }

    @Override
    public Mat process(Mat input) {
        if (!enabled || input == null || input.empty()) {
            return input;
        }
        int combinedType = TYPE_CODES[typeIndex] | MODIFIER_CODES[modifierIndex];
        Mat output = new Mat();

        // OTSU and TRIANGLE require grayscale
        if (modifierIndex > 0) {
            Mat gray = new Mat();
            if (input.channels() == 3) {
                Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
            } else {
                gray = input.clone();
            }
            returnedThreshold = Imgproc.threshold(gray, output, threshValue, maxValue, combinedType);
            gray.release();

            // Convert back to BGR
            Mat bgr = new Mat();
            Imgproc.cvtColor(output, bgr, Imgproc.COLOR_GRAY2BGR);
            output.release();
            output = bgr;
        } else {
            returnedThreshold = Imgproc.threshold(input, output, threshValue, maxValue, combinedType);
        }
        return output;
    }

    @Override
    public String getDescription() {
        return "Binary Threshold\ncv2.threshold(src, thresh, maxval, type)";
    }

    @Override
    public String getDisplayName() {
        return "Threshold";
    }

    @Override
    public String getCategory() {
        return "Basic";
    }

    @Override
    public void paint(GC gc) {
        // Call parent paint to draw node background, title, thumbnail, connection points
        super.paint(gc);

        // Draw return value below thumbnail
        String returnText = String.format("T = %.0f", returnedThreshold);
        gc.setForeground(display.getSystemColor(SWT.COLOR_DARK_BLUE));
        Font smallFont = new Font(display, "Arial", 9, SWT.NORMAL);
        gc.setFont(smallFont);
        Point textExtent = gc.textExtent(returnText);
        int textX = x + (width - textExtent.x) / 2;
        int textY = y + height - 15;
        gc.drawString(returnText, textX, textY, true);
        smallFont.dispose();
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Threshold Properties");
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

        // Threshold value
        new Label(dialog, SWT.NONE).setText("Threshold:");
        Scale threshScale = new Scale(dialog, SWT.HORIZONTAL);
        threshScale.setMinimum(0);
        threshScale.setMaximum(255);
        threshScale.setSelection(threshValue);
        threshScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label threshLabel = new Label(dialog, SWT.NONE);
        threshLabel.setText(String.valueOf(threshValue));
        threshScale.addListener(SWT.Selection, e -> threshLabel.setText(String.valueOf(threshScale.getSelection())));

        // Max value
        new Label(dialog, SWT.NONE).setText("Max Value:");
        Scale maxScale = new Scale(dialog, SWT.HORIZONTAL);
        maxScale.setMinimum(0);
        maxScale.setMaximum(255);
        maxScale.setSelection(maxValue);
        maxScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label maxLabel = new Label(dialog, SWT.NONE);
        maxLabel.setText(String.valueOf(maxValue));
        maxScale.addListener(SWT.Selection, e -> maxLabel.setText(String.valueOf(maxScale.getSelection())));

        // Type
        new Label(dialog, SWT.NONE).setText("Type:");
        Combo typeCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        typeCombo.setItems(TYPE_NAMES);
        typeCombo.select(typeIndex);
        GridData typeGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
        typeGd.horizontalSpan = 2;
        typeCombo.setLayoutData(typeGd);

        // Modifier
        new Label(dialog, SWT.NONE).setText("Modifier:");
        Combo modCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        modCombo.setItems(MODIFIER_NAMES);
        modCombo.select(modifierIndex);
        GridData modGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
        modGd.horizontalSpan = 2;
        modCombo.setLayoutData(modGd);

        // Buttons
        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(2, true));
        GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
        gd.horizontalSpan = 3;
        buttonComp.setLayoutData(gd);

        Button okBtn = new Button(buttonComp, SWT.PUSH);
        okBtn.setText("OK");
        okBtn.addListener(SWT.Selection, e -> {
            threshValue = threshScale.getSelection();
            maxValue = maxScale.getSelection();
            typeIndex = typeCombo.getSelectionIndex();
            modifierIndex = modCombo.getSelectionIndex();
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
