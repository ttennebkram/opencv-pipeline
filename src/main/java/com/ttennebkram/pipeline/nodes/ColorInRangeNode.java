package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.registry.NodeInfo;
import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

/**
 * Color In Range node - HSV/BGR color filtering.
 */
@NodeInfo(name = "ColorInRange", category = "Filter", aliases = {"Color In Range"})
public class ColorInRangeNode extends ProcessingNode {
    private boolean useHSV = true;
    private int hLow = 0, hHigh = 179;
    private int sLow = 0, sHigh = 255;
    private int vLow = 0, vHigh = 255;
    private int outputMode = 0; // 0=mask, 1=masked, 2=inverse

    private static final String[] OUTPUT_MODES = {"Mask Only", "Keep In-Range", "Keep Out-of-Range"};

    public ColorInRangeNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Color In Range", x, y);
    }

    // Getters/setters for serialization
    public boolean isUseHSV() { return useHSV; }
    public void setUseHSV(boolean v) { useHSV = v; }
    public int getHLow() { return hLow; }
    public void setHLow(int v) { hLow = v; }
    public int getHHigh() { return hHigh; }
    public void setHHigh(int v) { hHigh = v; }
    public int getSLow() { return sLow; }
    public void setSLow(int v) { sLow = v; }
    public int getSHigh() { return sHigh; }
    public void setSHigh(int v) { sHigh = v; }
    public int getVLow() { return vLow; }
    public void setVLow(int v) { vLow = v; }
    public int getVHigh() { return vHigh; }
    public void setVHigh(int v) { vHigh = v; }
    public int getOutputMode() { return outputMode; }
    public void setOutputMode(int v) { outputMode = v; }

    @Override
    public Mat process(Mat input) {
        if (!enabled || input == null || input.empty()) return input;

        // Ensure input is color
        Mat colorInput = input;
        if (input.channels() == 1) {
            colorInput = new Mat();
            Imgproc.cvtColor(input, colorInput, Imgproc.COLOR_GRAY2BGR);
        }

        // Convert to HSV if needed
        Mat converted = new Mat();
        if (useHSV) {
            Imgproc.cvtColor(colorInput, converted, Imgproc.COLOR_BGR2HSV);
        } else {
            converted = colorInput.clone();
        }

        // Create lower and upper bounds
        Scalar lower = new Scalar(hLow, sLow, vLow);
        Scalar upper = new Scalar(hHigh, sHigh, vHigh);

        // Create mask
        Mat mask = new Mat();
        Core.inRange(converted, lower, upper, mask);

        Mat result = new Mat();
        switch (outputMode) {
            case 0: // Mask only
                Imgproc.cvtColor(mask, result, Imgproc.COLOR_GRAY2BGR);
                break;
            case 1: // Keep in-range
                result = new Mat();
                colorInput.copyTo(result, mask);
                break;
            case 2: // Keep out-of-range (inverse)
                Mat invMask = new Mat();
                Core.bitwise_not(mask, invMask);
                result = new Mat();
                colorInput.copyTo(result, invMask);
                break;
            default:
                Imgproc.cvtColor(mask, result, Imgproc.COLOR_GRAY2BGR);
        }

        return result;
    }

    @Override
    public String getDescription() {
        return "Color Range Filter\ncv2.inRange(src, lowerb, upperb)";
    }

    @Override
    public String getDisplayName() {
        return "Color In Range";
    }

    @Override
    public String getCategory() {
        return "Basic";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Color In Range Properties");
        dialog.setLayout(new GridLayout(3, false));

        // Method signature
        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText(getDescription());
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sigGd.horizontalSpan = 3;
        sigLabel.setLayoutData(sigGd);

        // Color space checkbox
        Button hsvCheck = new Button(dialog, SWT.CHECK);
        hsvCheck.setText("Use HSV (uncheck for BGR)");
        hsvCheck.setSelection(useHSV);
        GridData checkGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
        checkGd.horizontalSpan = 3;
        hsvCheck.setLayoutData(checkGd);

        // H/B Low
        Label hLowLabelName = new Label(dialog, SWT.NONE);
        hLowLabelName.setText(useHSV ? "H Low:" : "B Low:");
        Scale hLowScale = new Scale(dialog, SWT.HORIZONTAL);
        hLowScale.setMinimum(0);
        hLowScale.setMaximum(useHSV ? 179 : 255);
        hLowScale.setSelection(Math.min(hLow, hLowScale.getMaximum()));
        hLowScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label hLowLabel = new Label(dialog, SWT.NONE);
        hLowLabel.setText(String.valueOf(hLowScale.getSelection()));
        hLowScale.addListener(SWT.Selection, e -> hLowLabel.setText(String.valueOf(hLowScale.getSelection())));

        // H/B High
        Label hHighLabelName = new Label(dialog, SWT.NONE);
        hHighLabelName.setText(useHSV ? "H High:" : "B High:");
        Scale hHighScale = new Scale(dialog, SWT.HORIZONTAL);
        hHighScale.setMinimum(0);
        hHighScale.setMaximum(useHSV ? 179 : 255);
        hHighScale.setSelection(Math.min(hHigh, hHighScale.getMaximum()));
        hHighScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label hHighLabel = new Label(dialog, SWT.NONE);
        hHighLabel.setText(String.valueOf(hHighScale.getSelection()));
        hHighScale.addListener(SWT.Selection, e -> hHighLabel.setText(String.valueOf(hHighScale.getSelection())));

        // Update slider maximums when HSV checkbox changes
        hsvCheck.addListener(SWT.Selection, e -> {
            boolean isHSV = hsvCheck.getSelection();
            int hMax = isHSV ? 179 : 255;
            hLowScale.setMaximum(hMax);
            hHighScale.setMaximum(hMax);
            if (hLowScale.getSelection() > hMax) {
                hLowScale.setSelection(hMax);
                hLowLabel.setText(String.valueOf(hMax));
            }
            if (hHighScale.getSelection() > hMax) {
                hHighScale.setSelection(hMax);
                hHighLabel.setText(String.valueOf(hMax));
            }
            hLowLabelName.setText(isHSV ? "H Low:" : "B Low:");
            hHighLabelName.setText(isHSV ? "H High:" : "B High:");
        });

        // S/G Low
        new Label(dialog, SWT.NONE).setText(useHSV ? "S Low:" : "G Low:");
        Scale sLowScale = new Scale(dialog, SWT.HORIZONTAL);
        sLowScale.setMinimum(0);
        sLowScale.setMaximum(255);
        sLowScale.setSelection(sLow);
        sLowScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label sLowLabel = new Label(dialog, SWT.NONE);
        sLowLabel.setText(String.valueOf(sLow));
        sLowScale.addListener(SWT.Selection, e -> sLowLabel.setText(String.valueOf(sLowScale.getSelection())));

        // S/G High
        new Label(dialog, SWT.NONE).setText(useHSV ? "S High:" : "G High:");
        Scale sHighScale = new Scale(dialog, SWT.HORIZONTAL);
        sHighScale.setMinimum(0);
        sHighScale.setMaximum(255);
        sHighScale.setSelection(sHigh);
        sHighScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label sHighLabel = new Label(dialog, SWT.NONE);
        sHighLabel.setText(String.valueOf(sHigh));
        sHighScale.addListener(SWT.Selection, e -> sHighLabel.setText(String.valueOf(sHighScale.getSelection())));

        // V/R Low
        new Label(dialog, SWT.NONE).setText(useHSV ? "V Low:" : "R Low:");
        Scale vLowScale = new Scale(dialog, SWT.HORIZONTAL);
        vLowScale.setMinimum(0);
        vLowScale.setMaximum(255);
        vLowScale.setSelection(vLow);
        vLowScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label vLowLabel = new Label(dialog, SWT.NONE);
        vLowLabel.setText(String.valueOf(vLow));
        vLowScale.addListener(SWT.Selection, e -> vLowLabel.setText(String.valueOf(vLowScale.getSelection())));

        // V/R High
        new Label(dialog, SWT.NONE).setText(useHSV ? "V High:" : "R High:");
        Scale vHighScale = new Scale(dialog, SWT.HORIZONTAL);
        vHighScale.setMinimum(0);
        vHighScale.setMaximum(255);
        vHighScale.setSelection(vHigh);
        vHighScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label vHighLabel = new Label(dialog, SWT.NONE);
        vHighLabel.setText(String.valueOf(vHigh));
        vHighScale.addListener(SWT.Selection, e -> vHighLabel.setText(String.valueOf(vHighScale.getSelection())));

        // Output mode
        new Label(dialog, SWT.NONE).setText("Output:");
        Combo modeCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        modeCombo.setItems(OUTPUT_MODES);
        modeCombo.select(outputMode);
        GridData comboGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
        comboGd.horizontalSpan = 2;
        modeCombo.setLayoutData(comboGd);

        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(2, true));
        GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
        gd.horizontalSpan = 3;
        buttonComp.setLayoutData(gd);

        Button okBtn = new Button(buttonComp, SWT.PUSH);
        okBtn.setText("OK");
        dialog.setDefaultButton(okBtn);
        okBtn.addListener(SWT.Selection, e -> {
            useHSV = hsvCheck.getSelection();
            hLow = hLowScale.getSelection();
            hHigh = hHighScale.getSelection();
            sLow = sLowScale.getSelection();
            sHigh = sHighScale.getSelection();
            vLow = vLowScale.getSelection();
            vHigh = vHighScale.getSelection();
            outputMode = modeCombo.getSelectionIndex();
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

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("useHSV", useHSV);
        json.addProperty("hLow", hLow);
        json.addProperty("hHigh", hHigh);
        json.addProperty("sLow", sLow);
        json.addProperty("sHigh", sHigh);
        json.addProperty("vLow", vLow);
        json.addProperty("vHigh", vHigh);
        json.addProperty("outputMode", outputMode);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        if (json.has("useHSV")) useHSV = json.get("useHSV").getAsBoolean();
        if (json.has("hLow")) hLow = json.get("hLow").getAsInt();
        if (json.has("hHigh")) hHigh = json.get("hHigh").getAsInt();
        if (json.has("sLow")) sLow = json.get("sLow").getAsInt();
        if (json.has("sHigh")) sHigh = json.get("sHigh").getAsInt();
        if (json.has("vLow")) vLow = json.get("vLow").getAsInt();
        if (json.has("vHigh")) vHigh = json.get("vHigh").getAsInt();
        if (json.has("outputMode")) outputMode = json.get("outputMode").getAsInt();
    }
}
