package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.registry.NodeInfo;
import org.eclipse.swt.SWT;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Mat;

/**
 * Gain effect node.
 */
@NodeInfo(
    name = "Gain",
    category = "Basic",
    aliases = {}
)
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
    protected Runnable addPropertiesContent(Shell dialog, int columns) {
        // Method signature
        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText(getDescription());
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sigGd.horizontalSpan = columns;
        sigLabel.setLayoutData(sigGd);

        // Separator
        Label sep = new Label(dialog, SWT.SEPARATOR | SWT.HORIZONTAL);
        GridData sepGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sepGd.horizontalSpan = columns;
        sep.setLayoutData(sepGd);

        new Label(dialog, SWT.NONE).setText("Gain (0.1x - 10x):");

        // Composite to hold scale and value label on same row
        Composite scaleComp = new Composite(dialog, SWT.NONE);
        scaleComp.setLayout(new GridLayout(2, false));
        scaleComp.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        Scale gainScale = new Scale(scaleComp, SWT.HORIZONTAL);
        gainScale.setMinimum(1);
        gainScale.setMaximum(100);
        // Use logarithmic mapping: scale value = log10(gain) * 50 + 50
        gainScale.setSelection((int)(Math.log10(gain) * 50 + 50));
        gainScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label gainLabel = new Label(scaleComp, SWT.NONE);
        gainLabel.setText(String.format("%.2fx", gain));
        GridData labelGd = new GridData(SWT.LEFT, SWT.CENTER, false, false);
        labelGd.widthHint = 50;
        gainLabel.setLayoutData(labelGd);

        gainScale.addListener(SWT.Selection, e -> {
            double logVal = (gainScale.getSelection() - 50) / 50.0;
            double g = Math.pow(10, logVal);
            gainLabel.setText(String.format("%.2fx", g));
        });

        return () -> {
            double logVal = (gainScale.getSelection() - 50) / 50.0;
            gain = Math.pow(10, logVal);
        };
    }

    @Override
    public void serializeProperties(JsonObject json) {
        super.serializeProperties(json);
        json.addProperty("gain", gain);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        super.deserializeProperties(json);
        if (json.has("gain")) {
            gain = json.get("gain").getAsDouble();
        }
    }
}
