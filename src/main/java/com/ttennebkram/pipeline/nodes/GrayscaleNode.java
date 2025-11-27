package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.registry.NodeInfo;
import org.eclipse.swt.SWT;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

/**
 * Grayscale / Color Conversion node.
 */
@NodeInfo(
    name = "Grayscale",
    category = "Basic",
    aliases = {"Grayscale/Color Convert"}
)
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

        new Label(dialog, SWT.NONE).setText("Conversion:");
        Combo combo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        combo.setItems(CONVERSION_NAMES);
        combo.select(conversionIndex);

        return () -> {
            conversionIndex = combo.getSelectionIndex();
        };
    }

    @Override
    public void serializeProperties(JsonObject json) {
        super.serializeProperties(json);
        json.addProperty("conversionIndex", conversionIndex);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        super.deserializeProperties(json);
        if (json.has("conversionIndex")) {
            conversionIndex = json.get("conversionIndex").getAsInt();
        }
    }
}
