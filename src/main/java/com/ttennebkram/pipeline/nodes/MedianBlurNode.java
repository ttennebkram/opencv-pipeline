package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.registry.NodeInfo;
import org.eclipse.swt.SWT;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

/**
 * Median Blur node.
 */
@NodeInfo(
    name = "MedianBlur",
    category = "Blur",
    aliases = {"Median Blur"}
)
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
    protected int getPropertiesDialogColumns() {
        return 3;
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

        // Kernel Size
        new Label(dialog, SWT.NONE).setText("Kernel Size:");
        Scale kScale = new Scale(dialog, SWT.HORIZONTAL);
        kScale.setMinimum(1);
        kScale.setMaximum(31);
        // Clamp slider position to valid range, but keep actual value
        int kSliderPos = Math.min(Math.max(kernelSize, 1), 31);
        kScale.setSelection(kSliderPos);
        kScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label kLabel = new Label(dialog, SWT.NONE);
        kLabel.setText(String.valueOf(kernelSize)); // Show real value
        kScale.addListener(SWT.Selection, e -> kLabel.setText(String.valueOf(kScale.getSelection())));

        return () -> {
            kernelSize = kScale.getSelection();
        };
    }

    @Override
    public void serializeProperties(JsonObject json) {
        super.serializeProperties(json);
        json.addProperty("kernelSize", kernelSize);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        super.deserializeProperties(json);
        if (json.has("kernelSize")) kernelSize = json.get("kernelSize").getAsInt();
    }
}
