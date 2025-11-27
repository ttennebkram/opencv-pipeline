package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.registry.NodeInfo;
import org.eclipse.swt.SWT;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

/**
 * Adaptive Threshold node.
 */
@NodeInfo(
    name = "AdaptiveThreshold",
    category = "Basic",
    aliases = {"Adaptive Threshold"}
)
public class AdaptiveThresholdNode extends ProcessingNode {
    private static final String[] ADAPTIVE_METHODS = {"Mean", "Gaussian"};
    private static final int[] ADAPTIVE_VALUES = {Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C};
    private static final String[] THRESH_TYPES = {"Binary", "Binary Inv"};
    private static final int[] THRESH_VALUES = {Imgproc.THRESH_BINARY, Imgproc.THRESH_BINARY_INV};

    private int maxValue = 255;
    private int methodIndex = 1; // Gaussian
    private int typeIndex = 0; // Binary
    private int blockSize = 11;
    private int cValue = 2;

    public AdaptiveThresholdNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Adaptive Threshold", x, y);
    }

    // Getters/setters for serialization
    public int getMaxValue() { return maxValue; }
    public void setMaxValue(int v) { maxValue = v; }
    public int getMethodIndex() { return methodIndex; }
    public void setMethodIndex(int v) { methodIndex = v; }
    public int getTypeIndex() { return typeIndex; }
    public void setTypeIndex(int v) { typeIndex = v; }
    public int getBlockSize() { return blockSize; }
    public void setBlockSize(int v) { blockSize = v; }
    public int getCValue() { return cValue; }
    public void setCValue(int v) { cValue = v; }

    @Override
    public Mat process(Mat input) {
        if (!enabled || input == null || input.empty()) {
            return input;
        }

        // Convert to grayscale if needed
        Mat gray;
        if (input.channels() == 3) {
            gray = new Mat();
            Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
        } else {
            gray = input;
        }

        // Ensure block size is odd
        int bsize = (blockSize % 2 == 0) ? blockSize + 1 : blockSize;

        Mat thresh = new Mat();
        Imgproc.adaptiveThreshold(gray, thresh, maxValue,
            ADAPTIVE_VALUES[methodIndex], THRESH_VALUES[typeIndex], bsize, cValue);

        // Convert back to BGR for display
        Mat output = new Mat();
        Imgproc.cvtColor(thresh, output, Imgproc.COLOR_GRAY2BGR);

        if (gray != input) {
            gray.release();
        }
        thresh.release();

        return output;
    }

    @Override
    public String getDescription() {
        return "Adaptive Threshold\ncv2.adaptiveThreshold(src, maxValue, method, type, blockSize, C)";
    }

    @Override
    public String getDisplayName() {
        return "Adaptive Threshold";
    }

    @Override
    public String getCategory() {
        return "Basic";
    }

    @Override
    protected int getPropertiesDialogColumns() {
        return 3;
    }

    @Override
    protected Runnable addPropertiesContent(Shell dialog, int columns) {
        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText(getDescription());
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sigGd.horizontalSpan = columns;
        sigLabel.setLayoutData(sigGd);

        Label sep = new Label(dialog, SWT.SEPARATOR | SWT.HORIZONTAL);
        GridData sepGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sepGd.horizontalSpan = columns;
        sep.setLayoutData(sepGd);

        // Max Value
        new Label(dialog, SWT.NONE).setText("Max Value:");
        Scale maxScale = new Scale(dialog, SWT.HORIZONTAL);
        maxScale.setMinimum(0);
        maxScale.setMaximum(255);
        maxScale.setSelection(maxValue);
        maxScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label maxLabel = new Label(dialog, SWT.NONE);
        maxLabel.setText(String.valueOf(maxValue));
        maxScale.addListener(SWT.Selection, e -> maxLabel.setText(String.valueOf(maxScale.getSelection())));

        // Adaptive Method
        new Label(dialog, SWT.NONE).setText("Adaptive Method:");
        Combo methodCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        methodCombo.setItems(ADAPTIVE_METHODS);
        methodCombo.select(methodIndex);
        GridData comboGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
        comboGd.horizontalSpan = 2;
        methodCombo.setLayoutData(comboGd);

        // Threshold Type
        new Label(dialog, SWT.NONE).setText("Threshold Type:");
        Combo typeCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        typeCombo.setItems(THRESH_TYPES);
        typeCombo.select(typeIndex);
        GridData typeGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
        typeGd.horizontalSpan = 2;
        typeCombo.setLayoutData(typeGd);

        // Block Size
        new Label(dialog, SWT.NONE).setText("Block Size:");
        Scale blockScale = new Scale(dialog, SWT.HORIZONTAL);
        blockScale.setMinimum(3);
        blockScale.setMaximum(99);
        blockScale.setSelection(blockSize);
        blockScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label blockLabel = new Label(dialog, SWT.NONE);
        blockLabel.setText(String.valueOf(blockSize));
        blockScale.addListener(SWT.Selection, e -> blockLabel.setText(String.valueOf(blockScale.getSelection())));

        // C Value
        new Label(dialog, SWT.NONE).setText("C (constant):");
        Scale cScale = new Scale(dialog, SWT.HORIZONTAL);
        cScale.setMinimum(0);
        cScale.setMaximum(50);
        cScale.setSelection(cValue);
        cScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label cLabel = new Label(dialog, SWT.NONE);
        cLabel.setText(String.valueOf(cValue));
        cScale.addListener(SWT.Selection, e -> cLabel.setText(String.valueOf(cScale.getSelection())));

        return () -> {
            maxValue = maxScale.getSelection();
            methodIndex = methodCombo.getSelectionIndex();
            typeIndex = typeCombo.getSelectionIndex();
            blockSize = blockScale.getSelection();
            cValue = cScale.getSelection();
        };
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("maxValue", maxValue);
        json.addProperty("methodIndex", methodIndex);
        json.addProperty("typeIndex", typeIndex);
        json.addProperty("blockSize", blockSize);
        json.addProperty("cValue", cValue);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        if (json.has("maxValue")) {
            maxValue = json.get("maxValue").getAsInt();
        }
        if (json.has("methodIndex")) {
            methodIndex = json.get("methodIndex").getAsInt();
        }
        if (json.has("typeIndex")) {
            typeIndex = json.get("typeIndex").getAsInt();
        }
        if (json.has("blockSize")) {
            blockSize = json.get("blockSize").getAsInt();
        }
        if (json.has("cValue")) {
            cValue = json.get("cValue").getAsInt();
        }
    }
}
