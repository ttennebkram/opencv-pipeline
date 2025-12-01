package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Slider;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

/**
 * Adaptive Threshold processor.
 * Applies adaptive thresholding where the threshold value varies across the image.
 */
@FXProcessorInfo(nodeType = "AdaptiveThreshold", category = "Threshold")
public class AdaptiveThresholdProcessor extends FXProcessorBase {

    // Properties with defaults
    private int maxValue = 255;
    private int methodIndex = 1;  // 0=Mean, 1=Gaussian
    private int typeIndex = 0;    // 0=Binary, 1=Binary Inv
    private int blockSize = 11;
    private int cValue = 2;

    private static final String[] METHODS = {"Mean", "Gaussian"};
    private static final int[] ADAPTIVE_METHODS = {
        Imgproc.ADAPTIVE_THRESH_MEAN_C,
        Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C
    };
    private static final String[] TYPES = {"Binary", "Binary Inv"};
    private static final int[] THRESHOLD_TYPES = {
        Imgproc.THRESH_BINARY,
        Imgproc.THRESH_BINARY_INV
    };

    @Override
    public String getNodeType() {
        return "AdaptiveThreshold";
    }

    @Override
    public String getCategory() {
        return "Threshold";
    }

    @Override
    public String getDescription() {
        return "Adaptive Threshold\nImgproc.adaptiveThreshold(src, dst, maxValue, method, type, blockSize, C)";
    }

    @Override
    public Mat process(Mat input) {
        if (isInvalidInput(input)) {
            return input;
        }

        // Convert to grayscale if needed
        Mat gray = new Mat();
        if (input.channels() == 3) {
            Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
        } else {
            gray = input.clone();
        }

        // Ensure blockSize is odd and >= 3
        int bs = blockSize;
        if (bs % 2 == 0) bs++;
        if (bs < 3) bs = 3;

        int method = ADAPTIVE_METHODS[Math.min(methodIndex, ADAPTIVE_METHODS.length - 1)];
        int type = THRESHOLD_TYPES[Math.min(typeIndex, THRESHOLD_TYPES.length - 1)];

        Mat output = new Mat();
        Imgproc.adaptiveThreshold(gray, output, maxValue, method, type, bs, cValue);

        gray.release();

        // Convert back to BGR for display
        Mat bgr = new Mat();
        Imgproc.cvtColor(output, bgr, Imgproc.COLOR_GRAY2BGR);
        output.release();

        return bgr;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        Slider maxValueSlider = dialog.addSlider("Max Value:", 0, 255, maxValue, "%.0f");
        ComboBox<String> methodCombo = dialog.addComboBox("Method:", METHODS,
                METHODS[Math.min(methodIndex, METHODS.length - 1)]);
        ComboBox<String> typeCombo = dialog.addComboBox("Type:", TYPES,
                TYPES[Math.min(typeIndex, TYPES.length - 1)]);
        Slider blockSizeSlider = dialog.addSlider("Block Size:", 3, 99, blockSize, "%.0f");
        Slider cValueSlider = dialog.addSlider("C Value:", 0, 50, cValue, "%.0f");

        // Save callback
        dialog.setOnOk(() -> {
            maxValue = (int) maxValueSlider.getValue();
            methodIndex = methodCombo.getSelectionModel().getSelectedIndex();
            typeIndex = typeCombo.getSelectionModel().getSelectedIndex();
            int bs = (int) blockSizeSlider.getValue();
            if (bs % 2 == 0) bs++;
            if (bs < 3) bs = 3;
            blockSize = bs;
            cValue = (int) cValueSlider.getValue();
        });
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
        maxValue = getJsonInt(json, "maxValue", 255);
        methodIndex = getJsonInt(json, "methodIndex", 1);
        typeIndex = getJsonInt(json, "typeIndex", 0);
        blockSize = getJsonInt(json, "blockSize", 11);
        cValue = getJsonInt(json, "cValue", 2);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        maxValue = getInt(node.properties, "maxValue", 255);
        methodIndex = getInt(node.properties, "methodIndex", 1);
        typeIndex = getInt(node.properties, "typeIndex", 0);
        blockSize = getInt(node.properties, "blockSize", 11);
        cValue = getInt(node.properties, "cValue", 2);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("maxValue", maxValue);
        node.properties.put("methodIndex", methodIndex);
        node.properties.put("typeIndex", typeIndex);
        node.properties.put("blockSize", blockSize);
        node.properties.put("cValue", cValue);
    }
}
