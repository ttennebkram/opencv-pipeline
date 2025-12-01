package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.Slider;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

/**
 * Median Blur processor.
 * Applies a median filter - effective for salt-and-pepper noise removal.
 */
@FXProcessorInfo(nodeType = "MedianBlur", category = "Blur")
public class MedianBlurProcessor extends FXProcessorBase {

    // Properties with defaults
    private int kernelSize = 5;

    @Override
    public String getNodeType() {
        return "MedianBlur";
    }

    @Override
    public String getCategory() {
        return "Blur";
    }

    @Override
    public String getDescription() {
        return "Median Blur\nImgproc.medianBlur(src, dst, ksize)";
    }

    @Override
    public Mat process(Mat input) {
        if (isInvalidInput(input)) {
            return input;
        }

        // Kernel size must be positive odd number
        int ksize = Math.max(1, kernelSize);
        if (ksize % 2 == 0) ksize++;

        Mat output = new Mat();
        Imgproc.medianBlur(input, output, ksize);
        return output;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        // Kernel size slider (odd values only)
        Slider ksizeSlider = dialog.addOddKernelSlider("Kernel Size:", kernelSize, 255);

        // Save callback
        dialog.setOnOk(() -> {
            kernelSize = (int) ksizeSlider.getValue();
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("ksize", kernelSize);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        kernelSize = getJsonInt(json, "ksize", 5);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        kernelSize = getInt(node.properties, "ksize", 5);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("ksize", kernelSize);
    }
}
