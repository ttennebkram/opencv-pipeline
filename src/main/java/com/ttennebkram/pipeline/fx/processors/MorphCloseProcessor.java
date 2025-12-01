package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.Slider;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Morphological Close processor.
 * Closing = dilation followed by erosion. Useful for closing small holes.
 */
@FXProcessorInfo(nodeType = "MorphClose", category = "Morphology")
public class MorphCloseProcessor extends FXProcessorBase {

    // Properties with defaults
    private int kernelSize = 5;

    @Override
    public String getNodeType() {
        return "MorphClose";
    }

    @Override
    public String getCategory() {
        return "Morphology";
    }

    @Override
    public String getDescription() {
        return "Morphological Close\nImgproc.morphologyEx(src, dst, MORPH_CLOSE, kernel)";
    }

    @Override
    public Mat process(Mat input) {
        if (isInvalidInput(input)) {
            return input;
        }

        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,
            new Size(kernelSize, kernelSize));

        Mat output = new Mat();
        Imgproc.morphologyEx(input, output, Imgproc.MORPH_CLOSE, kernel);

        kernel.release();
        return output;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        Slider ksizeSlider = dialog.addSlider("Kernel Size:", 1, 21, kernelSize, "%.0f");

        // Save callback
        dialog.setOnOk(() -> {
            kernelSize = (int) ksizeSlider.getValue();
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("kernelSize", kernelSize);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        kernelSize = getJsonInt(json, "kernelSize", 5);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        kernelSize = getInt(node.properties, "kernelSize", 5);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("kernelSize", kernelSize);
    }
}
