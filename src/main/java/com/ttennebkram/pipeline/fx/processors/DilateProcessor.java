package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.Slider;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Dilate processor.
 * Expands the boundaries of foreground objects.
 */
@FXProcessorInfo(nodeType = "Dilate", category = "Morphology")
public class DilateProcessor extends FXProcessorBase {

    // Properties with defaults
    private int kernelSize = 5;
    private int iterations = 1;

    @Override
    public String getNodeType() {
        return "Dilate";
    }

    @Override
    public String getCategory() {
        return "Morphology";
    }

    @Override
    public String getDescription() {
        return "Dilate\nImgproc.dilate(src, dst, kernel, iterations)";
    }

    @Override
    public Mat process(Mat input) {
        if (isInvalidInput(input)) {
            return input;
        }

        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,
            new Size(kernelSize, kernelSize));

        Mat output = new Mat();
        Imgproc.dilate(input, output, kernel, new Point(-1, -1), iterations);

        kernel.release();
        return output;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        Slider ksizeSlider = dialog.addSlider("Kernel Size:", 1, 21, kernelSize, "%.0f");
        Slider iterSlider = dialog.addSlider("Iterations:", 1, 10, iterations, "%.0f");

        // Save callback
        dialog.setOnOk(() -> {
            kernelSize = (int) ksizeSlider.getValue();
            iterations = (int) iterSlider.getValue();
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("kernelSize", kernelSize);
        json.addProperty("iterations", iterations);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        kernelSize = getJsonInt(json, "kernelSize", 5);
        iterations = getJsonInt(json, "iterations", 1);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        kernelSize = getInt(node.properties, "kernelSize", 5);
        iterations = getInt(node.properties, "iterations", 1);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("kernelSize", kernelSize);
        node.properties.put("iterations", iterations);
    }
}
