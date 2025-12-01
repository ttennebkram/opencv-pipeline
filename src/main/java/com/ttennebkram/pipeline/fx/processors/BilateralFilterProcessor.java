package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.Slider;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

/**
 * Bilateral Filter processor.
 * Edge-preserving smoothing filter - reduces noise while keeping edges sharp.
 */
@FXProcessorInfo(nodeType = "BilateralFilter", category = "Blur")
public class BilateralFilterProcessor extends FXProcessorBase {

    // Properties with defaults
    private int diameter = 9;
    private int sigmaColor = 75;
    private int sigmaSpace = 75;

    @Override
    public String getNodeType() {
        return "BilateralFilter";
    }

    @Override
    public String getCategory() {
        return "Blur";
    }

    @Override
    public String getDescription() {
        return "Bilateral Filter (Edge-Preserving)\nImgproc.bilateralFilter(src, dst, d, sigmaColor, sigmaSpace)";
    }

    @Override
    public Mat process(Mat input) {
        if (isInvalidInput(input)) {
            return input;
        }

        Mat output = new Mat();
        Imgproc.bilateralFilter(input, output, diameter, sigmaColor, sigmaSpace);
        return output;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        Slider diameterSlider = dialog.addSlider("Diameter:", 1, 25, diameter, "%.0f");
        Slider sigmaColorSlider = dialog.addSlider("Sigma Color:", 1, 200, sigmaColor, "%.0f");
        Slider sigmaSpaceSlider = dialog.addSlider("Sigma Space:", 1, 200, sigmaSpace, "%.0f");

        // Save callback
        dialog.setOnOk(() -> {
            diameter = (int) diameterSlider.getValue();
            sigmaColor = (int) sigmaColorSlider.getValue();
            sigmaSpace = (int) sigmaSpaceSlider.getValue();
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("diameter", diameter);
        json.addProperty("sigmaColor", sigmaColor);
        json.addProperty("sigmaSpace", sigmaSpace);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        diameter = getJsonInt(json, "diameter", 9);
        sigmaColor = getJsonInt(json, "sigmaColor", 75);
        sigmaSpace = getJsonInt(json, "sigmaSpace", 75);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        diameter = getInt(node.properties, "diameter", 9);
        sigmaColor = getInt(node.properties, "sigmaColor", 75);
        sigmaSpace = getInt(node.properties, "sigmaSpace", 75);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("diameter", diameter);
        node.properties.put("sigmaColor", sigmaColor);
        node.properties.put("sigmaSpace", sigmaSpace);
    }
}
