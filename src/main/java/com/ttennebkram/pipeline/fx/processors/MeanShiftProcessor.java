package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.Slider;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

/**
 * Mean Shift Filtering processor.
 * Applies mean shift filtering for image segmentation/smoothing.
 */
@FXProcessorInfo(nodeType = "MeanShift", category = "Filter")
public class MeanShiftProcessor extends FXProcessorBase {

    // Properties with defaults
    private int spatialRadius = 20;
    private int colorRadius = 40;
    private int maxLevel = 1;

    @Override
    public String getNodeType() {
        return "MeanShift";
    }

    @Override
    public String getCategory() {
        return "Filter";
    }

    @Override
    public String getDescription() {
        return "Mean Shift Filtering\nImgproc.pyrMeanShiftFiltering(src, dst, sp, sr, maxLevel)";
    }

    @Override
    public Mat process(Mat input) {
        if (isInvalidInput(input)) {
            return input;
        }

        // pyrMeanShiftFiltering requires 3-channel (BGR) image
        Mat output = new Mat();
        if (input.channels() == 3) {
            Imgproc.pyrMeanShiftFiltering(input, output, spatialRadius, colorRadius, maxLevel);
        } else {
            // For grayscale, convert to BGR, process, then back
            Mat bgr = new Mat();
            Imgproc.cvtColor(input, bgr, Imgproc.COLOR_GRAY2BGR);
            Imgproc.pyrMeanShiftFiltering(bgr, output, spatialRadius, colorRadius, maxLevel);
            bgr.release();
            Mat gray = new Mat();
            Imgproc.cvtColor(output, gray, Imgproc.COLOR_BGR2GRAY);
            output.release();
            output = gray;
        }
        return output;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        Slider spatialSlider = dialog.addSlider("Spatial Radius:", 1, 100, spatialRadius, "%.0f");
        Slider colorSlider = dialog.addSlider("Color Radius:", 1, 100, colorRadius, "%.0f");
        Slider levelSlider = dialog.addSlider("Max Pyramid Level:", 0, 4, maxLevel, "%.0f");

        // Save callback
        dialog.setOnOk(() -> {
            spatialRadius = (int) spatialSlider.getValue();
            colorRadius = (int) colorSlider.getValue();
            maxLevel = (int) levelSlider.getValue();
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("spatialRadius", spatialRadius);
        json.addProperty("colorRadius", colorRadius);
        json.addProperty("maxLevel", maxLevel);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        spatialRadius = getJsonInt(json, "spatialRadius", 20);
        colorRadius = getJsonInt(json, "colorRadius", 40);
        maxLevel = getJsonInt(json, "maxLevel", 1);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        spatialRadius = getInt(node.properties, "spatialRadius", 20);
        colorRadius = getInt(node.properties, "colorRadius", 40);
        maxLevel = getInt(node.properties, "maxLevel", 1);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("spatialRadius", spatialRadius);
        node.properties.put("colorRadius", colorRadius);
        node.properties.put("maxLevel", maxLevel);
    }
}
