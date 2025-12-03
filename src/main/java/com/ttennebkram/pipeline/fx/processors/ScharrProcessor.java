package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Slider;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

/**
 * Scharr edge detection processor.
 * Computes first-order derivatives using Scharr operator (more accurate than Sobel for 3x3).
 */
@FXProcessorInfo(
    nodeType = "Scharr",
    displayName = "Scharr Edges",
    buttonName = "Scharr",
    category = "Edges",
    description = "Scharr derivatives\nImgproc.Scharr(src, dst, ddepth, dx, dy)"
)
public class ScharrProcessor extends FXProcessorBase {

    // Properties with defaults
    private int directionIndex = 2;  // 0=X, 1=Y, 2=Both
    private int scalePercent = 100;
    private int delta = 0;

    private static final String[] DIRECTIONS = {"X", "Y", "Both"};

    @Override
    public String getNodeType() {
        return "Scharr";
    }

    @Override
    public String getCategory() {
        return "Edge Detection";
    }

    @Override
    public String getDescription() {
        return "Scharr Derivatives\nImgproc.Scharr(src, dst, ddepth, dx, dy, scale, delta)";
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

        double scale = scalePercent / 100.0;
        Mat output = new Mat();

        if (directionIndex == 0) {
            // X direction only
            Imgproc.Scharr(gray, output, CvType.CV_16S, 1, 0, scale, delta);
            Core.convertScaleAbs(output, output);
        } else if (directionIndex == 1) {
            // Y direction only
            Imgproc.Scharr(gray, output, CvType.CV_16S, 0, 1, scale, delta);
            Core.convertScaleAbs(output, output);
        } else {
            // Both directions - compute separately and combine
            Mat scharrX = new Mat();
            Mat scharrY = new Mat();
            Imgproc.Scharr(gray, scharrX, CvType.CV_16S, 1, 0, scale, delta);
            Imgproc.Scharr(gray, scharrY, CvType.CV_16S, 0, 1, scale, delta);
            Core.convertScaleAbs(scharrX, scharrX);
            Core.convertScaleAbs(scharrY, scharrY);
            Core.addWeighted(scharrX, 0.5, scharrY, 0.5, 0, output);
            scharrX.release();
            scharrY.release();
        }

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

        ComboBox<String> dirCombo = dialog.addComboBox("Direction:", DIRECTIONS,
                DIRECTIONS[Math.min(directionIndex, DIRECTIONS.length - 1)]);
        Slider scaleSlider = dialog.addSlider("Scale (%):", 10, 500, scalePercent, "%.0f%%");
        Slider deltaSlider = dialog.addSlider("Delta:", 0, 255, delta, "%.0f");

        // Save callback
        dialog.setOnOk(() -> {
            directionIndex = dirCombo.getSelectionModel().getSelectedIndex();
            scalePercent = (int) scaleSlider.getValue();
            delta = (int) deltaSlider.getValue();
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("directionIndex", directionIndex);
        json.addProperty("scalePercent", scalePercent);
        json.addProperty("delta", delta);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        directionIndex = getJsonInt(json, "directionIndex", 2);
        scalePercent = getJsonInt(json, "scalePercent", 100);
        delta = getJsonInt(json, "delta", 0);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        directionIndex = getInt(node.properties, "directionIndex", 2);
        scalePercent = getInt(node.properties, "scalePercent", 100);
        delta = getInt(node.properties, "delta", 0);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("directionIndex", directionIndex);
        node.properties.put("scalePercent", scalePercent);
        node.properties.put("delta", delta);
    }
}
