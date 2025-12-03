package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Slider;
import javafx.scene.control.Spinner;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Contours detection and drawing processor.
 * Finds and draws contours in the image.
 */
@FXProcessorInfo(
    nodeType = "Contours",
    displayName = "Contours",
    category = "Detection",
    description = "Contour detection\nImgproc.findContours(src, contours, hierarchy, mode, method)"
)
public class ContoursProcessor extends FXProcessorBase {

    // Properties with defaults
    private boolean showOriginal = false;
    private int thickness = 2;
    private int colorR = 0;
    private int colorG = 255;
    private int colorB = 0;

    @Override
    public String getNodeType() {
        return "Contours";
    }

    @Override
    public String getCategory() {
        return "Feature Detection";
    }

    @Override
    public String getDescription() {
        return "Contour Detection\nImgproc.findContours(gray, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE)";
    }

    @Override
    public Mat process(Mat input) {
        if (isInvalidInput(input)) {
            return input;
        }

        // Convert to grayscale
        Mat gray = new Mat();
        if (input.channels() == 3) {
            Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
        } else {
            gray = input.clone();
        }

        // Apply threshold
        Mat binary = new Mat();
        Imgproc.threshold(gray, binary, 127, 255, Imgproc.THRESH_BINARY);

        // Find contours
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(binary, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        // Prepare output
        Mat output;
        if (showOriginal) {
            output = input.clone();
        } else {
            output = Mat.zeros(input.rows(), input.cols(), CvType.CV_8UC3);
        }

        // Draw contours
        Scalar color = new Scalar(colorB, colorG, colorR);
        Imgproc.drawContours(output, contours, -1, color, thickness);

        gray.release();
        binary.release();
        hierarchy.release();
        for (MatOfPoint contour : contours) {
            contour.release();
        }

        return output;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        CheckBox showOrigCheckBox = dialog.addCheckbox("Show Original Image", showOriginal);
        Slider thicknessSlider = dialog.addSlider("Thickness:", 1, 10, thickness, "%.0f");
        Spinner<Integer> rSpinner = dialog.addSpinner("Color R:", 0, 255, colorR);
        Spinner<Integer> gSpinner = dialog.addSpinner("Color G:", 0, 255, colorG);
        Spinner<Integer> bSpinner = dialog.addSpinner("Color B:", 0, 255, colorB);

        // Save callback
        dialog.setOnOk(() -> {
            showOriginal = showOrigCheckBox.isSelected();
            thickness = (int) thicknessSlider.getValue();
            colorR = rSpinner.getValue();
            colorG = gSpinner.getValue();
            colorB = bSpinner.getValue();
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("showOriginal", showOriginal);
        json.addProperty("thickness", thickness);
        json.addProperty("colorR", colorR);
        json.addProperty("colorG", colorG);
        json.addProperty("colorB", colorB);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        showOriginal = json.has("showOriginal") ? json.get("showOriginal").getAsBoolean() : false;
        thickness = getJsonInt(json, "thickness", 2);
        colorR = getJsonInt(json, "colorR", 0);
        colorG = getJsonInt(json, "colorG", 255);
        colorB = getJsonInt(json, "colorB", 0);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        showOriginal = getBool(node.properties, "showOriginal", false);
        thickness = getInt(node.properties, "thickness", 2);
        colorR = getInt(node.properties, "colorR", 0);
        colorG = getInt(node.properties, "colorG", 255);
        colorB = getInt(node.properties, "colorB", 0);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("showOriginal", showOriginal);
        node.properties.put("thickness", thickness);
        node.properties.put("colorR", colorR);
        node.properties.put("colorG", colorG);
        node.properties.put("colorB", colorB);
    }

    private boolean getBool(java.util.Map<String, Object> props, String key, boolean defaultValue) {
        if (props.containsKey(key)) {
            Object val = props.get(key);
            if (val instanceof Boolean) return (Boolean) val;
        }
        return defaultValue;
    }
}
