package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Slider;
import javafx.scene.control.Spinner;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

/**
 * Hough Circles detection processor.
 * Detects circles in the image using the Hough transform.
 */
@FXProcessorInfo(
    nodeType = "HoughCircles",
    displayName = "Hough Circles",
    category = "Detection",
    description = "Hough circle detection\nImgproc.HoughCircles(src, circles, method, dp, minDist, ...)"
)
public class HoughCirclesProcessor extends FXProcessorBase {

    // Properties with defaults
    private boolean showOriginal = true;
    private int minDist = 50;
    private int param1 = 100;
    private int param2 = 30;
    private int minRadius = 10;
    private int maxRadius = 100;
    private int thickness = 2;
    private int colorR = 0;
    private int colorG = 255;
    private int colorB = 0;

    @Override
    public String getNodeType() {
        return "HoughCircles";
    }

    @Override
    public String getCategory() {
        return "Feature Detection";
    }

    @Override
    public String getDescription() {
        return "Hough Circle Detection\nImgproc.HoughCircles(gray, circles, HOUGH_GRADIENT, 1, minDist, param1, param2, minRadius, maxRadius)";
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

        // Apply blur to reduce noise
        Imgproc.medianBlur(gray, gray, 5);

        // Detect circles
        Mat circles = new Mat();
        Imgproc.HoughCircles(gray, circles, Imgproc.HOUGH_GRADIENT, 1, minDist,
            param1, param2, minRadius, maxRadius);

        // Prepare output
        Mat output;
        if (showOriginal) {
            output = input.clone();
        } else {
            output = new Mat(input.rows(), input.cols(), CvType.CV_8UC3, new Scalar(0, 0, 0));
        }

        // Draw circles
        for (int i = 0; i < circles.cols(); i++) {
            double[] circle = circles.get(0, i);
            if (circle != null) {
                Point center = new Point(circle[0], circle[1]);
                int radius = (int) Math.round(circle[2]);
                // Draw circle
                Imgproc.circle(output, center, radius, new Scalar(colorB, colorG, colorR), thickness);
                // Draw center point
                Imgproc.circle(output, center, 3, new Scalar(colorB, colorG, colorR), -1);
            }
        }

        gray.release();
        circles.release();

        return output;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        CheckBox showOrigCheckBox = dialog.addCheckbox("Show Original Image", showOriginal);
        Slider minDistSlider = dialog.addSlider("Min Distance:", 10, 200, minDist, "%.0f");
        Slider param1Slider = dialog.addSlider("Param1 (Canny threshold):", 10, 300, param1, "%.0f");
        Slider param2Slider = dialog.addSlider("Param2 (accumulator threshold):", 10, 100, param2, "%.0f");
        Slider minRadiusSlider = dialog.addSlider("Min Radius:", 0, 200, minRadius, "%.0f");
        Slider maxRadiusSlider = dialog.addSlider("Max Radius:", 10, 500, maxRadius, "%.0f");
        Spinner<Integer> thicknessSpinner = dialog.addSpinner("Thickness:", 1, 10, thickness);
        Spinner<Integer> rSpinner = dialog.addSpinner("Color R:", 0, 255, colorR);
        Spinner<Integer> gSpinner = dialog.addSpinner("Color G:", 0, 255, colorG);
        Spinner<Integer> bSpinner = dialog.addSpinner("Color B:", 0, 255, colorB);

        // Save callback
        dialog.setOnOk(() -> {
            showOriginal = showOrigCheckBox.isSelected();
            minDist = (int) minDistSlider.getValue();
            param1 = (int) param1Slider.getValue();
            param2 = (int) param2Slider.getValue();
            minRadius = (int) minRadiusSlider.getValue();
            maxRadius = (int) maxRadiusSlider.getValue();
            thickness = thicknessSpinner.getValue();
            colorR = rSpinner.getValue();
            colorG = gSpinner.getValue();
            colorB = bSpinner.getValue();
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("showOriginal", showOriginal);
        json.addProperty("minDist", minDist);
        json.addProperty("param1", param1);
        json.addProperty("param2", param2);
        json.addProperty("minRadius", minRadius);
        json.addProperty("maxRadius", maxRadius);
        json.addProperty("thickness", thickness);
        json.addProperty("colorR", colorR);
        json.addProperty("colorG", colorG);
        json.addProperty("colorB", colorB);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        showOriginal = json.has("showOriginal") ? json.get("showOriginal").getAsBoolean() : true;
        minDist = getJsonInt(json, "minDist", 50);
        param1 = getJsonInt(json, "param1", 100);
        param2 = getJsonInt(json, "param2", 30);
        minRadius = getJsonInt(json, "minRadius", 10);
        maxRadius = getJsonInt(json, "maxRadius", 100);
        thickness = getJsonInt(json, "thickness", 2);
        colorR = getJsonInt(json, "colorR", 0);
        colorG = getJsonInt(json, "colorG", 255);
        colorB = getJsonInt(json, "colorB", 0);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        showOriginal = getBool(node.properties, "showOriginal", true);
        minDist = getInt(node.properties, "minDist", 50);
        param1 = getInt(node.properties, "param1", 100);
        param2 = getInt(node.properties, "param2", 30);
        minRadius = getInt(node.properties, "minRadius", 10);
        maxRadius = getInt(node.properties, "maxRadius", 100);
        thickness = getInt(node.properties, "thickness", 2);
        colorR = getInt(node.properties, "colorR", 0);
        colorG = getInt(node.properties, "colorG", 255);
        colorB = getInt(node.properties, "colorB", 0);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("showOriginal", showOriginal);
        node.properties.put("minDist", minDist);
        node.properties.put("param1", param1);
        node.properties.put("param2", param2);
        node.properties.put("minRadius", minRadius);
        node.properties.put("maxRadius", maxRadius);
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
