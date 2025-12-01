package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Slider;
import javafx.scene.control.Spinner;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

/**
 * Shi-Tomasi corner detection processor.
 * Detects corners using the Shi-Tomasi method (Good Features to Track).
 */
@FXProcessorInfo(nodeType = "ShiTomasi", category = "Feature Detection")
public class ShiTomasiProcessor extends FXProcessorBase {

    // Properties with defaults
    private int maxCorners = 100;
    private int qualityLevel = 1;  // 0.01 * 100
    private int minDistance = 10;
    private int blockSize = 3;
    private boolean useHarrisDetector = false;
    private int kPercent = 4;  // 0.04 * 100
    private int markerSize = 5;
    private boolean drawFeatures = true;

    @Override
    public String getNodeType() {
        return "ShiTomasi";
    }

    @Override
    public String getCategory() {
        return "Feature Detection";
    }

    @Override
    public String getDescription() {
        return "Shi-Tomasi Corner Detection\nImgproc.goodFeaturesToTrack(gray, corners, maxCorners, qualityLevel, minDistance)";
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

        // Detect corners
        MatOfPoint corners = new MatOfPoint();
        double quality = qualityLevel / 100.0;
        double k = kPercent / 100.0;

        Imgproc.goodFeaturesToTrack(gray, corners, maxCorners, quality, minDistance,
            new Mat(), blockSize, useHarrisDetector, k);

        // Prepare output
        Mat output = input.clone();

        if (drawFeatures) {
            Point[] cornerArray = corners.toArray();
            for (Point corner : cornerArray) {
                Imgproc.circle(output, corner, markerSize, new Scalar(0, 255, 0), -1);
            }
        }

        gray.release();
        corners.release();

        return output;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        Spinner<Integer> cornersSpinner = dialog.addSpinner("Max Corners:", 1, 1000, maxCorners);
        Slider qualitySlider = dialog.addSlider("Quality Level (x100):", 1, 100, qualityLevel, "%.0f");
        Spinner<Integer> minDistSpinner = dialog.addSpinner("Min Distance:", 1, 100, minDistance);
        Spinner<Integer> blockSpinner = dialog.addSpinner("Block Size:", 2, 31, blockSize);
        CheckBox harrisCheck = dialog.addCheckbox("Use Harris Detector", useHarrisDetector);
        Slider kSlider = dialog.addSlider("K Parameter (%):", 1, 20, kPercent, "%.0f");
        Spinner<Integer> markerSpinner = dialog.addSpinner("Marker Size:", 1, 20, markerSize);
        CheckBox drawCheck = dialog.addCheckbox("Draw Features", drawFeatures);

        // Save callback
        dialog.setOnOk(() -> {
            maxCorners = cornersSpinner.getValue();
            qualityLevel = (int) qualitySlider.getValue();
            minDistance = minDistSpinner.getValue();
            blockSize = blockSpinner.getValue();
            useHarrisDetector = harrisCheck.isSelected();
            kPercent = (int) kSlider.getValue();
            markerSize = markerSpinner.getValue();
            drawFeatures = drawCheck.isSelected();
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("maxCorners", maxCorners);
        json.addProperty("qualityLevel", qualityLevel);
        json.addProperty("minDistance", minDistance);
        json.addProperty("blockSize", blockSize);
        json.addProperty("useHarrisDetector", useHarrisDetector);
        json.addProperty("kPercent", kPercent);
        json.addProperty("markerSize", markerSize);
        json.addProperty("drawFeatures", drawFeatures);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        maxCorners = getJsonInt(json, "maxCorners", 100);
        qualityLevel = getJsonInt(json, "qualityLevel", 1);
        minDistance = getJsonInt(json, "minDistance", 10);
        blockSize = getJsonInt(json, "blockSize", 3);
        useHarrisDetector = json.has("useHarrisDetector") ? json.get("useHarrisDetector").getAsBoolean() : false;
        kPercent = getJsonInt(json, "kPercent", 4);
        markerSize = getJsonInt(json, "markerSize", 5);
        drawFeatures = json.has("drawFeatures") ? json.get("drawFeatures").getAsBoolean() : true;
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        maxCorners = getInt(node.properties, "maxCorners", 100);
        qualityLevel = getInt(node.properties, "qualityLevel", 1);
        minDistance = getInt(node.properties, "minDistance", 10);
        blockSize = getInt(node.properties, "blockSize", 3);
        useHarrisDetector = getBool(node.properties, "useHarrisDetector", false);
        kPercent = getInt(node.properties, "kPercent", 4);
        markerSize = getInt(node.properties, "markerSize", 5);
        drawFeatures = getBool(node.properties, "drawFeatures", true);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("maxCorners", maxCorners);
        node.properties.put("qualityLevel", qualityLevel);
        node.properties.put("minDistance", minDistance);
        node.properties.put("blockSize", blockSize);
        node.properties.put("useHarrisDetector", useHarrisDetector);
        node.properties.put("kPercent", kPercent);
        node.properties.put("markerSize", markerSize);
        node.properties.put("drawFeatures", drawFeatures);
    }

    private boolean getBool(java.util.Map<String, Object> props, String key, boolean defaultValue) {
        if (props.containsKey(key)) {
            Object val = props.get(key);
            if (val instanceof Boolean) return (Boolean) val;
        }
        return defaultValue;
    }
}
