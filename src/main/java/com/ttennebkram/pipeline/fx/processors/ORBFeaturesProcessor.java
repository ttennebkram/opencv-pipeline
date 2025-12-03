package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Spinner;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.ORB;

/**
 * ORB (Oriented FAST and Rotated BRIEF) feature detection processor.
 * Detects and visualizes ORB keypoints.
 */
@FXProcessorInfo(
    nodeType = "ORBFeatures",
    displayName = "ORB Features",
    category = "Detection",
    description = "ORB feature detection\nORB.detectAndCompute(image, mask, keypoints, descriptors)"
)
public class ORBFeaturesProcessor extends FXProcessorBase {

    // Properties with defaults
    private int nFeatures = 500;
    private int fastThreshold = 20;
    private int nLevels = 8;
    private boolean showRich = true;

    @Override
    public String getNodeType() {
        return "ORBFeatures";
    }

    @Override
    public String getCategory() {
        return "Feature Detection";
    }

    @Override
    public String getDescription() {
        return "ORB Feature Detection\nORB.create(nFeatures, ...).detect(image, keypoints)";
    }

    @Override
    public Mat process(Mat input) {
        if (isInvalidInput(input)) {
            return input;
        }

        // Create ORB detector
        ORB orb = ORB.create(nFeatures, 1.2f, nLevels, 31, 0, 2, ORB.HARRIS_SCORE, 31, fastThreshold);

        // Detect keypoints
        MatOfKeyPoint keypoints = new MatOfKeyPoint();
        orb.detect(input, keypoints);

        // Draw keypoints
        Mat output = new Mat();
        int flags = showRich ? Features2d.DrawMatchesFlags_DRAW_RICH_KEYPOINTS : Features2d.DrawMatchesFlags_DEFAULT;
        Features2d.drawKeypoints(input, keypoints, output, new Scalar(0, 255, 0), flags);

        keypoints.release();

        return output;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        Spinner<Integer> featSpinner = dialog.addSpinner("Max Features:", 10, 5000, nFeatures);
        Spinner<Integer> threshSpinner = dialog.addSpinner("FAST Threshold:", 1, 100, fastThreshold);
        Spinner<Integer> levelSpinner = dialog.addSpinner("Pyramid Levels:", 1, 16, nLevels);
        CheckBox richCheck = dialog.addCheckbox("Show Rich Keypoints", showRich);

        // Save callback
        dialog.setOnOk(() -> {
            nFeatures = featSpinner.getValue();
            fastThreshold = threshSpinner.getValue();
            nLevels = levelSpinner.getValue();
            showRich = richCheck.isSelected();
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("nFeatures", nFeatures);
        json.addProperty("fastThreshold", fastThreshold);
        json.addProperty("nLevels", nLevels);
        json.addProperty("showRich", showRich);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        nFeatures = getJsonInt(json, "nFeatures", 500);
        fastThreshold = getJsonInt(json, "fastThreshold", 20);
        nLevels = getJsonInt(json, "nLevels", 8);
        showRich = json.has("showRich") ? json.get("showRich").getAsBoolean() : true;
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        nFeatures = getInt(node.properties, "nFeatures", 500);
        fastThreshold = getInt(node.properties, "fastThreshold", 20);
        nLevels = getInt(node.properties, "nLevels", 8);
        showRich = getBool(node.properties, "showRich", true);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("nFeatures", nFeatures);
        node.properties.put("fastThreshold", fastThreshold);
        node.properties.put("nLevels", nLevels);
        node.properties.put("showRich", showRich);
    }

    private boolean getBool(java.util.Map<String, Object> props, String key, boolean defaultValue) {
        if (props.containsKey(key)) {
            Object val = props.get(key);
            if (val instanceof Boolean) return (Boolean) val;
        }
        return defaultValue;
    }
}
