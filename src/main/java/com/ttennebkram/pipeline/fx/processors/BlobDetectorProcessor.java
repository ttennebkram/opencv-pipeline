package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Slider;
import javafx.scene.control.Spinner;
import org.opencv.core.*;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.SimpleBlobDetector;
import org.opencv.features2d.SimpleBlobDetector_Params;

/**
 * Simple Blob Detector processor.
 * Detects blobs based on various filter criteria.
 */
@FXProcessorInfo(
    nodeType = "BlobDetector",
    displayName = "Blob Detector",
    category = "Detection",
    description = "Blob detection\nSimpleBlobDetector.detect(image, keypoints)"
)
public class BlobDetectorProcessor extends FXProcessorBase {

    // Properties with defaults
    private int minThreshold = 10;
    private int maxThreshold = 200;
    private boolean showOriginal = true;
    private boolean filterByArea = true;
    private int minArea = 100;
    private int maxArea = 5000;
    private boolean filterByCircularity = false;
    private int minCircularity = 10;
    private boolean filterByConvexity = false;
    private int minConvexity = 87;
    private boolean filterByInertia = false;
    private int minInertiaRatio = 1;
    private boolean filterByColor = false;
    private int blobColor = 0;

    private static final String[] COLOR_OPTIONS = {"Dark (0)", "Light (255)"};

    @Override
    public String getNodeType() {
        return "BlobDetector";
    }

    @Override
    public String getCategory() {
        return "Feature Detection";
    }

    @Override
    public String getDescription() {
        return "Simple Blob Detector\nSimpleBlobDetector.create(params).detect(image, keypoints)";
    }

    @Override
    public Mat process(Mat input) {
        if (isInvalidInput(input)) {
            return input;
        }

        // Set up blob detector parameters
        SimpleBlobDetector_Params params = new SimpleBlobDetector_Params();
        params.set_minThreshold(minThreshold);
        params.set_maxThreshold(maxThreshold);
        params.set_filterByArea(filterByArea);
        params.set_minArea(minArea);
        params.set_maxArea(maxArea);
        params.set_filterByCircularity(filterByCircularity);
        params.set_minCircularity(minCircularity / 100.0f);
        params.set_filterByConvexity(filterByConvexity);
        params.set_minConvexity(minConvexity / 100.0f);
        params.set_filterByInertia(filterByInertia);
        params.set_minInertiaRatio(minInertiaRatio / 100.0f);
        // Note: filterByColor is enabled but blobColor value is not settable in Java bindings
        params.set_filterByColor(filterByColor);

        // Create detector
        SimpleBlobDetector detector = SimpleBlobDetector.create(params);

        // Detect blobs
        MatOfKeyPoint keypoints = new MatOfKeyPoint();
        detector.detect(input, keypoints);

        // Draw keypoints
        Mat output;
        if (showOriginal) {
            output = new Mat();
            Features2d.drawKeypoints(input, keypoints, output, new Scalar(0, 0, 255),
                Features2d.DrawMatchesFlags_DRAW_RICH_KEYPOINTS);
        } else {
            output = Mat.zeros(input.rows(), input.cols(), CvType.CV_8UC3);
            Features2d.drawKeypoints(output, keypoints, output, new Scalar(0, 0, 255),
                Features2d.DrawMatchesFlags_DRAW_RICH_KEYPOINTS);
        }

        keypoints.release();

        return output;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        CheckBox showOrigCheck = dialog.addCheckbox("Show Original Background", showOriginal);
        Slider minThreshSlider = dialog.addSlider("Min Threshold:", 0, 255, minThreshold, "%.0f");
        Slider maxThreshSlider = dialog.addSlider("Max Threshold:", 0, 255, maxThreshold, "%.0f");
        CheckBox areaCheck = dialog.addCheckbox("Filter by Area", filterByArea);
        Spinner<Integer> minAreaSpinner = dialog.addSpinner("Min Area:", 1, 10000, minArea);
        Spinner<Integer> maxAreaSpinner = dialog.addSpinner("Max Area:", 1, 50000, maxArea);
        CheckBox circCheck = dialog.addCheckbox("Filter by Circularity", filterByCircularity);
        Slider circSlider = dialog.addSlider("Min Circularity %:", 1, 100, minCircularity, "%.0f");
        CheckBox convCheck = dialog.addCheckbox("Filter by Convexity", filterByConvexity);
        Slider convSlider = dialog.addSlider("Min Convexity %:", 1, 100, minConvexity, "%.0f");
        CheckBox inertiaCheck = dialog.addCheckbox("Filter by Inertia", filterByInertia);
        Slider inertiaSlider = dialog.addSlider("Min Inertia %:", 1, 100, minInertiaRatio, "%.0f");
        CheckBox colorCheck = dialog.addCheckbox("Filter by Color", filterByColor);
        ComboBox<String> colorCombo = dialog.addComboBox("Blob Color:", COLOR_OPTIONS,
                COLOR_OPTIONS[blobColor == 0 ? 0 : 1]);

        // Save callback
        dialog.setOnOk(() -> {
            showOriginal = showOrigCheck.isSelected();
            minThreshold = (int) minThreshSlider.getValue();
            maxThreshold = (int) maxThreshSlider.getValue();
            filterByArea = areaCheck.isSelected();
            minArea = minAreaSpinner.getValue();
            maxArea = maxAreaSpinner.getValue();
            filterByCircularity = circCheck.isSelected();
            minCircularity = (int) circSlider.getValue();
            filterByConvexity = convCheck.isSelected();
            minConvexity = (int) convSlider.getValue();
            filterByInertia = inertiaCheck.isSelected();
            minInertiaRatio = (int) inertiaSlider.getValue();
            filterByColor = colorCheck.isSelected();
            blobColor = colorCombo.getSelectionModel().getSelectedIndex() == 0 ? 0 : 255;
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("minThreshold", minThreshold);
        json.addProperty("maxThreshold", maxThreshold);
        json.addProperty("showOriginal", showOriginal);
        json.addProperty("filterByArea", filterByArea);
        json.addProperty("minArea", minArea);
        json.addProperty("maxArea", maxArea);
        json.addProperty("filterByCircularity", filterByCircularity);
        json.addProperty("minCircularity", minCircularity);
        json.addProperty("filterByConvexity", filterByConvexity);
        json.addProperty("minConvexity", minConvexity);
        json.addProperty("filterByInertia", filterByInertia);
        json.addProperty("minInertiaRatio", minInertiaRatio);
        json.addProperty("filterByColor", filterByColor);
        json.addProperty("blobColor", blobColor);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        minThreshold = getJsonInt(json, "minThreshold", 10);
        maxThreshold = getJsonInt(json, "maxThreshold", 200);
        showOriginal = json.has("showOriginal") ? json.get("showOriginal").getAsBoolean() : true;
        filterByArea = json.has("filterByArea") ? json.get("filterByArea").getAsBoolean() : true;
        minArea = getJsonInt(json, "minArea", 100);
        maxArea = getJsonInt(json, "maxArea", 5000);
        filterByCircularity = json.has("filterByCircularity") ? json.get("filterByCircularity").getAsBoolean() : false;
        minCircularity = getJsonInt(json, "minCircularity", 10);
        filterByConvexity = json.has("filterByConvexity") ? json.get("filterByConvexity").getAsBoolean() : false;
        minConvexity = getJsonInt(json, "minConvexity", 87);
        filterByInertia = json.has("filterByInertia") ? json.get("filterByInertia").getAsBoolean() : false;
        minInertiaRatio = getJsonInt(json, "minInertiaRatio", 1);
        filterByColor = json.has("filterByColor") ? json.get("filterByColor").getAsBoolean() : false;
        blobColor = getJsonInt(json, "blobColor", 0);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        minThreshold = getInt(node.properties, "minThreshold", 10);
        maxThreshold = getInt(node.properties, "maxThreshold", 200);
        showOriginal = getBool(node.properties, "showOriginal", true);
        filterByArea = getBool(node.properties, "filterByArea", true);
        minArea = getInt(node.properties, "minArea", 100);
        maxArea = getInt(node.properties, "maxArea", 5000);
        filterByCircularity = getBool(node.properties, "filterByCircularity", false);
        minCircularity = getInt(node.properties, "minCircularity", 10);
        filterByConvexity = getBool(node.properties, "filterByConvexity", false);
        minConvexity = getInt(node.properties, "minConvexity", 87);
        filterByInertia = getBool(node.properties, "filterByInertia", false);
        minInertiaRatio = getInt(node.properties, "minInertiaRatio", 1);
        filterByColor = getBool(node.properties, "filterByColor", false);
        blobColor = getInt(node.properties, "blobColor", 0);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("minThreshold", minThreshold);
        node.properties.put("maxThreshold", maxThreshold);
        node.properties.put("showOriginal", showOriginal);
        node.properties.put("filterByArea", filterByArea);
        node.properties.put("minArea", minArea);
        node.properties.put("maxArea", maxArea);
        node.properties.put("filterByCircularity", filterByCircularity);
        node.properties.put("minCircularity", minCircularity);
        node.properties.put("filterByConvexity", filterByConvexity);
        node.properties.put("minConvexity", minConvexity);
        node.properties.put("filterByInertia", filterByInertia);
        node.properties.put("minInertiaRatio", minInertiaRatio);
        node.properties.put("filterByColor", filterByColor);
        node.properties.put("blobColor", blobColor);
    }

    private boolean getBool(java.util.Map<String, Object> props, String key, boolean defaultValue) {
        if (props.containsKey(key)) {
            Object val = props.get(key);
            if (val instanceof Boolean) return (Boolean) val;
        }
        return defaultValue;
    }
}
