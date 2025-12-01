package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Slider;
import javafx.scene.control.Spinner;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.SIFT;

/**
 * SIFT (Scale-Invariant Feature Transform) feature detection processor.
 * Detects and visualizes SIFT keypoints.
 */
@FXProcessorInfo(nodeType = "SIFTFeatures", category = "Feature Detection")
public class SIFTFeaturesProcessor extends FXProcessorBase {

    // Properties with defaults
    private int nFeatures = 500;
    private int nOctaveLayers = 3;
    private int contrastThreshold = 4;  // 0.04 * 100
    private int edgeThreshold = 10;
    private int sigma = 16;  // 1.6 * 10
    private boolean showRichKeypoints = true;
    private int colorIndex = 0;

    private static final String[] COLORS = {"Green", "Red", "Blue", "Yellow", "White"};
    private static final Scalar[] COLOR_VALUES = {
        new Scalar(0, 255, 0),
        new Scalar(0, 0, 255),
        new Scalar(255, 0, 0),
        new Scalar(0, 255, 255),
        new Scalar(255, 255, 255)
    };

    @Override
    public String getNodeType() {
        return "SIFTFeatures";
    }

    @Override
    public String getCategory() {
        return "Feature Detection";
    }

    @Override
    public String getDescription() {
        return "SIFT Feature Detection\nSIFT.create(nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma)";
    }

    @Override
    public Mat process(Mat input) {
        if (isInvalidInput(input)) {
            return input;
        }

        // Create SIFT detector
        double contrast = contrastThreshold / 100.0;
        double sig = sigma / 10.0;
        SIFT sift = SIFT.create(nFeatures, nOctaveLayers, contrast, edgeThreshold, sig);

        // Detect keypoints
        MatOfKeyPoint keypoints = new MatOfKeyPoint();
        sift.detect(input, keypoints);

        // Draw keypoints
        Mat output = new Mat();
        int flags = showRichKeypoints ? Features2d.DrawMatchesFlags_DRAW_RICH_KEYPOINTS : Features2d.DrawMatchesFlags_DEFAULT;
        Scalar color = COLOR_VALUES[Math.min(colorIndex, COLOR_VALUES.length - 1)];
        Features2d.drawKeypoints(input, keypoints, output, color, flags);

        keypoints.release();

        return output;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        Spinner<Integer> featSpinner = dialog.addSpinner("Max Features (0=all):", 0, 5000, nFeatures);
        Spinner<Integer> octaveSpinner = dialog.addSpinner("Octave Layers:", 1, 10, nOctaveLayers);
        Slider contrastSlider = dialog.addSlider("Contrast Thresh (x100):", 1, 20, contrastThreshold, "%.0f");
        Spinner<Integer> edgeSpinner = dialog.addSpinner("Edge Threshold:", 1, 50, edgeThreshold);
        Slider sigmaSlider = dialog.addSlider("Sigma (x10):", 10, 30, sigma, "%.0f");
        CheckBox richCheck = dialog.addCheckbox("Show Rich Keypoints", showRichKeypoints);
        ComboBox<String> colorCombo = dialog.addComboBox("Keypoint Color:", COLORS,
                COLORS[Math.min(colorIndex, COLORS.length - 1)]);

        // Save callback
        dialog.setOnOk(() -> {
            nFeatures = featSpinner.getValue();
            nOctaveLayers = octaveSpinner.getValue();
            contrastThreshold = (int) contrastSlider.getValue();
            edgeThreshold = edgeSpinner.getValue();
            sigma = (int) sigmaSlider.getValue();
            showRichKeypoints = richCheck.isSelected();
            colorIndex = colorCombo.getSelectionModel().getSelectedIndex();
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("nFeatures", nFeatures);
        json.addProperty("nOctaveLayers", nOctaveLayers);
        json.addProperty("contrastThreshold", contrastThreshold);
        json.addProperty("edgeThreshold", edgeThreshold);
        json.addProperty("sigma", sigma);
        json.addProperty("showRichKeypoints", showRichKeypoints);
        json.addProperty("colorIndex", colorIndex);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        nFeatures = getJsonInt(json, "nFeatures", 500);
        nOctaveLayers = getJsonInt(json, "nOctaveLayers", 3);
        contrastThreshold = getJsonInt(json, "contrastThreshold", 4);
        edgeThreshold = getJsonInt(json, "edgeThreshold", 10);
        sigma = getJsonInt(json, "sigma", 16);
        showRichKeypoints = json.has("showRichKeypoints") ? json.get("showRichKeypoints").getAsBoolean() : true;
        colorIndex = getJsonInt(json, "colorIndex", 0);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        nFeatures = getInt(node.properties, "nFeatures", 500);
        nOctaveLayers = getInt(node.properties, "nOctaveLayers", 3);
        contrastThreshold = getInt(node.properties, "contrastThreshold", 4);
        edgeThreshold = getInt(node.properties, "edgeThreshold", 10);
        sigma = getInt(node.properties, "sigma", 16);
        showRichKeypoints = getBool(node.properties, "showRichKeypoints", true);
        colorIndex = getInt(node.properties, "colorIndex", 0);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("nFeatures", nFeatures);
        node.properties.put("nOctaveLayers", nOctaveLayers);
        node.properties.put("contrastThreshold", contrastThreshold);
        node.properties.put("edgeThreshold", edgeThreshold);
        node.properties.put("sigma", sigma);
        node.properties.put("showRichKeypoints", showRichKeypoints);
        node.properties.put("colorIndex", colorIndex);
    }

    private boolean getBool(java.util.Map<String, Object> props, String key, boolean defaultValue) {
        if (props.containsKey(key)) {
            Object val = props.get(key);
            if (val instanceof Boolean) return (Boolean) val;
        }
        return defaultValue;
    }
}
