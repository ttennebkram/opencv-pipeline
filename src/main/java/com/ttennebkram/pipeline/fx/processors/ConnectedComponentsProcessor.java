package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Slider;
import javafx.scene.control.Spinner;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.util.Random;

/**
 * Connected Components labeling processor.
 * Labels connected regions with distinct colors.
 */
@FXProcessorInfo(
    nodeType = "ConnectedComponents",
    displayName = "Connected Components",
    category = "Detection",
    description = "Connected component labeling\nImgproc.connectedComponentsWithStats(...)"
)
public class ConnectedComponentsProcessor extends FXProcessorBase {

    // Properties with defaults
    private int threshold = 127;
    private boolean invertThreshold = false;
    private int connectivity = 8;
    private int minArea = 0;

    private static final String[] CONNECTIVITY_OPTIONS = {"4", "8"};

    @Override
    public String getNodeType() {
        return "ConnectedComponents";
    }

    @Override
    public String getCategory() {
        return "Feature Detection";
    }

    @Override
    public String getDescription() {
        return "Connected Components Labeling\nImgproc.connectedComponentsWithStats(image, labels, stats, centroids)";
    }

    @Override
    public Mat process(Mat input) {
        if (isInvalidInput(input)) {
            return input;
        }

        Mat gray = null;
        Mat binary = null;
        Mat labels = null;
        Mat stats = null;
        Mat centroids = null;
        Mat result = null;

        try {
            // Convert to grayscale
            gray = new Mat();
            if (input.channels() == 3) {
                Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
            } else {
                gray = input.clone();
            }

            // Apply threshold
            binary = new Mat();
            int threshType = invertThreshold ? Imgproc.THRESH_BINARY_INV : Imgproc.THRESH_BINARY;
            Imgproc.threshold(gray, binary, threshold, 255, threshType);

            // Get connected components with stats
            labels = new Mat();
            stats = new Mat();
            centroids = new Mat();
            int numLabels = Imgproc.connectedComponentsWithStats(binary, labels, stats, centroids, connectivity, CvType.CV_32S);

            // Generate random colors for each label (with consistent seed)
            Random rand = new Random(42);
            int[][] colors = new int[numLabels][3];
            colors[0] = new int[]{0, 0, 0}; // Background is black
            for (int i = 1; i < numLabels; i++) {
                colors[i] = new int[]{rand.nextInt(256), rand.nextInt(256), rand.nextInt(256)};
            }

            // Filter by min area - set small components to black
            if (minArea > 0) {
                for (int i = 1; i < numLabels; i++) {
                    int area = (int) stats.get(i, Imgproc.CC_STAT_AREA)[0];
                    if (area < minArea) {
                        colors[i] = new int[]{0, 0, 0};
                    }
                }
            }

            // Create colored output
            result = new Mat(input.rows(), input.cols(), CvType.CV_8UC3);
            for (int row = 0; row < labels.rows(); row++) {
                for (int col = 0; col < labels.cols(); col++) {
                    int label = (int) labels.get(row, col)[0];
                    result.put(row, col, colors[label][0], colors[label][1], colors[label][2]);
                }
            }

            return result;
        } finally {
            // Release intermediate Mats (but not result which is returned)
            if (gray != null) gray.release();
            if (binary != null) binary.release();
            if (labels != null) labels.release();
            if (stats != null) stats.release();
            if (centroids != null) centroids.release();
        }
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        Slider threshSlider = dialog.addSlider("Threshold:", 0, 255, threshold, "%.0f");
        CheckBox invertCheck = dialog.addCheckbox("Invert Threshold", invertThreshold);
        ComboBox<String> connCombo = dialog.addComboBox("Connectivity:", CONNECTIVITY_OPTIONS,
                connectivity == 4 ? "4" : "8");
        Spinner<Integer> minAreaSpinner = dialog.addSpinner("Min Area:", 0, 10000, minArea);

        // Save callback
        dialog.setOnOk(() -> {
            threshold = (int) threshSlider.getValue();
            invertThreshold = invertCheck.isSelected();
            connectivity = connCombo.getSelectionModel().getSelectedIndex() == 0 ? 4 : 8;
            minArea = minAreaSpinner.getValue();
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("threshold", threshold);
        json.addProperty("invertThreshold", invertThreshold);
        json.addProperty("connectivity", connectivity);
        json.addProperty("minArea", minArea);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        threshold = getJsonInt(json, "threshold", 127);
        invertThreshold = json.has("invertThreshold") ? json.get("invertThreshold").getAsBoolean() : false;
        connectivity = getJsonInt(json, "connectivity", 8);
        minArea = getJsonInt(json, "minArea", 0);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        threshold = getInt(node.properties, "threshold", 127);
        invertThreshold = getBool(node.properties, "invertThreshold", false);
        connectivity = getInt(node.properties, "connectivity", 8);
        minArea = getInt(node.properties, "minArea", 0);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("threshold", threshold);
        node.properties.put("invertThreshold", invertThreshold);
        node.properties.put("connectivity", connectivity);
        node.properties.put("minArea", minArea);
    }

    private boolean getBool(java.util.Map<String, Object> props, String key, boolean defaultValue) {
        if (props.containsKey(key)) {
            Object val = props.get(key);
            if (val instanceof Boolean) return (Boolean) val;
        }
        return defaultValue;
    }
}
