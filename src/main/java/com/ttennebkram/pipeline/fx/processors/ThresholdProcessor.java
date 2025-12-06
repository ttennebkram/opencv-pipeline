package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Slider;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

/**
 * Threshold processor.
 * Applies binary thresholding to convert images to black/white.
 *
 * Demonstrates:
 * - Sliders for numeric values
 * - Radio buttons for threshold type selection
 */
@FXProcessorInfo(
    nodeType = "Threshold",
    displayName = "Threshold",
    category = "Basic",
    description = "Binary threshold\nImgproc.threshold(src, dst, thresh, maxval, type)"
)
public class ThresholdProcessor extends FXProcessorBase {

    // Properties with defaults
    private int threshold = 127;
    private int maxValue = 255;
    private int thresholdType = Imgproc.THRESH_BINARY;

    // Threshold type options
    private static final String[] TYPE_NAMES = {
            "Binary", "Binary Inv", "Trunc", "To Zero", "To Zero Inv",
            "Otsu", "Otsu Inv", "Triangle", "Triangle Inv"
    };
    private static final int[] TYPE_VALUES = {
            Imgproc.THRESH_BINARY,
            Imgproc.THRESH_BINARY_INV,
            Imgproc.THRESH_TRUNC,
            Imgproc.THRESH_TOZERO,
            Imgproc.THRESH_TOZERO_INV,
            Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU,
            Imgproc.THRESH_BINARY_INV | Imgproc.THRESH_OTSU,
            Imgproc.THRESH_BINARY | Imgproc.THRESH_TRIANGLE,
            Imgproc.THRESH_BINARY_INV | Imgproc.THRESH_TRIANGLE
    };

    // Store computed threshold for display (for OTSU/TRIANGLE modes)
    private double computedThreshold = -1;

    @Override
    public String getNodeType() {
        return "Threshold";
    }

    @Override
    public String getCategory() {
        return "Basic";
    }

    @Override
    public String getDescription() {
        return "Binary Threshold\nImgproc.threshold(src, dst, thresh, maxval, type)";
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

        // Apply threshold - returns computed threshold for OTSU/TRIANGLE modes
        Mat thresholded = new Mat();
        computedThreshold = Imgproc.threshold(gray, thresholded, threshold, maxValue, thresholdType);
        gray.release();

        // Update status text with threshold values
        // For OTSU/TRIANGLE: show both set threshold and calculated threshold
        // For manual modes: computed == set, so just show set
        if (fxNode != null) {
            if (usesAutoThreshold()) {
                fxNode.statusText = String.format("T: %d  Calc: %.0f", threshold, computedThreshold);
            } else {
                fxNode.statusText = String.format("T: %d", threshold);
            }
        }

        // Convert back to BGR for display
        Mat output = new Mat();
        Imgproc.cvtColor(thresholded, output, Imgproc.COLOR_GRAY2BGR);
        thresholded.release();

        return output;
    }

    /**
     * Check if this processor uses automatic threshold computation (OTSU or TRIANGLE).
     */
    public boolean usesAutoThreshold() {
        return (thresholdType & Imgproc.THRESH_OTSU) != 0 ||
               (thresholdType & Imgproc.THRESH_TRIANGLE) != 0;
    }

    /**
     * Get the computed threshold value (for OTSU/TRIANGLE modes).
     */
    public double getComputedThreshold() {
        return computedThreshold;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        // Threshold slider
        Slider threshSlider = dialog.addSlider("Threshold:", 0, 255, threshold, "%.0f");

        // Max value slider
        Slider maxValSlider = dialog.addSlider("Max Value:", 0, 255, maxValue, "%.0f");

        // Threshold type dropdown
        int currentTypeIndex = getTypeIndex(thresholdType);
        ComboBox<String> typeCombo = dialog.addComboBox("Type:", TYPE_NAMES,
                TYPE_NAMES[Math.min(currentTypeIndex, TYPE_NAMES.length - 1)]);

        // Save callback
        dialog.setOnOk(() -> {
            threshold = (int) threshSlider.getValue();
            maxValue = (int) maxValSlider.getValue();
            int selectedIndex = typeCombo.getSelectionModel().getSelectedIndex();
            thresholdType = TYPE_VALUES[selectedIndex];
        });
    }

    /**
     * Get the index of a threshold type in TYPE_VALUES.
     */
    private int getTypeIndex(int type) {
        for (int i = 0; i < TYPE_VALUES.length; i++) {
            if (TYPE_VALUES[i] == type) {
                return i;
            }
        }
        return 0;
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("threshold", threshold);
        json.addProperty("maxValue", maxValue);
        json.addProperty("thresholdType", thresholdType);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        threshold = getJsonInt(json, "threshold", 127);
        maxValue = getJsonInt(json, "maxValue", 255);
        thresholdType = getJsonInt(json, "thresholdType", Imgproc.THRESH_BINARY);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        threshold = getInt(node.properties, "threshold", 127);
        maxValue = getInt(node.properties, "maxValue", 255);
        thresholdType = getInt(node.properties, "thresholdType", Imgproc.THRESH_BINARY);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("threshold", threshold);
        node.properties.put("maxValue", maxValue);
        node.properties.put("thresholdType", thresholdType);
    }

    // Getters/setters
    public int getThreshold() {
        return threshold;
    }

    public void setThreshold(int threshold) {
        this.threshold = threshold;
    }

    public int getMaxValue() {
        return maxValue;
    }

    public void setMaxValue(int maxValue) {
        this.maxValue = maxValue;
    }

    public int getThresholdType() {
        return thresholdType;
    }

    public void setThresholdType(int thresholdType) {
        this.thresholdType = thresholdType;
    }
}
