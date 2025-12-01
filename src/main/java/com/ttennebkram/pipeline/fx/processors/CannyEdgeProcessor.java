package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Slider;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

/**
 * Canny Edge detector processor.
 * Detects edges using the Canny algorithm.
 *
 * Demonstrates multiple control types:
 * - Sliders for thresholds
 * - ComboBox for aperture size
 * - CheckBox for L2 gradient
 */
@FXProcessorInfo(
    nodeType = "CannyEdge",
    displayName = "Canny Edges",
    buttonName = "Canny",
    category = "Edges",
    description = "Canny edge detection\nImgproc.Canny(src, dst, threshold1, threshold2)"
)
public class CannyEdgeProcessor extends FXProcessorBase {

    // Properties with defaults
    private int threshold1 = 30;
    private int threshold2 = 150;
    private int apertureIndex = 0;  // 0=3, 1=5, 2=7
    private boolean l2Gradient = false;

    // Aperture size options
    private static final Integer[] APERTURE_OPTIONS = {3, 5, 7};

    @Override
    public String getNodeType() {
        return "CannyEdge";
    }

    @Override
    public String getCategory() {
        return "Edges";
    }

    @Override
    public String getDescription() {
        return "Canny Edge Detection\nImgproc.Canny(src, dst, threshold1, threshold2, apertureSize, L2gradient)";
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

        // Get aperture size from index
        int apertureSize = APERTURE_OPTIONS[Math.min(apertureIndex, APERTURE_OPTIONS.length - 1)];

        // Apply Canny edge detection
        Mat edges = new Mat();
        Imgproc.Canny(gray, edges, threshold1, threshold2, apertureSize, l2Gradient);
        gray.release();

        // Convert back to BGR for display
        Mat output = new Mat();
        Imgproc.cvtColor(edges, output, Imgproc.COLOR_GRAY2BGR);
        edges.release();

        return output;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        // Threshold 1 slider
        Slider thresh1Slider = dialog.addSlider("Threshold 1:", 0, 500, threshold1, "%.0f");

        // Threshold 2 slider
        Slider thresh2Slider = dialog.addSlider("Threshold 2:", 0, 500, threshold2, "%.0f");

        // Aperture size combo box
        ComboBox<Integer> apertureCombo = dialog.addComboBox("Aperture Size:", APERTURE_OPTIONS,
                APERTURE_OPTIONS[Math.min(apertureIndex, APERTURE_OPTIONS.length - 1)]);

        // L2 Gradient checkbox
        CheckBox l2Check = dialog.addCheckbox("Use L2 Gradient (more accurate)", l2Gradient);

        // Save callback
        dialog.setOnOk(() -> {
            threshold1 = (int) thresh1Slider.getValue();
            threshold2 = (int) thresh2Slider.getValue();
            // Find index of selected aperture
            Integer selectedAperture = apertureCombo.getValue();
            apertureIndex = 0;
            for (int i = 0; i < APERTURE_OPTIONS.length; i++) {
                if (APERTURE_OPTIONS[i].equals(selectedAperture)) {
                    apertureIndex = i;
                    break;
                }
            }
            l2Gradient = l2Check.isSelected();
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("threshold1", threshold1);
        json.addProperty("threshold2", threshold2);
        json.addProperty("apertureIndex", apertureIndex);
        json.addProperty("l2Gradient", l2Gradient);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        threshold1 = getJsonInt(json, "threshold1", 30);
        threshold2 = getJsonInt(json, "threshold2", 150);
        apertureIndex = getJsonInt(json, "apertureIndex", 0);
        l2Gradient = getJsonBoolean(json, "l2Gradient", false);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        threshold1 = getInt(node.properties, "threshold1", 30);
        threshold2 = getInt(node.properties, "threshold2", 150);
        apertureIndex = getInt(node.properties, "apertureIndex", 0);
        l2Gradient = getBoolean(node.properties, "l2Gradient", false);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("threshold1", threshold1);
        node.properties.put("threshold2", threshold2);
        node.properties.put("apertureIndex", apertureIndex);
        node.properties.put("l2Gradient", l2Gradient);
    }

    // Getters/setters
    public int getThreshold1() {
        return threshold1;
    }

    public void setThreshold1(int threshold1) {
        this.threshold1 = threshold1;
    }

    public int getThreshold2() {
        return threshold2;
    }

    public void setThreshold2(int threshold2) {
        this.threshold2 = threshold2;
    }

    public int getApertureIndex() {
        return apertureIndex;
    }

    public void setApertureIndex(int apertureIndex) {
        this.apertureIndex = apertureIndex;
    }

    public boolean isL2Gradient() {
        return l2Gradient;
    }

    public void setL2Gradient(boolean l2Gradient) {
        this.l2Gradient = l2Gradient;
    }
}
