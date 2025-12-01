package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.Slider;
import org.opencv.core.Mat;

/**
 * Gain processor.
 * Multiplies pixel values by a gain factor (logarithmic scale from 5% to 20x).
 */
@FXProcessorInfo(nodeType = "Gain", category = "Enhancement")
public class GainProcessor extends FXProcessorBase {

    // Properties with defaults
    private double gain = 1.0;

    private static final double LOG_RANGE = Math.log10(20.0);  // ~1.301

    @Override
    public String getNodeType() {
        return "Gain";
    }

    @Override
    public String getCategory() {
        return "Enhancement";
    }

    @Override
    public String getDescription() {
        return "Gain (Brightness Multiplier)\nMat.convertTo(dst, -1, gain, 0)";
    }

    @Override
    public Mat process(Mat input) {
        if (isInvalidInput(input)) {
            return input;
        }

        Mat output = new Mat();
        input.convertTo(output, -1, gain, 0);
        return output;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        // Convert gain to slider value (logarithmic scale)
        double sliderVal = (Math.log10(gain) / LOG_RANGE) * 50 + 50;
        sliderVal = Math.max(0, Math.min(100, sliderVal));

        // Custom converter for display
        java.util.function.Function<Double, String> gainConverter = (sliderValue) -> {
            double logVal = (sliderValue - 50) / 50.0 * LOG_RANGE;
            double g = Math.pow(10, logVal);
            if (g >= 0.995) {
                return String.format("%.1fx", g);
            } else {
                return String.format("%.0f%%", g * 100);
            }
        };

        Slider gainSlider = dialog.addLogGainSlider("Gain:", sliderVal, LOG_RANGE, gainConverter);
        dialog.addDescription("Logarithmic scale: 5% to 20x");

        // Save callback
        dialog.setOnOk(() -> {
            double logVal = (gainSlider.getValue() - 50) / 50.0 * LOG_RANGE;
            gain = Math.pow(10, logVal);
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("gain", gain);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        gain = json.has("gain") ? json.get("gain").getAsDouble() : 1.0;
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        Object gainObj = node.properties.get("gain");
        if (gainObj instanceof Number) {
            gain = ((Number) gainObj).doubleValue();
        } else {
            gain = 1.0;
        }
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("gain", gain);
    }
}
