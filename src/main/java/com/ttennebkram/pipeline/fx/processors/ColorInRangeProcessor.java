package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Slider;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

/**
 * ColorInRange processor.
 * Filters pixels based on color range in HSV or BGR color space.
 */
@FXProcessorInfo(
    nodeType = "ColorInRange",
    displayName = "Color In Range",
    category = "Filter",
    description = "Color range mask\nCore.inRange(src, lowerb, upperb, dst)"
)
public class ColorInRangeProcessor extends FXProcessorBase {

    // Properties with defaults
    private boolean useHSV = true;
    private int hLow = 0;
    private int hHigh = 179;
    private int sLow = 0;
    private int sHigh = 255;
    private int vLow = 0;
    private int vHigh = 255;
    private int outputMode = 0;  // 0=Mask Only, 1=Keep In-Range, 2=Keep Out-of-Range

    private static final String[] OUTPUT_MODES = {"Mask Only", "Keep In-Range", "Keep Out-of-Range"};

    @Override
    public String getNodeType() {
        return "ColorInRange";
    }

    @Override
    public String getCategory() {
        return "Color";
    }

    @Override
    public String getDescription() {
        return "Color In Range\nCore.inRange(src, lowerb, upperb, dst)";
    }

    @Override
    public Mat process(Mat input) {
        if (isInvalidInput(input)) {
            return input;
        }

        Mat converted = new Mat();
        if (useHSV && input.channels() == 3) {
            Imgproc.cvtColor(input, converted, Imgproc.COLOR_BGR2HSV);
        } else {
            input.copyTo(converted);
        }

        Mat mask = new Mat();
        Core.inRange(converted, new Scalar(hLow, sLow, vLow), new Scalar(hHigh, sHigh, vHigh), mask);

        Mat output = new Mat();
        switch (outputMode) {
            case 0: // Mask Only
                Imgproc.cvtColor(mask, output, Imgproc.COLOR_GRAY2BGR);
                break;
            case 1: // Keep In-Range
                input.copyTo(output, mask);
                break;
            case 2: // Keep Out-of-Range
                Mat invMask = new Mat();
                Core.bitwise_not(mask, invMask);
                input.copyTo(output, invMask);
                invMask.release();
                break;
            default:
                Imgproc.cvtColor(mask, output, Imgproc.COLOR_GRAY2BGR);
        }

        converted.release();
        mask.release();

        return output;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        CheckBox useHSVCheckBox = dialog.addCheckbox("Use HSV (unchecked = BGR)", useHSV);
        Slider hLowSlider = dialog.addSlider("H/B Low:", 0, 255, hLow, "%.0f");
        Slider hHighSlider = dialog.addSlider("H/B High:", 0, 255, hHigh, "%.0f");
        Slider sLowSlider = dialog.addSlider("S/G Low:", 0, 255, sLow, "%.0f");
        Slider sHighSlider = dialog.addSlider("S/G High:", 0, 255, sHigh, "%.0f");
        Slider vLowSlider = dialog.addSlider("V/R Low:", 0, 255, vLow, "%.0f");
        Slider vHighSlider = dialog.addSlider("V/R High:", 0, 255, vHigh, "%.0f");
        ComboBox<String> outputModeCombo = dialog.addComboBox("Output Mode:", OUTPUT_MODES,
                OUTPUT_MODES[Math.min(outputMode, OUTPUT_MODES.length - 1)]);

        // Save callback
        dialog.setOnOk(() -> {
            useHSV = useHSVCheckBox.isSelected();
            hLow = (int) hLowSlider.getValue();
            hHigh = (int) hHighSlider.getValue();
            sLow = (int) sLowSlider.getValue();
            sHigh = (int) sHighSlider.getValue();
            vLow = (int) vLowSlider.getValue();
            vHigh = (int) vHighSlider.getValue();
            outputMode = outputModeCombo.getSelectionModel().getSelectedIndex();
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("useHSV", useHSV);
        json.addProperty("hLow", hLow);
        json.addProperty("hHigh", hHigh);
        json.addProperty("sLow", sLow);
        json.addProperty("sHigh", sHigh);
        json.addProperty("vLow", vLow);
        json.addProperty("vHigh", vHigh);
        json.addProperty("outputMode", outputMode);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        useHSV = json.has("useHSV") ? json.get("useHSV").getAsBoolean() : true;
        hLow = getJsonInt(json, "hLow", 0);
        hHigh = getJsonInt(json, "hHigh", 179);
        sLow = getJsonInt(json, "sLow", 0);
        sHigh = getJsonInt(json, "sHigh", 255);
        vLow = getJsonInt(json, "vLow", 0);
        vHigh = getJsonInt(json, "vHigh", 255);
        outputMode = getJsonInt(json, "outputMode", 0);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        useHSV = getBool(node.properties, "useHSV", true);
        hLow = getInt(node.properties, "hLow", 0);
        hHigh = getInt(node.properties, "hHigh", 179);
        sLow = getInt(node.properties, "sLow", 0);
        sHigh = getInt(node.properties, "sHigh", 255);
        vLow = getInt(node.properties, "vLow", 0);
        vHigh = getInt(node.properties, "vHigh", 255);
        outputMode = getInt(node.properties, "outputMode", 0);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("useHSV", useHSV);
        node.properties.put("hLow", hLow);
        node.properties.put("hHigh", hHigh);
        node.properties.put("sLow", sLow);
        node.properties.put("sHigh", sHigh);
        node.properties.put("vLow", vLow);
        node.properties.put("vHigh", vHigh);
        node.properties.put("outputMode", outputMode);
    }

    private boolean getBool(java.util.Map<String, Object> props, String key, boolean defaultValue) {
        if (props.containsKey(key)) {
            Object val = props.get(key);
            if (val instanceof Boolean) return (Boolean) val;
        }
        return defaultValue;
    }
}
