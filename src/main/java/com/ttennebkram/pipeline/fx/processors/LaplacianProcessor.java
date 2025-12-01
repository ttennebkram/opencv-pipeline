package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Slider;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

/**
 * Laplacian edge detection processor.
 * Computes second-order derivatives for edge detection.
 */
@FXProcessorInfo(nodeType = "Laplacian", category = "Edge Detection")
public class LaplacianProcessor extends FXProcessorBase {

    // Properties with defaults
    private int kernelSizeIndex = 1;  // Index into {1, 3, 5, 7}
    private int scalePercent = 100;
    private int delta = 0;
    private boolean useAbsolute = true;

    private static final int[] KERNEL_SIZES = {1, 3, 5, 7};

    @Override
    public String getNodeType() {
        return "Laplacian";
    }

    @Override
    public String getCategory() {
        return "Edge Detection";
    }

    @Override
    public String getDescription() {
        return "Laplacian (2nd Derivative)\nImgproc.Laplacian(src, dst, ddepth, ksize, scale, delta)";
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

        int ksize = KERNEL_SIZES[Math.min(kernelSizeIndex, KERNEL_SIZES.length - 1)];
        double scale = scalePercent / 100.0;

        Mat output = new Mat();
        Imgproc.Laplacian(gray, output, CvType.CV_16S, ksize, scale, delta);

        if (useAbsolute) {
            Core.convertScaleAbs(output, output);
        } else {
            output.convertTo(output, CvType.CV_8U);
        }

        gray.release();

        // Convert back to BGR for display
        Mat bgr = new Mat();
        Imgproc.cvtColor(output, bgr, Imgproc.COLOR_GRAY2BGR);
        output.release();

        return bgr;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        String[] ksizes = {"1", "3", "5", "7"};
        ComboBox<String> ksizeCombo = dialog.addComboBox("Kernel Size:", ksizes,
                ksizes[Math.min(kernelSizeIndex, ksizes.length - 1)]);
        Slider scaleSlider = dialog.addSlider("Scale (%):", 10, 500, scalePercent, "%.0f%%");
        Slider deltaSlider = dialog.addSlider("Delta:", 0, 255, delta, "%.0f");
        CheckBox absCheckBox = dialog.addCheckbox("Use Absolute Value", useAbsolute);

        // Save callback
        dialog.setOnOk(() -> {
            kernelSizeIndex = ksizeCombo.getSelectionModel().getSelectedIndex();
            scalePercent = (int) scaleSlider.getValue();
            delta = (int) deltaSlider.getValue();
            useAbsolute = absCheckBox.isSelected();
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("kernelSizeIndex", kernelSizeIndex);
        json.addProperty("scalePercent", scalePercent);
        json.addProperty("delta", delta);
        json.addProperty("useAbsolute", useAbsolute);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        kernelSizeIndex = getJsonInt(json, "kernelSizeIndex", 1);
        scalePercent = getJsonInt(json, "scalePercent", 100);
        delta = getJsonInt(json, "delta", 0);
        useAbsolute = json.has("useAbsolute") ? json.get("useAbsolute").getAsBoolean() : true;
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        kernelSizeIndex = getInt(node.properties, "kernelSizeIndex", 1);
        scalePercent = getInt(node.properties, "scalePercent", 100);
        delta = getInt(node.properties, "delta", 0);
        useAbsolute = getBool(node.properties, "useAbsolute", true);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("kernelSizeIndex", kernelSizeIndex);
        node.properties.put("scalePercent", scalePercent);
        node.properties.put("delta", delta);
        node.properties.put("useAbsolute", useAbsolute);
    }

    private boolean getBool(java.util.Map<String, Object> props, String key, boolean defaultValue) {
        if (props.containsKey(key)) {
            Object val = props.get(key);
            if (val instanceof Boolean) return (Boolean) val;
        }
        return defaultValue;
    }
}
