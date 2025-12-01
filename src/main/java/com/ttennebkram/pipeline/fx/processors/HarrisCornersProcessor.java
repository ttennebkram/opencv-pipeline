package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Slider;
import javafx.scene.control.Spinner;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

/**
 * Harris Corner detection processor.
 * Detects corners using the Harris corner detection algorithm.
 */
@FXProcessorInfo(
    nodeType = "HarrisCorners",
    displayName = "Harris Corners",
    category = "Detection",
    description = "Harris corner detection\nImgproc.cornerHarris(src, dst, blockSize, ksize, k)"
)
public class HarrisCornersProcessor extends FXProcessorBase {

    // Properties with defaults
    private boolean showOriginal = true;
    private int blockSize = 2;
    private int ksize = 3;
    private int kPercent = 4;
    private int thresholdPercent = 1;
    private int markerSize = 5;
    private int colorR = 255;
    private int colorG = 0;
    private int colorB = 0;

    private static final String[] KERNEL_SIZES = {"3", "5", "7"};
    private static final int[] KSIZE_VALUES = {3, 5, 7};

    @Override
    public String getNodeType() {
        return "HarrisCorners";
    }

    @Override
    public String getCategory() {
        return "Feature Detection";
    }

    @Override
    public String getDescription() {
        return "Harris Corner Detection\nImgproc.cornerHarris(src, dst, blockSize, ksize, k)";
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
        gray.convertTo(gray, CvType.CV_32F);

        // Apply Harris corner detection
        Mat dst = new Mat();
        double k = kPercent / 100.0;
        Imgproc.cornerHarris(gray, dst, blockSize, ksize, k);

        // Normalize
        Mat dstNorm = new Mat();
        Core.normalize(dst, dstNorm, 0, 255, Core.NORM_MINMAX);
        Core.convertScaleAbs(dstNorm, dstNorm);

        // Prepare output
        Mat output;
        if (showOriginal) {
            output = input.clone();
        } else {
            output = new Mat(input.rows(), input.cols(), CvType.CV_8UC3, new Scalar(0, 0, 0));
        }

        // Draw corners
        double threshold = 255.0 * thresholdPercent / 100.0;
        for (int y = 0; y < dstNorm.rows(); y++) {
            for (int x = 0; x < dstNorm.cols(); x++) {
                double[] value = dstNorm.get(y, x);
                if (value != null && value[0] > threshold) {
                    Imgproc.circle(output, new Point(x, y), markerSize,
                        new Scalar(colorB, colorG, colorR), -1);
                }
            }
        }

        gray.release();
        dst.release();
        dstNorm.release();

        return output;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        CheckBox showOrigCheckBox = dialog.addCheckbox("Show Original Image", showOriginal);
        Slider blockSizeSlider = dialog.addSlider("Block Size:", 2, 10, blockSize, "%.0f");
        int ksizeIdx = (ksize == 5) ? 1 : (ksize == 7) ? 2 : 0;
        ComboBox<String> ksizeCombo = dialog.addComboBox("Kernel Size:", KERNEL_SIZES, KERNEL_SIZES[ksizeIdx]);
        Slider kSlider = dialog.addSlider("K (%):", 1, 10, kPercent, "%.0f%%");
        Slider threshSlider = dialog.addSlider("Threshold (%):", 1, 100, thresholdPercent, "%.0f%%");
        Slider markerSlider = dialog.addSlider("Marker Size:", 1, 15, markerSize, "%.0f");
        Spinner<Integer> rSpinner = dialog.addSpinner("Color R:", 0, 255, colorR);
        Spinner<Integer> gSpinner = dialog.addSpinner("Color G:", 0, 255, colorG);
        Spinner<Integer> bSpinner = dialog.addSpinner("Color B:", 0, 255, colorB);

        // Save callback
        dialog.setOnOk(() -> {
            showOriginal = showOrigCheckBox.isSelected();
            blockSize = (int) blockSizeSlider.getValue();
            int idx = ksizeCombo.getSelectionModel().getSelectedIndex();
            ksize = KSIZE_VALUES[Math.min(idx, KSIZE_VALUES.length - 1)];
            kPercent = (int) kSlider.getValue();
            thresholdPercent = (int) threshSlider.getValue();
            markerSize = (int) markerSlider.getValue();
            colorR = rSpinner.getValue();
            colorG = gSpinner.getValue();
            colorB = bSpinner.getValue();
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("showOriginal", showOriginal);
        json.addProperty("blockSize", blockSize);
        json.addProperty("ksize", ksize);
        json.addProperty("kPercent", kPercent);
        json.addProperty("thresholdPercent", thresholdPercent);
        json.addProperty("markerSize", markerSize);
        json.addProperty("colorR", colorR);
        json.addProperty("colorG", colorG);
        json.addProperty("colorB", colorB);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        showOriginal = json.has("showOriginal") ? json.get("showOriginal").getAsBoolean() : true;
        blockSize = getJsonInt(json, "blockSize", 2);
        ksize = getJsonInt(json, "ksize", 3);
        kPercent = getJsonInt(json, "kPercent", 4);
        thresholdPercent = getJsonInt(json, "thresholdPercent", 1);
        markerSize = getJsonInt(json, "markerSize", 5);
        colorR = getJsonInt(json, "colorR", 255);
        colorG = getJsonInt(json, "colorG", 0);
        colorB = getJsonInt(json, "colorB", 0);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        showOriginal = getBool(node.properties, "showOriginal", true);
        blockSize = getInt(node.properties, "blockSize", 2);
        ksize = getInt(node.properties, "ksize", 3);
        kPercent = getInt(node.properties, "kPercent", 4);
        thresholdPercent = getInt(node.properties, "thresholdPercent", 1);
        markerSize = getInt(node.properties, "markerSize", 5);
        colorR = getInt(node.properties, "colorR", 255);
        colorG = getInt(node.properties, "colorG", 0);
        colorB = getInt(node.properties, "colorB", 0);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("showOriginal", showOriginal);
        node.properties.put("blockSize", blockSize);
        node.properties.put("ksize", ksize);
        node.properties.put("kPercent", kPercent);
        node.properties.put("thresholdPercent", thresholdPercent);
        node.properties.put("markerSize", markerSize);
        node.properties.put("colorR", colorR);
        node.properties.put("colorG", colorG);
        node.properties.put("colorB", colorB);
    }

    private boolean getBool(java.util.Map<String, Object> props, String key, boolean defaultValue) {
        if (props.containsKey(key)) {
            Object val = props.get(key);
            if (val instanceof Boolean) return (Boolean) val;
        }
        return defaultValue;
    }
}
