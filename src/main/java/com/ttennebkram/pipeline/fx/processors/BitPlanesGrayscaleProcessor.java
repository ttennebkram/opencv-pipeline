package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Label;
import javafx.scene.control.Slider;
import javafx.scene.layout.HBox;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

/**
 * Bit Planes Grayscale processor.
 * Select and adjust individual bit planes in a grayscale representation.
 */
@FXProcessorInfo(
    nodeType = "BitPlanesGrayscale",
    displayName = "Bit Planes Gray",
    category = "Basic",
    description = "Bit plane decomposition (grayscale)\nBit masking with gain"
)
public class BitPlanesGrayscaleProcessor extends FXProcessorBase {

    // Properties with defaults
    private boolean[] bitEnabled = new boolean[8];
    private double[] bitGain = new double[8];

    public BitPlanesGrayscaleProcessor() {
        // Initialize defaults (all enabled, gain 1.0)
        for (int i = 0; i < 8; i++) {
            bitEnabled[i] = true;
            bitGain[i] = 1.0;
        }
    }

    @Override
    public String getNodeType() {
        return "BitPlanesGrayscale";
    }

    @Override
    public String getCategory() {
        return "Color";
    }

    @Override
    public String getDescription() {
        return "Bit Planes Grayscale\nSelect and adjust bit planes";
    }

    @Override
    public Mat process(Mat input) {
        if (isInvalidInput(input)) {
            return input;
        }

        Mat gray = null;

        try {
            // Convert to grayscale if needed
            gray = new Mat();
            if (input.channels() == 3) {
                Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
            } else {
                gray = input.clone();
            }

            // Get grayscale data
            byte[] grayData = new byte[gray.rows() * gray.cols()];
            gray.get(0, 0, grayData);

            float[] resultData = new float[grayData.length];

            // Process each bit plane
            for (int i = 0; i < 8; i++) {
                if (!bitEnabled[i]) {
                    continue;
                }

                // Extract bit plane (bit 7-i, since i=0 is MSB)
                int bitIndex = 7 - i;

                for (int j = 0; j < grayData.length; j++) {
                    int pixelValue = grayData[j] & 0xFF;
                    int bit = (pixelValue >> bitIndex) & 1;
                    // Scale to original bit weight and apply gain
                    resultData[j] += bit * (1 << bitIndex) * (float) bitGain[i];
                }
            }

            // Clip to valid range [0, 255]
            for (int j = 0; j < resultData.length; j++) {
                resultData[j] = Math.max(0, Math.min(255, resultData[j]));
            }

            // Convert to 8-bit grayscale
            Mat resultMat = new Mat(gray.rows(), gray.cols(), CvType.CV_32F);
            resultMat.put(0, 0, resultData);

            Mat result8u = new Mat();
            resultMat.convertTo(result8u, CvType.CV_8U);
            resultMat.release();

            // Convert back to BGR for display
            Mat output = new Mat();
            Imgproc.cvtColor(result8u, output, Imgproc.COLOR_GRAY2BGR);
            result8u.release();

            return output;
        } finally {
            if (gray != null) gray.release();
        }
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        CheckBox[] checkBoxes = new CheckBox[8];
        Slider[] gainSliders = new Slider[8];

        for (int i = 0; i < 8; i++) {
            final int bitIndex = i;
            HBox row = new HBox(10);
            Label bitLabel = new Label("Bit " + i + ":");
            bitLabel.setMinWidth(50);
            CheckBox cb = new CheckBox();
            cb.setSelected(bitEnabled[i]);
            double sliderVal = Math.log10(bitGain[i]) * 100 + 100;
            Slider slider = new Slider(0, 200, Math.max(0, Math.min(200, sliderVal)));
            slider.setPrefWidth(120);
            Label gainLabel = new Label(String.format("%.2fx", bitGain[i]));
            gainLabel.setMinWidth(50);
            slider.valueProperty().addListener((obs, oldVal, newVal) -> {
                double g = Math.pow(10, (newVal.doubleValue() - 100) / 100.0);
                gainLabel.setText(String.format("%.2fx", g));
            });
            row.getChildren().addAll(bitLabel, cb, slider, gainLabel);
            dialog.addCustomContent(row);
            checkBoxes[bitIndex] = cb;
            gainSliders[bitIndex] = slider;
        }

        // Save callback
        dialog.setOnOk(() -> {
            for (int i = 0; i < 8; i++) {
                bitEnabled[i] = checkBoxes[i].isSelected();
                bitGain[i] = Math.pow(10, (gainSliders[i].getValue() - 100) / 100.0);
            }
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        JsonArray enabledArr = new JsonArray();
        JsonArray gainArr = new JsonArray();
        for (int i = 0; i < 8; i++) {
            enabledArr.add(bitEnabled[i]);
            gainArr.add(bitGain[i]);
        }
        json.add("bitEnabled", enabledArr);
        json.add("bitGain", gainArr);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        if (json.has("bitEnabled") && json.get("bitEnabled").isJsonArray()) {
            JsonArray arr = json.getAsJsonArray("bitEnabled");
            for (int i = 0; i < Math.min(arr.size(), 8); i++) {
                bitEnabled[i] = arr.get(i).getAsBoolean();
            }
        }
        if (json.has("bitGain") && json.get("bitGain").isJsonArray()) {
            JsonArray arr = json.getAsJsonArray("bitGain");
            for (int i = 0; i < Math.min(arr.size(), 8); i++) {
                bitGain[i] = arr.get(i).getAsDouble();
            }
        }
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        if (node.properties.containsKey("bitEnabled")) {
            Object obj = node.properties.get("bitEnabled");
            if (obj instanceof boolean[]) {
                boolean[] arr = (boolean[]) obj;
                for (int i = 0; i < Math.min(arr.length, 8); i++) {
                    bitEnabled[i] = arr[i];
                }
            }
        }
        if (node.properties.containsKey("bitGain")) {
            Object obj = node.properties.get("bitGain");
            if (obj instanceof double[]) {
                double[] arr = (double[]) obj;
                for (int i = 0; i < Math.min(arr.length, 8); i++) {
                    bitGain[i] = arr[i];
                }
            }
        }
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("bitEnabled", bitEnabled.clone());
        node.properties.put("bitGain", bitGain.clone());
    }
}
