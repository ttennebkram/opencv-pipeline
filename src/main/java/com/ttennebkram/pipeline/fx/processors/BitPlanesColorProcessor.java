package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Label;
import javafx.scene.control.Slider;
import javafx.scene.layout.HBox;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Bit Planes Color processor.
 * Select and adjust individual bit planes per color channel.
 */
@FXProcessorInfo(nodeType = "BitPlanesColor", category = "Color")
public class BitPlanesColorProcessor extends FXProcessorBase {

    // Properties with defaults - [channel][bit]
    // Channel order: 0=Red, 1=Green, 2=Blue
    private boolean[][] bitEnabled = new boolean[3][8];
    private double[][] bitGain = new double[3][8];

    private static final String[] CHANNEL_NAMES = {"Red", "Green", "Blue"};
    private static final String[] PROP_NAMES = {"red", "green", "blue"};

    public BitPlanesColorProcessor() {
        // Initialize defaults (all enabled, gain 1.0)
        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < 8; i++) {
                bitEnabled[c][i] = true;
                bitGain[c][i] = 1.0;
            }
        }
    }

    @Override
    public String getNodeType() {
        return "BitPlanesColor";
    }

    @Override
    public String getCategory() {
        return "Color";
    }

    @Override
    public String getDescription() {
        return "Bit Planes Color\nSelect and adjust bit planes per channel";
    }

    @Override
    public Mat process(Mat input) {
        if (isInvalidInput(input)) {
            return input;
        }

        Mat color = null;
        boolean colorCreated = false;
        List<Mat> channels = new ArrayList<>();
        List<Mat> resultChannels = new ArrayList<>();

        try {
            // Ensure we have a color image
            if (input.channels() == 1) {
                color = new Mat();
                colorCreated = true;
                Imgproc.cvtColor(input, color, Imgproc.COLOR_GRAY2BGR);
            } else {
                color = input;
            }

            // Split into BGR channels
            Core.split(color, channels);

            // Process each channel (BGR order in OpenCV)
            // channels: 0=Blue, 1=Green, 2=Red
            int[] channelMap = {2, 1, 0}; // Red, Green, Blue -> BGR indices

            // Initialize result channels list
            for (int c = 0; c < 3; c++) {
                resultChannels.add(null);
            }

            for (int colorIdx = 0; colorIdx < 3; colorIdx++) {
                int bgrIdx = channelMap[colorIdx];
                Mat channel = channels.get(bgrIdx);

                // Get channel data
                byte[] channelData = new byte[channel.rows() * channel.cols()];
                channel.get(0, 0, channelData);

                float[] resultData = new float[channelData.length];

                // Process each bit plane
                for (int i = 0; i < 8; i++) {
                    if (!bitEnabled[colorIdx][i]) {
                        continue;
                    }

                    // Extract bit plane (bit 7-i, since i=0 is MSB)
                    int bitIndex = 7 - i;

                    for (int j = 0; j < channelData.length; j++) {
                        int pixelValue = channelData[j] & 0xFF;
                        int bit = (pixelValue >> bitIndex) & 1;
                        // Scale to original bit weight and apply gain
                        resultData[j] += bit * (1 << bitIndex) * (float) bitGain[colorIdx][i];
                    }
                }

                // Clip to valid range [0, 255]
                for (int j = 0; j < resultData.length; j++) {
                    resultData[j] = Math.max(0, Math.min(255, resultData[j]));
                }

                // Convert to 8-bit
                Mat resultMat = new Mat(channel.rows(), channel.cols(), CvType.CV_32F);
                Mat result8u = new Mat();
                try {
                    resultMat.put(0, 0, resultData);
                    resultMat.convertTo(result8u, CvType.CV_8U);
                    resultChannels.set(bgrIdx, result8u);
                } finally {
                    resultMat.release();
                }
            }

            // Merge channels back
            Mat output = new Mat();
            Core.merge(resultChannels, output);

            return output;
        } finally {
            // Release intermediate Mats
            if (colorCreated && color != null) color.release();
            for (Mat ch : channels) {
                if (ch != null) ch.release();
            }
            for (Mat ch : resultChannels) {
                if (ch != null) ch.release();
            }
        }
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        CheckBox[][] checkBoxes = new CheckBox[3][8];
        Slider[][] gainSliders = new Slider[3][8];

        for (int c = 0; c < 3; c++) {
            Label channelLabel = new Label(CHANNEL_NAMES[c] + " Channel:");
            channelLabel.setStyle("-fx-font-weight: bold;");
            dialog.addCustomContent(channelLabel);

            for (int i = 0; i < 8; i++) {
                final int channelIndex = c;
                final int bitIndex = i;
                HBox row = new HBox(10);
                Label bitLabel = new Label("Bit " + i + ":");
                bitLabel.setMinWidth(50);
                CheckBox cb = new CheckBox();
                cb.setSelected(bitEnabled[c][i]);
                double sliderVal = Math.log10(bitGain[c][i]) * 100 + 100;
                Slider slider = new Slider(0, 200, Math.max(0, Math.min(200, sliderVal)));
                slider.setPrefWidth(100);
                Label gainLabel = new Label(String.format("%.2fx", bitGain[c][i]));
                gainLabel.setMinWidth(50);
                slider.valueProperty().addListener((obs, oldVal, newVal) -> {
                    double g = Math.pow(10, (newVal.doubleValue() - 100) / 100.0);
                    gainLabel.setText(String.format("%.2fx", g));
                });
                row.getChildren().addAll(bitLabel, cb, slider, gainLabel);
                dialog.addCustomContent(row);
                checkBoxes[channelIndex][bitIndex] = cb;
                gainSliders[channelIndex][bitIndex] = slider;
            }
        }

        // Save callback
        dialog.setOnOk(() -> {
            for (int c = 0; c < 3; c++) {
                for (int i = 0; i < 8; i++) {
                    bitEnabled[c][i] = checkBoxes[c][i].isSelected();
                    bitGain[c][i] = Math.pow(10, (gainSliders[c][i].getValue() - 100) / 100.0);
                }
            }
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        for (int c = 0; c < 3; c++) {
            JsonArray enabledArr = new JsonArray();
            JsonArray gainArr = new JsonArray();
            for (int i = 0; i < 8; i++) {
                enabledArr.add(bitEnabled[c][i]);
                gainArr.add(bitGain[c][i]);
            }
            json.add(PROP_NAMES[c] + "BitEnabled", enabledArr);
            json.add(PROP_NAMES[c] + "BitGain", gainArr);
        }
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        for (int c = 0; c < 3; c++) {
            String enabledKey = PROP_NAMES[c] + "BitEnabled";
            String gainKey = PROP_NAMES[c] + "BitGain";

            if (json.has(enabledKey) && json.get(enabledKey).isJsonArray()) {
                JsonArray arr = json.getAsJsonArray(enabledKey);
                for (int i = 0; i < Math.min(arr.size(), 8); i++) {
                    bitEnabled[c][i] = arr.get(i).getAsBoolean();
                }
            }
            if (json.has(gainKey) && json.get(gainKey).isJsonArray()) {
                JsonArray arr = json.getAsJsonArray(gainKey);
                for (int i = 0; i < Math.min(arr.size(), 8); i++) {
                    bitGain[c][i] = arr.get(i).getAsDouble();
                }
            }
        }
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        for (int c = 0; c < 3; c++) {
            String enabledKey = PROP_NAMES[c] + "BitEnabled";
            String gainKey = PROP_NAMES[c] + "BitGain";

            if (node.properties.containsKey(enabledKey)) {
                Object obj = node.properties.get(enabledKey);
                if (obj instanceof boolean[]) {
                    boolean[] arr = (boolean[]) obj;
                    for (int i = 0; i < Math.min(arr.length, 8); i++) {
                        bitEnabled[c][i] = arr[i];
                    }
                }
            }
            if (node.properties.containsKey(gainKey)) {
                Object obj = node.properties.get(gainKey);
                if (obj instanceof double[]) {
                    double[] arr = (double[]) obj;
                    for (int i = 0; i < Math.min(arr.length, 8); i++) {
                        bitGain[c][i] = arr[i];
                    }
                }
            }
        }
    }

    @Override
    public void syncToFXNode(FXNode node) {
        for (int c = 0; c < 3; c++) {
            node.properties.put(PROP_NAMES[c] + "BitEnabled", bitEnabled[c].clone());
            node.properties.put(PROP_NAMES[c] + "BitGain", bitGain[c].clone());
        }
    }
}
