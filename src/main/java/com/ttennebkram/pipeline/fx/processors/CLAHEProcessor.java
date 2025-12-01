package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Slider;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * CLAHE (Contrast Limited Adaptive Histogram Equalization) processor.
 * Enhances local contrast while limiting noise amplification.
 */
@FXProcessorInfo(nodeType = "CLAHE", category = "Enhancement")
public class CLAHEProcessor extends FXProcessorBase {

    // Properties with defaults
    private double clipLimit = 2.0;
    private int tileSize = 8;
    private int colorModeIndex = 0;  // 0=LAB, 1=HSV, 2=Grayscale

    private static final String[] COLOR_MODES = {"LAB", "HSV", "Grayscale"};

    @Override
    public String getNodeType() {
        return "CLAHE";
    }

    @Override
    public String getCategory() {
        return "Enhancement";
    }

    @Override
    public String getDescription() {
        return "CLAHE (Contrast Limited Adaptive Histogram Equalization)\nImgproc.createCLAHE(clipLimit, tileSize)";
    }

    @Override
    public Mat process(Mat input) {
        if (isInvalidInput(input)) {
            return input;
        }

        CLAHE clahe = Imgproc.createCLAHE(clipLimit, new Size(tileSize, tileSize));
        Mat output = new Mat();

        switch (colorModeIndex) {
            case 0: // LAB
                if (input.channels() == 3) {
                    Mat lab = new Mat();
                    Imgproc.cvtColor(input, lab, Imgproc.COLOR_BGR2Lab);
                    List<Mat> labChannels = new ArrayList<>();
                    Core.split(lab, labChannels);
                    // Apply CLAHE to L channel
                    Mat lChannel = new Mat();
                    clahe.apply(labChannels.get(0), lChannel);
                    labChannels.set(0, lChannel);
                    Core.merge(labChannels, lab);
                    Imgproc.cvtColor(lab, output, Imgproc.COLOR_Lab2BGR);
                    lab.release();
                    for (Mat m : labChannels) m.release();
                    lChannel.release();
                } else {
                    clahe.apply(input, output);
                    Imgproc.cvtColor(output, output, Imgproc.COLOR_GRAY2BGR);
                }
                break;
            case 1: // HSV
                if (input.channels() == 3) {
                    Mat hsv = new Mat();
                    Imgproc.cvtColor(input, hsv, Imgproc.COLOR_BGR2HSV);
                    List<Mat> hsvChannels = new ArrayList<>();
                    Core.split(hsv, hsvChannels);
                    // Apply CLAHE to V channel
                    Mat vChannel = new Mat();
                    clahe.apply(hsvChannels.get(2), vChannel);
                    hsvChannels.set(2, vChannel);
                    Core.merge(hsvChannels, hsv);
                    Imgproc.cvtColor(hsv, output, Imgproc.COLOR_HSV2BGR);
                    hsv.release();
                    for (Mat m : hsvChannels) m.release();
                    vChannel.release();
                } else {
                    clahe.apply(input, output);
                    Imgproc.cvtColor(output, output, Imgproc.COLOR_GRAY2BGR);
                }
                break;
            case 2: // Grayscale
            default:
                Mat gray = new Mat();
                if (input.channels() == 3) {
                    Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
                } else {
                    input.copyTo(gray);
                }
                Mat claheDst = new Mat();
                clahe.apply(gray, claheDst);
                Imgproc.cvtColor(claheDst, output, Imgproc.COLOR_GRAY2BGR);
                gray.release();
                claheDst.release();
                break;
        }

        return output;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        Slider clipLimitSlider = dialog.addSlider("Clip Limit:", 1.0, 40.0, clipLimit, "%.1f");
        Slider tileSizeSlider = dialog.addSlider("Tile Size:", 2, 32, tileSize, "%.0f");
        ComboBox<String> colorModeCombo = dialog.addComboBox("Color Mode:", COLOR_MODES,
                COLOR_MODES[Math.min(colorModeIndex, COLOR_MODES.length - 1)]);

        // Save callback
        dialog.setOnOk(() -> {
            clipLimit = clipLimitSlider.getValue();
            tileSize = (int) tileSizeSlider.getValue();
            colorModeIndex = colorModeCombo.getSelectionModel().getSelectedIndex();
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("clipLimit", clipLimit);
        json.addProperty("tileSize", tileSize);
        json.addProperty("colorModeIndex", colorModeIndex);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        clipLimit = json.has("clipLimit") ? json.get("clipLimit").getAsDouble() : 2.0;
        tileSize = getJsonInt(json, "tileSize", 8);
        colorModeIndex = getJsonInt(json, "colorModeIndex", 0);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        Object clipObj = node.properties.get("clipLimit");
        if (clipObj instanceof Number) {
            clipLimit = ((Number) clipObj).doubleValue();
        } else {
            clipLimit = 2.0;
        }
        tileSize = getInt(node.properties, "tileSize", 8);
        colorModeIndex = getInt(node.properties, "colorModeIndex", 0);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("clipLimit", clipLimit);
        node.properties.put("tileSize", tileSize);
        node.properties.put("colorModeIndex", colorModeIndex);
    }
}
