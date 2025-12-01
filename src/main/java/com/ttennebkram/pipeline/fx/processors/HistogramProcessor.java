package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Spinner;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Histogram visualization processor.
 * Displays histogram of the image in various modes.
 */
@FXProcessorInfo(nodeType = "Histogram", category = "Analysis")
public class HistogramProcessor extends FXProcessorBase {

    // Properties with defaults
    private int modeIndex = 0;       // 0=Color (BGR), 1=Grayscale, 2=Per Channel
    private int backgroundMode = 0;  // 0=White, 1=Black, 2=Background Image
    private boolean fillBars = false;
    private int lineThickness = 4;

    private static final String[] MODES = {"Color (BGR)", "Grayscale", "Per Channel"};
    private static final String[] BG_MODES = {"White", "Black", "Background Image"};

    @Override
    public String getNodeType() {
        return "Histogram";
    }

    @Override
    public String getCategory() {
        return "Analysis";
    }

    @Override
    public String getDescription() {
        return "Histogram Visualization\nImgproc.calcHist()";
    }

    @Override
    public Mat process(Mat input) {
        if (isInvalidInput(input)) {
            return input;
        }

        int histSize = 256;
        int histWidth = input.cols();
        int histHeight = input.rows();

        // Create background
        Mat histImage;
        switch (backgroundMode) {
            case 1: // Black
                histImage = new Mat(histHeight, histWidth, CvType.CV_8UC3, new Scalar(0, 0, 0));
                break;
            case 2: // Background Image
                histImage = input.clone();
                break;
            default: // White
                histImage = new Mat(histHeight, histWidth, CvType.CV_8UC3, new Scalar(255, 255, 255));
        }

        MatOfFloat ranges = new MatOfFloat(0, 256);
        MatOfInt histSizeMat = new MatOfInt(histSize);

        if (modeIndex == 1) {
            // Grayscale histogram
            Mat gray = new Mat();
            if (input.channels() == 3) {
                Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
            } else {
                input.copyTo(gray);
            }
            Mat hist = new Mat();
            List<Mat> images = new ArrayList<>();
            images.add(gray);
            Imgproc.calcHist(images, new MatOfInt(0), new Mat(), hist, histSizeMat, ranges);
            Core.normalize(hist, hist, 0, histHeight, Core.NORM_MINMAX);

            drawHistogram(histImage, hist, new Scalar(128, 128, 128), histWidth, histHeight);
            hist.release();
            gray.release();
        } else {
            // Color or Per Channel
            if (input.channels() == 3) {
                List<Mat> bgrPlanes = new ArrayList<>();
                Core.split(input, bgrPlanes);

                Scalar[] colors = {new Scalar(255, 0, 0), new Scalar(0, 255, 0), new Scalar(0, 0, 255)};

                for (int i = 0; i < 3; i++) {
                    Mat hist = new Mat();
                    List<Mat> images = new ArrayList<>();
                    images.add(bgrPlanes.get(i));
                    Imgproc.calcHist(images, new MatOfInt(0), new Mat(), hist, histSizeMat, ranges);
                    Core.normalize(hist, hist, 0, histHeight, Core.NORM_MINMAX);

                    drawHistogram(histImage, hist, colors[i], histWidth, histHeight);
                    hist.release();
                }

                for (Mat plane : bgrPlanes) plane.release();
            } else {
                // Single channel - treat as grayscale
                Mat hist = new Mat();
                List<Mat> images = new ArrayList<>();
                images.add(input);
                Imgproc.calcHist(images, new MatOfInt(0), new Mat(), hist, histSizeMat, ranges);
                Core.normalize(hist, hist, 0, histHeight, Core.NORM_MINMAX);
                drawHistogram(histImage, hist, new Scalar(128, 128, 128), histWidth, histHeight);
                hist.release();
            }
        }

        ranges.release();
        histSizeMat.release();

        return histImage;
    }

    private void drawHistogram(Mat histImage, Mat hist, Scalar color, int histWidth, int histHeight) {
        int binWidth = Math.max(1, histWidth / 256);
        for (int i = 1; i < 256; i++) {
            double[] val0 = hist.get(i - 1, 0);
            double[] val1 = hist.get(i, 0);
            if (val0 != null && val1 != null) {
                if (fillBars) {
                    Imgproc.rectangle(histImage,
                        new Point(binWidth * (i - 1), histHeight),
                        new Point(binWidth * i, histHeight - val1[0]),
                        color, -1);
                } else {
                    Imgproc.line(histImage,
                        new Point(binWidth * (i - 1), histHeight - val0[0]),
                        new Point(binWidth * i, histHeight - val1[0]),
                        color, lineThickness);
                }
            }
        }
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        ComboBox<String> modeCombo = dialog.addComboBox("Mode:", MODES, MODES[Math.min(modeIndex, MODES.length - 1)]);
        ComboBox<String> bgCombo = dialog.addComboBox("Background:", BG_MODES, BG_MODES[Math.min(backgroundMode, BG_MODES.length - 1)]);
        CheckBox fillCheck = dialog.addCheckbox("Fill Bars", fillBars);
        Spinner<Integer> thickSpinner = dialog.addSpinner("Line Thickness:", 1, 10, lineThickness);

        // Save callback
        dialog.setOnOk(() -> {
            modeIndex = modeCombo.getSelectionModel().getSelectedIndex();
            backgroundMode = bgCombo.getSelectionModel().getSelectedIndex();
            fillBars = fillCheck.isSelected();
            lineThickness = thickSpinner.getValue();
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("modeIndex", modeIndex);
        json.addProperty("backgroundMode", backgroundMode);
        json.addProperty("fillBars", fillBars);
        json.addProperty("lineThickness", lineThickness);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        modeIndex = getJsonInt(json, "modeIndex", 0);
        backgroundMode = getJsonInt(json, "backgroundMode", 0);
        fillBars = json.has("fillBars") ? json.get("fillBars").getAsBoolean() : false;
        lineThickness = getJsonInt(json, "lineThickness", 4);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        modeIndex = getInt(node.properties, "modeIndex", 0);
        backgroundMode = getInt(node.properties, "backgroundMode", 0);
        fillBars = getBool(node.properties, "fillBars", false);
        lineThickness = getInt(node.properties, "lineThickness", 4);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("modeIndex", modeIndex);
        node.properties.put("backgroundMode", backgroundMode);
        node.properties.put("fillBars", fillBars);
        node.properties.put("lineThickness", lineThickness);
    }

    private boolean getBool(java.util.Map<String, Object> props, String key, boolean defaultValue) {
        if (props.containsKey(key)) {
            Object val = props.get(key);
            if (val instanceof Boolean) return (Boolean) val;
        }
        return defaultValue;
    }
}
