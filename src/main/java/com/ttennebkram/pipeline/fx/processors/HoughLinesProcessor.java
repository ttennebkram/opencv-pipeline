package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.Slider;
import javafx.scene.control.Spinner;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

/**
 * Hough Lines detection processor.
 * Detects line segments using the probabilistic Hough transform.
 */
@FXProcessorInfo(nodeType = "HoughLines", category = "Feature Detection")
public class HoughLinesProcessor extends FXProcessorBase {

    // Properties with defaults
    private int threshold = 50;
    private int minLineLength = 50;
    private int maxLineGap = 10;
    private int thickness = 2;
    private int colorR = 255;
    private int colorG = 0;
    private int colorB = 0;

    @Override
    public String getNodeType() {
        return "HoughLines";
    }

    @Override
    public String getCategory() {
        return "Feature Detection";
    }

    @Override
    public String getDescription() {
        return "Hough Line Detection\nImgproc.HoughLinesP(edges, lines, rho, theta, threshold, minLineLength, maxLineGap)";
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

        // Apply Canny edge detection
        Mat edges = new Mat();
        Imgproc.Canny(gray, edges, 50, 150);

        // Detect lines
        Mat lines = new Mat();
        Imgproc.HoughLinesP(edges, lines, 1, Math.PI / 180, threshold, minLineLength, maxLineGap);

        // Draw lines on output
        Mat output = input.clone();
        for (int i = 0; i < lines.rows(); i++) {
            double[] line = lines.get(i, 0);
            if (line != null && line.length >= 4) {
                Point pt1 = new Point(line[0], line[1]);
                Point pt2 = new Point(line[2], line[3]);
                Imgproc.line(output, pt1, pt2, new Scalar(colorB, colorG, colorR), thickness);
            }
        }

        gray.release();
        edges.release();
        lines.release();

        return output;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        Slider thresholdSlider = dialog.addSlider("Threshold:", 1, 200, threshold, "%.0f");
        Slider minLengthSlider = dialog.addSlider("Min Line Length:", 1, 200, minLineLength, "%.0f");
        Slider maxGapSlider = dialog.addSlider("Max Line Gap:", 1, 100, maxLineGap, "%.0f");
        Slider thicknessSlider = dialog.addSlider("Thickness:", 1, 10, thickness, "%.0f");
        Spinner<Integer> rSpinner = dialog.addSpinner("Color R:", 0, 255, colorR);
        Spinner<Integer> gSpinner = dialog.addSpinner("Color G:", 0, 255, colorG);
        Spinner<Integer> bSpinner = dialog.addSpinner("Color B:", 0, 255, colorB);

        // Save callback
        dialog.setOnOk(() -> {
            threshold = (int) thresholdSlider.getValue();
            minLineLength = (int) minLengthSlider.getValue();
            maxLineGap = (int) maxGapSlider.getValue();
            thickness = (int) thicknessSlider.getValue();
            colorR = rSpinner.getValue();
            colorG = gSpinner.getValue();
            colorB = bSpinner.getValue();
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("threshold", threshold);
        json.addProperty("minLineLength", minLineLength);
        json.addProperty("maxLineGap", maxLineGap);
        json.addProperty("thickness", thickness);
        json.addProperty("colorR", colorR);
        json.addProperty("colorG", colorG);
        json.addProperty("colorB", colorB);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        threshold = getJsonInt(json, "threshold", 50);
        minLineLength = getJsonInt(json, "minLineLength", 50);
        maxLineGap = getJsonInt(json, "maxLineGap", 10);
        thickness = getJsonInt(json, "thickness", 2);
        colorR = getJsonInt(json, "colorR", 255);
        colorG = getJsonInt(json, "colorG", 0);
        colorB = getJsonInt(json, "colorB", 0);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        threshold = getInt(node.properties, "threshold", 50);
        minLineLength = getInt(node.properties, "minLineLength", 50);
        maxLineGap = getInt(node.properties, "maxLineGap", 10);
        thickness = getInt(node.properties, "thickness", 2);
        colorR = getInt(node.properties, "colorR", 255);
        colorG = getInt(node.properties, "colorG", 0);
        colorB = getInt(node.properties, "colorB", 0);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("threshold", threshold);
        node.properties.put("minLineLength", minLineLength);
        node.properties.put("maxLineGap", maxLineGap);
        node.properties.put("thickness", thickness);
        node.properties.put("colorR", colorR);
        node.properties.put("colorG", colorG);
        node.properties.put("colorB", colorB);
    }
}
