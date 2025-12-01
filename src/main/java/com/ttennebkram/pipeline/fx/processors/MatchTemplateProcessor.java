package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Spinner;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

/**
 * Template Matching processor - finds where a template image appears in a source image.
 * Input 1: Source image
 * Input 2: Template image
 * Output: Result matrix showing correlation/matching scores at each location
 */
@FXProcessorInfo(nodeType = "MatchTemplate", category = "Feature Detection")
public class MatchTemplateProcessor extends FXDualInputProcessor {

    // Properties with defaults
    private int method = 5; // TM_CCOEFF_NORMED index
    private int outputMode = 0; // 0=result only, 1=source+rectangle, 2=source+overlay
    private int rectColorR = 0;
    private int rectColorG = 255;
    private int rectColorB = 0;
    private int rectThickness = 5;

    private static final String[] METHOD_NAMES = {
        "TM_SQDIFF", "TM_SQDIFF_NORMED",
        "TM_CCORR", "TM_CCORR_NORMED",
        "TM_CCOEFF", "TM_CCOEFF_NORMED"
    };
    private static final int[] METHOD_VALUES = {
        Imgproc.TM_SQDIFF, Imgproc.TM_SQDIFF_NORMED,
        Imgproc.TM_CCORR, Imgproc.TM_CCORR_NORMED,
        Imgproc.TM_CCOEFF, Imgproc.TM_CCOEFF_NORMED
    };
    private static final String[] OUTPUT_MODE_NAMES = {
        "Result Matrix Only",
        "Source + Rectangle",
        "Source + Result Overlay"
    };

    @Override
    public String getNodeType() {
        return "MatchTemplate";
    }

    @Override
    public String getCategory() {
        return "Feature Detection";
    }

    @Override
    public String getDescription() {
        return "Template Matching\nImgproc.matchTemplate(source, template, result, method)\n\nInput 1: Source image\nInput 2: Template to find";
    }

    @Override
    public Mat processDual(Mat source, Mat template) {
        if (source == null || source.empty()) {
            return source;
        }
        if (template == null || template.empty()) {
            return source.clone();
        }

        // Template must be smaller than source
        if (template.width() >= source.width() || template.height() >= source.height()) {
            return source.clone();
        }

        // Template must have at least 1x1 size
        if (template.width() < 1 || template.height() < 1) {
            return source.clone();
        }

        int methodValue = METHOD_VALUES[Math.min(method, METHOD_VALUES.length - 1)];
        Mat result = new Mat();

        try {
            // Perform template matching - returns correlation matrix (CV_32F)
            Imgproc.matchTemplate(source, template, result, methodValue);

            // Find the best match location
            Core.MinMaxLocResult mmr = Core.minMaxLoc(result);

            // For TM_SQDIFF and TM_SQDIFF_NORMED, best match is minimum; for others, maximum
            Point matchLoc;
            if (methodValue == Imgproc.TM_SQDIFF || methodValue == Imgproc.TM_SQDIFF_NORMED) {
                matchLoc = mmr.minLoc;
            } else {
                matchLoc = mmr.maxLoc;
            }

            // Handle output mode
            if (outputMode == 1) {
                // Mode 1: Source + Rectangle
                Mat output = source.clone();

                // Calculate rectangle coordinates
                int x1 = (int) Math.max(0, matchLoc.x);
                int y1 = (int) Math.max(0, matchLoc.y);
                int x2 = (int) Math.min(source.width() - 1, matchLoc.x + template.cols());
                int y2 = (int) Math.min(source.height() - 1, matchLoc.y + template.rows());

                // Draw rectangle around the matched region
                Point topLeft = new Point(x1, y1);
                Point bottomRight = new Point(x2, y2);
                Imgproc.rectangle(output, topLeft, bottomRight,
                                new Scalar(rectColorB, rectColorG, rectColorR), rectThickness);

                result.release();
                return output;
            } else if (outputMode == 2) {
                // Mode 2: Source + Result Overlay
                Mat normalized = new Mat();
                Core.normalize(result, normalized, 0, 255, Core.NORM_MINMAX);
                Mat result8u = new Mat();
                normalized.convertTo(result8u, CvType.CV_8UC1);

                // Resize result to match source dimensions
                Mat resizedResult = new Mat();
                Imgproc.resize(result8u, resizedResult, new Size(source.width(), source.height()));

                // Apply colormap to result for better visualization
                Mat resultColor = new Mat();
                Imgproc.applyColorMap(resizedResult, resultColor, Imgproc.COLORMAP_JET);

                // Blend with source image
                Mat output = new Mat();
                Core.addWeighted(source, 0.6, resultColor, 0.4, 0, output);

                result.release();
                normalized.release();
                result8u.release();
                resizedResult.release();
                resultColor.release();

                return output;
            } else {
                // Mode 0: Result Matrix Only (default)
                Mat normalized = new Mat();
                Core.normalize(result, normalized, 0, 255, Core.NORM_MINMAX);

                Mat result8u = new Mat();
                normalized.convertTo(result8u, CvType.CV_8UC1);

                result.release();
                normalized.release();

                return result8u;
            }
        } catch (Exception e) {
            if (!result.empty()) {
                result.release();
            }
            return source.clone();
        }
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        ComboBox<String> methodCombo = dialog.addComboBox("Method:", METHOD_NAMES,
                METHOD_NAMES[Math.min(method, METHOD_NAMES.length - 1)]);
        ComboBox<String> outputCombo = dialog.addComboBox("Output Mode:", OUTPUT_MODE_NAMES,
                OUTPUT_MODE_NAMES[Math.min(outputMode, OUTPUT_MODE_NAMES.length - 1)]);
        Spinner<Integer> rSpinner = dialog.addSpinner("Rect Red:", 0, 255, rectColorR);
        Spinner<Integer> gSpinner = dialog.addSpinner("Rect Green:", 0, 255, rectColorG);
        Spinner<Integer> bSpinner = dialog.addSpinner("Rect Blue:", 0, 255, rectColorB);
        Spinner<Integer> thickSpinner = dialog.addSpinner("Rect Thickness:", 1, 20, rectThickness);

        // Save callback
        dialog.setOnOk(() -> {
            method = methodCombo.getSelectionModel().getSelectedIndex();
            outputMode = outputCombo.getSelectionModel().getSelectedIndex();
            rectColorR = rSpinner.getValue();
            rectColorG = gSpinner.getValue();
            rectColorB = bSpinner.getValue();
            rectThickness = thickSpinner.getValue();
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("method", method);
        json.addProperty("outputMode", outputMode);
        json.addProperty("rectColorR", rectColorR);
        json.addProperty("rectColorG", rectColorG);
        json.addProperty("rectColorB", rectColorB);
        json.addProperty("rectThickness", rectThickness);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        method = getJsonInt(json, "method", 5);
        outputMode = getJsonInt(json, "outputMode", 0);
        rectColorR = getJsonInt(json, "rectColorR", 0);
        rectColorG = getJsonInt(json, "rectColorG", 255);
        rectColorB = getJsonInt(json, "rectColorB", 0);
        rectThickness = getJsonInt(json, "rectThickness", 5);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        method = getInt(node.properties, "method", 5);
        outputMode = getInt(node.properties, "outputMode", 0);
        rectColorR = getInt(node.properties, "rectColorR", 0);
        rectColorG = getInt(node.properties, "rectColorG", 255);
        rectColorB = getInt(node.properties, "rectColorB", 0);
        rectThickness = getInt(node.properties, "rectThickness", 5);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("method", method);
        node.properties.put("outputMode", outputMode);
        node.properties.put("rectColorR", rectColorR);
        node.properties.put("rectColorG", rectColorG);
        node.properties.put("rectColorB", rectColorB);
        node.properties.put("rectThickness", rectThickness);
    }
}
