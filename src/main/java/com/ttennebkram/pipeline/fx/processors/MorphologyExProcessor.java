package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Slider;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * MorphologyEx processor.
 * Performs advanced morphological operations (erode, dilate, open, close, gradient, top hat, black hat).
 */
@FXProcessorInfo(
    nodeType = "MorphologyEx",
    displayName = "Morphology Ex",
    category = "Morphology",
    description = "Extended morphology operations\nImgproc.morphologyEx(src, dst, op, kernel)"
)
public class MorphologyExProcessor extends FXProcessorBase {

    // Properties with defaults
    private int operationIndex = 0;  // Index into OPERATIONS
    private int shapeIndex = 0;       // Index into SHAPES
    private int kernelWidth = 3;
    private int kernelHeight = 3;
    private int iterations = 1;

    private static final String[] OPERATIONS = {"Erode", "Dilate", "Open", "Close", "Gradient", "Top Hat", "Black Hat"};
    private static final int[] MORPH_OPS = {
        Imgproc.MORPH_ERODE,
        Imgproc.MORPH_DILATE,
        Imgproc.MORPH_OPEN,
        Imgproc.MORPH_CLOSE,
        Imgproc.MORPH_GRADIENT,
        Imgproc.MORPH_TOPHAT,
        Imgproc.MORPH_BLACKHAT
    };
    private static final String[] SHAPES = {"Rectangle", "Cross", "Ellipse"};
    private static final int[] MORPH_SHAPES = {
        Imgproc.MORPH_RECT,
        Imgproc.MORPH_CROSS,
        Imgproc.MORPH_ELLIPSE
    };

    @Override
    public String getNodeType() {
        return "MorphologyEx";
    }

    @Override
    public String getCategory() {
        return "Morphology";
    }

    @Override
    public String getDescription() {
        return "Morphological Operations\nImgproc.morphologyEx(src, dst, op, kernel, iterations)";
    }

    @Override
    public Mat process(Mat input) {
        if (isInvalidInput(input)) {
            return input;
        }

        int opIdx = Math.min(operationIndex, MORPH_OPS.length - 1);
        int shapeIdx = Math.min(shapeIndex, MORPH_SHAPES.length - 1);

        Mat kernel = Imgproc.getStructuringElement(
            MORPH_SHAPES[shapeIdx],
            new Size(kernelWidth, kernelHeight)
        );

        Mat output = new Mat();
        Imgproc.morphologyEx(input, output, MORPH_OPS[opIdx], kernel,
            new org.opencv.core.Point(-1, -1), iterations);

        kernel.release();
        return output;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        ComboBox<String> opCombo = dialog.addComboBox("Operation:", OPERATIONS,
                OPERATIONS[Math.min(operationIndex, OPERATIONS.length - 1)]);
        ComboBox<String> shapeCombo = dialog.addComboBox("Shape:", SHAPES,
                SHAPES[Math.min(shapeIndex, SHAPES.length - 1)]);
        Slider widthSlider = dialog.addSlider("Kernel Width:", 1, 31, kernelWidth, "%.0f");
        Slider heightSlider = dialog.addSlider("Kernel Height:", 1, 31, kernelHeight, "%.0f");
        Slider iterSlider = dialog.addSlider("Iterations:", 1, 20, iterations, "%.0f");

        // Save callback
        dialog.setOnOk(() -> {
            operationIndex = opCombo.getSelectionModel().getSelectedIndex();
            shapeIndex = shapeCombo.getSelectionModel().getSelectedIndex();
            kernelWidth = (int) widthSlider.getValue();
            kernelHeight = (int) heightSlider.getValue();
            iterations = (int) iterSlider.getValue();
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("operationIndex", operationIndex);
        json.addProperty("shapeIndex", shapeIndex);
        json.addProperty("kernelWidth", kernelWidth);
        json.addProperty("kernelHeight", kernelHeight);
        json.addProperty("iterations", iterations);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        operationIndex = getJsonInt(json, "operationIndex", 0);
        shapeIndex = getJsonInt(json, "shapeIndex", 0);
        kernelWidth = getJsonInt(json, "kernelWidth", 3);
        kernelHeight = getJsonInt(json, "kernelHeight", 3);
        iterations = getJsonInt(json, "iterations", 1);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        operationIndex = getInt(node.properties, "operationIndex", 0);
        shapeIndex = getInt(node.properties, "shapeIndex", 0);
        kernelWidth = getInt(node.properties, "kernelWidth", 3);
        kernelHeight = getInt(node.properties, "kernelHeight", 3);
        iterations = getInt(node.properties, "iterations", 1);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("operationIndex", operationIndex);
        node.properties.put("shapeIndex", shapeIndex);
        node.properties.put("kernelWidth", kernelWidth);
        node.properties.put("kernelHeight", kernelHeight);
        node.properties.put("iterations", iterations);
    }
}
