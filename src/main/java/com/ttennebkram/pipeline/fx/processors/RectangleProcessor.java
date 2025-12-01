package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Spinner;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

/**
 * Rectangle drawing processor.
 * Draws a rectangle on the image.
 */
@FXProcessorInfo(nodeType = "Rectangle", category = "Drawing")
public class RectangleProcessor extends FXProcessorBase {

    // Properties with defaults
    private int x1 = 50;
    private int y1 = 50;
    private int x2 = 200;
    private int y2 = 150;
    private int colorR = 0;
    private int colorG = 255;
    private int colorB = 0;
    private int thickness = 2;
    private boolean filled = false;

    @Override
    public String getNodeType() {
        return "Rectangle";
    }

    @Override
    public String getCategory() {
        return "Drawing";
    }

    @Override
    public String getDescription() {
        return "Draw Rectangle\nImgproc.rectangle(img, pt1, pt2, color, thickness)";
    }

    @Override
    public Mat process(Mat input) {
        if (isInvalidInput(input)) {
            return input;
        }

        Mat output = input.clone();
        int thick = filled ? -1 : thickness;
        Imgproc.rectangle(output, new Point(x1, y1), new Point(x2, y2),
            new Scalar(colorB, colorG, colorR), thick);
        return output;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        Spinner<Integer> x1Spinner = dialog.addSpinner("X1:", -4096, 4096, x1);
        Spinner<Integer> y1Spinner = dialog.addSpinner("Y1:", -4096, 4096, y1);
        Spinner<Integer> x2Spinner = dialog.addSpinner("X2:", -4096, 4096, x2);
        Spinner<Integer> y2Spinner = dialog.addSpinner("Y2:", -4096, 4096, y2);
        Spinner<Integer> rSpinner = dialog.addSpinner("Color R:", 0, 255, colorR);
        Spinner<Integer> gSpinner = dialog.addSpinner("Color G:", 0, 255, colorG);
        Spinner<Integer> bSpinner = dialog.addSpinner("Color B:", 0, 255, colorB);
        Spinner<Integer> thicknessSpinner = dialog.addSpinner("Thickness:", 1, 50, thickness);
        CheckBox filledCheckBox = dialog.addCheckbox("Filled", filled);

        // Save callback
        dialog.setOnOk(() -> {
            x1 = x1Spinner.getValue();
            y1 = y1Spinner.getValue();
            x2 = x2Spinner.getValue();
            y2 = y2Spinner.getValue();
            colorR = rSpinner.getValue();
            colorG = gSpinner.getValue();
            colorB = bSpinner.getValue();
            thickness = thicknessSpinner.getValue();
            filled = filledCheckBox.isSelected();
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("x1", x1);
        json.addProperty("y1", y1);
        json.addProperty("x2", x2);
        json.addProperty("y2", y2);
        json.addProperty("colorR", colorR);
        json.addProperty("colorG", colorG);
        json.addProperty("colorB", colorB);
        json.addProperty("thickness", thickness);
        json.addProperty("filled", filled);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        x1 = getJsonInt(json, "x1", 50);
        y1 = getJsonInt(json, "y1", 50);
        x2 = getJsonInt(json, "x2", 200);
        y2 = getJsonInt(json, "y2", 150);
        colorR = getJsonInt(json, "colorR", 0);
        colorG = getJsonInt(json, "colorG", 255);
        colorB = getJsonInt(json, "colorB", 0);
        thickness = getJsonInt(json, "thickness", 2);
        filled = json.has("filled") ? json.get("filled").getAsBoolean() : false;
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        x1 = getInt(node.properties, "x1", 50);
        y1 = getInt(node.properties, "y1", 50);
        x2 = getInt(node.properties, "x2", 200);
        y2 = getInt(node.properties, "y2", 150);
        colorR = getInt(node.properties, "colorR", 0);
        colorG = getInt(node.properties, "colorG", 255);
        colorB = getInt(node.properties, "colorB", 0);
        thickness = getInt(node.properties, "thickness", 2);
        filled = getBool(node.properties, "filled", false);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("x1", x1);
        node.properties.put("y1", y1);
        node.properties.put("x2", x2);
        node.properties.put("y2", y2);
        node.properties.put("colorR", colorR);
        node.properties.put("colorG", colorG);
        node.properties.put("colorB", colorB);
        node.properties.put("thickness", thickness);
        node.properties.put("filled", filled);
    }

    private boolean getBool(java.util.Map<String, Object> props, String key, boolean defaultValue) {
        if (props.containsKey(key)) {
            Object val = props.get(key);
            if (val instanceof Boolean) return (Boolean) val;
        }
        return defaultValue;
    }
}
