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
 * Circle drawing processor.
 * Draws a circle on the image.
 */
@FXProcessorInfo(
    nodeType = "Circle",
    displayName = "Circle",
    category = "Content",
    description = "Draw circle\nImgproc.circle(img, center, radius, color, thickness)"
)
public class CircleProcessor extends FXProcessorBase {

    // Properties with defaults
    private int centerX = 100;
    private int centerY = 100;
    private int radius = 50;
    private int colorR = 0;
    private int colorG = 255;
    private int colorB = 0;
    private int thickness = 2;
    private boolean filled = false;

    @Override
    public String getNodeType() {
        return "Circle";
    }

    @Override
    public String getCategory() {
        return "Drawing";
    }

    @Override
    public String getDescription() {
        return "Draw Circle\nImgproc.circle(img, center, radius, color, thickness)";
    }

    @Override
    public Mat process(Mat input) {
        if (isInvalidInput(input)) {
            return input;
        }

        Mat output = input.clone();
        int thick = filled ? -1 : thickness;
        Imgproc.circle(output, new Point(centerX, centerY), radius,
            new Scalar(colorB, colorG, colorR), thick);
        return output;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        Spinner<Integer> cxSpinner = dialog.addSpinner("Center X:", -4096, 4096, centerX);
        Spinner<Integer> cySpinner = dialog.addSpinner("Center Y:", -4096, 4096, centerY);
        Spinner<Integer> radiusSpinner = dialog.addSpinner("Radius:", 1, 2000, radius);
        Spinner<Integer> rSpinner = dialog.addSpinner("Color R:", 0, 255, colorR);
        Spinner<Integer> gSpinner = dialog.addSpinner("Color G:", 0, 255, colorG);
        Spinner<Integer> bSpinner = dialog.addSpinner("Color B:", 0, 255, colorB);
        Spinner<Integer> thicknessSpinner = dialog.addSpinner("Thickness:", 1, 50, thickness);
        CheckBox filledCheckBox = dialog.addCheckbox("Filled", filled);

        // Save callback
        dialog.setOnOk(() -> {
            centerX = cxSpinner.getValue();
            centerY = cySpinner.getValue();
            radius = radiusSpinner.getValue();
            colorR = rSpinner.getValue();
            colorG = gSpinner.getValue();
            colorB = bSpinner.getValue();
            thickness = thicknessSpinner.getValue();
            filled = filledCheckBox.isSelected();
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("centerX", centerX);
        json.addProperty("centerY", centerY);
        json.addProperty("radius", radius);
        json.addProperty("colorR", colorR);
        json.addProperty("colorG", colorG);
        json.addProperty("colorB", colorB);
        json.addProperty("thickness", thickness);
        json.addProperty("filled", filled);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        centerX = getJsonInt(json, "centerX", 100);
        centerY = getJsonInt(json, "centerY", 100);
        radius = getJsonInt(json, "radius", 50);
        colorR = getJsonInt(json, "colorR", 0);
        colorG = getJsonInt(json, "colorG", 255);
        colorB = getJsonInt(json, "colorB", 0);
        thickness = getJsonInt(json, "thickness", 2);
        filled = json.has("filled") ? json.get("filled").getAsBoolean() : false;
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        centerX = getInt(node.properties, "centerX", 100);
        centerY = getInt(node.properties, "centerY", 100);
        radius = getInt(node.properties, "radius", 50);
        colorR = getInt(node.properties, "colorR", 0);
        colorG = getInt(node.properties, "colorG", 255);
        colorB = getInt(node.properties, "colorB", 0);
        thickness = getInt(node.properties, "thickness", 2);
        filled = getBool(node.properties, "filled", false);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("centerX", centerX);
        node.properties.put("centerY", centerY);
        node.properties.put("radius", radius);
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
