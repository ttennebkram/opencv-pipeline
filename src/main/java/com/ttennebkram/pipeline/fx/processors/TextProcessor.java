package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Slider;
import javafx.scene.control.Spinner;
import javafx.scene.control.TextField;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

/**
 * Text drawing processor.
 * Draws text on the image.
 */
@FXProcessorInfo(nodeType = "Text", category = "Drawing")
public class TextProcessor extends FXProcessorBase {

    // Properties with defaults
    private String text = "Hello";
    private int posX = 50;
    private int posY = 100;
    private int fontIndex = 0;
    private double fontScale = 1.0;
    private int colorR = 0;
    private int colorG = 255;
    private int colorB = 0;
    private int thickness = 2;

    private static final String[] FONTS = {
        "Simplex", "Plain", "Duplex", "Complex", "Triplex",
        "Complex Small", "Script Simplex", "Script Complex"
    };
    private static final int[] FONT_VALUES = {
        Imgproc.FONT_HERSHEY_SIMPLEX,
        Imgproc.FONT_HERSHEY_PLAIN,
        Imgproc.FONT_HERSHEY_DUPLEX,
        Imgproc.FONT_HERSHEY_COMPLEX,
        Imgproc.FONT_HERSHEY_TRIPLEX,
        Imgproc.FONT_HERSHEY_COMPLEX_SMALL,
        Imgproc.FONT_HERSHEY_SCRIPT_SIMPLEX,
        Imgproc.FONT_HERSHEY_SCRIPT_COMPLEX
    };

    @Override
    public String getNodeType() {
        return "Text";
    }

    @Override
    public String getCategory() {
        return "Drawing";
    }

    @Override
    public String getDescription() {
        return "Draw Text\nImgproc.putText(img, text, org, fontFace, fontScale, color, thickness)";
    }

    @Override
    public Mat process(Mat input) {
        if (isInvalidInput(input)) {
            return input;
        }

        Mat output = input.clone();
        int font = FONT_VALUES[Math.min(fontIndex, FONT_VALUES.length - 1)];
        Imgproc.putText(output, text, new Point(posX, posY), font, fontScale,
            new Scalar(colorB, colorG, colorR), thickness);
        return output;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        TextField textField = dialog.addTextField("Text:", text);
        Spinner<Integer> xSpinner = dialog.addSpinner("Position X:", -4096, 4096, posX);
        Spinner<Integer> ySpinner = dialog.addSpinner("Position Y:", -4096, 4096, posY);
        ComboBox<String> fontCombo = dialog.addComboBox("Font:", FONTS,
                FONTS[Math.min(fontIndex, FONTS.length - 1)]);
        Slider scaleSlider = dialog.addSlider("Font Scale:", 0.1, 10.0, fontScale, "%.1f");
        Spinner<Integer> rSpinner = dialog.addSpinner("Color R:", 0, 255, colorR);
        Spinner<Integer> gSpinner = dialog.addSpinner("Color G:", 0, 255, colorG);
        Spinner<Integer> bSpinner = dialog.addSpinner("Color B:", 0, 255, colorB);
        Spinner<Integer> thicknessSpinner = dialog.addSpinner("Thickness:", 1, 20, thickness);

        // Save callback
        dialog.setOnOk(() -> {
            text = textField.getText();
            posX = xSpinner.getValue();
            posY = ySpinner.getValue();
            fontIndex = fontCombo.getSelectionModel().getSelectedIndex();
            fontScale = scaleSlider.getValue();
            colorR = rSpinner.getValue();
            colorG = gSpinner.getValue();
            colorB = bSpinner.getValue();
            thickness = thicknessSpinner.getValue();
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("text", text);
        json.addProperty("posX", posX);
        json.addProperty("posY", posY);
        json.addProperty("fontIndex", fontIndex);
        json.addProperty("fontScale", fontScale);
        json.addProperty("colorR", colorR);
        json.addProperty("colorG", colorG);
        json.addProperty("colorB", colorB);
        json.addProperty("thickness", thickness);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        text = json.has("text") ? json.get("text").getAsString() : "Hello";
        posX = getJsonInt(json, "posX", 50);
        posY = getJsonInt(json, "posY", 100);
        fontIndex = getJsonInt(json, "fontIndex", 0);
        fontScale = json.has("fontScale") ? json.get("fontScale").getAsDouble() : 1.0;
        colorR = getJsonInt(json, "colorR", 0);
        colorG = getJsonInt(json, "colorG", 255);
        colorB = getJsonInt(json, "colorB", 0);
        thickness = getJsonInt(json, "thickness", 2);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        Object textObj = node.properties.get("text");
        text = textObj != null ? textObj.toString() : "Hello";
        posX = getInt(node.properties, "posX", 50);
        posY = getInt(node.properties, "posY", 100);
        fontIndex = getInt(node.properties, "fontIndex", 0);
        Object scaleObj = node.properties.get("fontScale");
        if (scaleObj instanceof Number) {
            fontScale = ((Number) scaleObj).doubleValue();
        } else {
            fontScale = 1.0;
        }
        colorR = getInt(node.properties, "colorR", 0);
        colorG = getInt(node.properties, "colorG", 255);
        colorB = getInt(node.properties, "colorB", 0);
        thickness = getInt(node.properties, "thickness", 2);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("text", text);
        node.properties.put("posX", posX);
        node.properties.put("posY", posY);
        node.properties.put("fontIndex", fontIndex);
        node.properties.put("fontScale", fontScale);
        node.properties.put("colorR", colorR);
        node.properties.put("colorG", colorG);
        node.properties.put("colorB", colorB);
        node.properties.put("thickness", thickness);
    }
}
