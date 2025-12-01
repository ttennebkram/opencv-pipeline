package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Spinner;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

/**
 * Blank Source processor.
 * Generates a solid color image.
 */
@FXProcessorInfo(nodeType = "BlankSource", category = "Source")
public class BlankSourceProcessor extends FXProcessorBase {

    // Properties with defaults
    private int imageWidth = 640;
    private int imageHeight = 480;
    private int colorIndex = 0;
    private int fpsIndex = 2;

    private static final String[] COLOR_OPTIONS = {"Black", "White", "Red", "Green", "Blue", "Yellow"};
    private static final String[] FPS_OPTIONS = {"1", "15", "30", "60"};

    private static final Scalar[] COLORS = {
        new Scalar(0, 0, 0),       // Black
        new Scalar(255, 255, 255), // White
        new Scalar(0, 0, 255),     // Red (BGR)
        new Scalar(0, 255, 0),     // Green
        new Scalar(255, 0, 0),     // Blue (BGR)
        new Scalar(0, 255, 255)    // Yellow (BGR)
    };

    @Override
    public String getNodeType() {
        return "BlankSource";
    }

    @Override
    public String getCategory() {
        return "Source";
    }

    @Override
    public String getDescription() {
        return "Blank Source\nGenerates a solid color image";
    }

    @Override
    public Mat process(Mat input) {
        // Generate a blank image
        Mat output = new Mat(imageHeight, imageWidth, CvType.CV_8UC3);
        Scalar color = COLORS[Math.min(colorIndex, COLORS.length - 1)];
        output.setTo(color);
        return output;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        Spinner<Integer> widthSpinner = dialog.addSpinner("Width:", 1, 4096, imageWidth);
        Spinner<Integer> heightSpinner = dialog.addSpinner("Height:", 1, 4096, imageHeight);
        ComboBox<String> colorCombo = dialog.addComboBox("Color:", COLOR_OPTIONS,
                COLOR_OPTIONS[Math.min(colorIndex, COLOR_OPTIONS.length - 1)]);
        ComboBox<String> fpsCombo = dialog.addComboBox("FPS:", FPS_OPTIONS,
                FPS_OPTIONS[Math.min(fpsIndex, FPS_OPTIONS.length - 1)]);

        // Save callback
        dialog.setOnOk(() -> {
            imageWidth = widthSpinner.getValue();
            imageHeight = heightSpinner.getValue();
            colorIndex = colorCombo.getSelectionModel().getSelectedIndex();
            fpsIndex = fpsCombo.getSelectionModel().getSelectedIndex();
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("imageWidth", imageWidth);
        json.addProperty("imageHeight", imageHeight);
        json.addProperty("colorIndex", colorIndex);
        json.addProperty("fpsIndex", fpsIndex);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        imageWidth = getJsonInt(json, "imageWidth", 640);
        imageHeight = getJsonInt(json, "imageHeight", 480);
        colorIndex = getJsonInt(json, "colorIndex", 0);
        fpsIndex = getJsonInt(json, "fpsIndex", 2);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        imageWidth = getInt(node.properties, "imageWidth", 640);
        imageHeight = getInt(node.properties, "imageHeight", 480);
        colorIndex = getInt(node.properties, "colorIndex", 0);
        fpsIndex = getInt(node.properties, "fpsIndex", 2);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("imageWidth", imageWidth);
        node.properties.put("imageHeight", imageHeight);
        node.properties.put("colorIndex", colorIndex);
        node.properties.put("fpsIndex", fpsIndex);
    }
}
