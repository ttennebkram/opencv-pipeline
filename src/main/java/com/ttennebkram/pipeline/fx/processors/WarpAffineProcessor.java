package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Slider;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Warp Affine (geometric transformation) processor.
 * Applies translation, rotation, and scaling.
 */
@FXProcessorInfo(
    nodeType = "WarpAffine",
    displayName = "Warp Affine",
    category = "Transform",
    description = "Affine transformation\nImgproc.warpAffine(src, dst, M, dsize)"
)
public class WarpAffineProcessor extends FXProcessorBase {

    // Properties with defaults
    private double scaleX = 1.0;
    private double scaleY = 1.0;
    private double rotation = 0.0;
    private double translateX = 0.0;
    private double translateY = 0.0;
    private int borderModeIndex = 0;

    private static final String[] BORDER_MODE_NAMES = {
        "Constant (black)", "Replicate edge", "Reflect", "Wrap"
    };
    private static final int[] BORDER_MODES = {
        Core.BORDER_CONSTANT, Core.BORDER_REPLICATE, Core.BORDER_REFLECT, Core.BORDER_WRAP
    };

    @Override
    public String getNodeType() {
        return "WarpAffine";
    }

    @Override
    public String getCategory() {
        return "Transform";
    }

    @Override
    public String getDescription() {
        return "Warp Affine: Translation, rotation, scaling\nImgproc.warpAffine(src, dst, M, dsize)";
    }

    @Override
    public Mat process(Mat input) {
        if (isInvalidInput(input)) {
            return input;
        }

        int height = input.rows();
        int width = input.cols();

        // Determine center point (always use image center)
        double cx = width / 2.0;
        double cy = height / 2.0;

        // Build transformation matrix
        // Use average of scaleX and scaleY for uniform scaling in getRotationMatrix2D
        double scale = (scaleX + scaleY) / 2.0;
        // Negate angle so positive = clockwise (more intuitive)
        Mat M = Imgproc.getRotationMatrix2D(new Point(cx, cy), -rotation, scale);

        // Add translation
        double[] row0 = M.get(0, 2);
        double[] row1 = M.get(1, 2);
        M.put(0, 2, row0[0] + translateX);
        M.put(1, 2, row1[0] + translateY);

        // Get border mode
        int borderMode = BORDER_MODES[borderModeIndex < BORDER_MODES.length ? borderModeIndex : 0];

        // Apply transformation
        Mat output = new Mat();
        Imgproc.warpAffine(input, output, M, new Size(width, height), Imgproc.INTER_LINEAR, borderMode);
        M.release();

        return output;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        Slider scaleXSlider = dialog.addSlider("Scale X:", 0.1, 5.0, scaleX, "%.2f");
        Slider scaleYSlider = dialog.addSlider("Scale Y:", 0.1, 5.0, scaleY, "%.2f");
        Slider rotSlider = dialog.addSlider("Rotation:", -180, 180, rotation, "%.1f°");
        Slider txSlider = dialog.addSlider("Translate X:", -500, 500, translateX, "%.0f");
        Slider tySlider = dialog.addSlider("Translate Y:", -500, 500, translateY, "%.0f");
        ComboBox<String> borderCombo = dialog.addComboBox("Border Mode:", BORDER_MODE_NAMES,
                BORDER_MODE_NAMES[borderModeIndex < BORDER_MODE_NAMES.length ? borderModeIndex : 0]);

        // Save callback
        dialog.setOnOk(() -> {
            scaleX = scaleXSlider.getValue();
            scaleY = scaleYSlider.getValue();
            rotation = rotSlider.getValue();
            translateX = txSlider.getValue();
            translateY = tySlider.getValue();
            borderModeIndex = borderCombo.getSelectionModel().getSelectedIndex();
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("scaleX", scaleX);
        json.addProperty("scaleY", scaleY);
        json.addProperty("rotation", rotation);
        json.addProperty("translateX", translateX);
        json.addProperty("translateY", translateY);
        json.addProperty("borderModeIndex", borderModeIndex);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        scaleX = json.has("scaleX") ? json.get("scaleX").getAsDouble() : 1.0;
        scaleY = json.has("scaleY") ? json.get("scaleY").getAsDouble() : 1.0;
        rotation = json.has("rotation") ? json.get("rotation").getAsDouble() : 0.0;
        translateX = json.has("translateX") ? json.get("translateX").getAsDouble() : 0.0;
        translateY = json.has("translateY") ? json.get("translateY").getAsDouble() : 0.0;
        borderModeIndex = getJsonInt(json, "borderModeIndex", 0);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        scaleX = getDouble(node.properties, "scaleX", 1.0);
        scaleY = getDouble(node.properties, "scaleY", 1.0);
        rotation = getDouble(node.properties, "rotation", 0.0);
        translateX = getDouble(node.properties, "translateX", 0.0);
        translateY = getDouble(node.properties, "translateY", 0.0);
        borderModeIndex = getInt(node.properties, "borderModeIndex", 0);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("scaleX", scaleX);
        node.properties.put("scaleY", scaleY);
        node.properties.put("rotation", rotation);
        node.properties.put("translateX", translateX);
        node.properties.put("translateY", translateY);
        node.properties.put("borderModeIndex", borderModeIndex);
    }

}
