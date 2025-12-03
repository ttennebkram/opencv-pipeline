package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.Spinner;
import org.opencv.core.Mat;
import org.opencv.core.Rect;

/**
 * Crop processor.
 * Crops a rectangular region from the image.
 */
@FXProcessorInfo(
    nodeType = "Crop",
    displayName = "Crop",
    category = "Transform",
    description = "Crop region of interest\nMat.submat(roi)"
)
public class CropProcessor extends FXProcessorBase {

    // Properties with defaults
    private int cropX = 0;
    private int cropY = 0;
    private int cropWidth = 100;
    private int cropHeight = 100;

    @Override
    public String getNodeType() {
        return "Crop";
    }

    @Override
    public String getCategory() {
        return "Transform";
    }

    @Override
    public String getDescription() {
        return "Crop\nMat.submat(rect)";
    }

    @Override
    public Mat process(Mat input) {
        if (isInvalidInput(input)) {
            return input;
        }

        // Clamp crop region to image bounds
        int x = Math.max(0, Math.min(cropX, input.cols() - 1));
        int y = Math.max(0, Math.min(cropY, input.rows() - 1));
        int w = Math.max(1, Math.min(cropWidth, input.cols() - x));
        int h = Math.max(1, Math.min(cropHeight, input.rows() - y));

        Rect roi = new Rect(x, y, w, h);
        return input.submat(roi).clone();
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        Spinner<Integer> xSpinner = dialog.addSpinner("X:", -4096, 4096, cropX);
        Spinner<Integer> ySpinner = dialog.addSpinner("Y:", -4096, 4096, cropY);
        Spinner<Integer> widthSpinner = dialog.addSpinner("Width:", 1, 4096, cropWidth);
        Spinner<Integer> heightSpinner = dialog.addSpinner("Height:", 1, 4096, cropHeight);

        // Save callback
        dialog.setOnOk(() -> {
            cropX = xSpinner.getValue();
            cropY = ySpinner.getValue();
            cropWidth = widthSpinner.getValue();
            cropHeight = heightSpinner.getValue();
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("cropX", cropX);
        json.addProperty("cropY", cropY);
        json.addProperty("cropWidth", cropWidth);
        json.addProperty("cropHeight", cropHeight);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        cropX = getJsonInt(json, "cropX", 0);
        cropY = getJsonInt(json, "cropY", 0);
        cropWidth = getJsonInt(json, "cropWidth", 100);
        cropHeight = getJsonInt(json, "cropHeight", 100);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        cropX = getInt(node.properties, "cropX", 0);
        cropY = getInt(node.properties, "cropY", 0);
        cropWidth = getInt(node.properties, "cropWidth", 100);
        cropHeight = getInt(node.properties, "cropHeight", 100);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("cropX", cropX);
        node.properties.put("cropY", cropY);
        node.properties.put("cropWidth", cropWidth);
        node.properties.put("cropHeight", cropHeight);
    }
}
