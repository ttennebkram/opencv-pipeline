package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.Spinner;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Resize processor.
 * Resizes the image to specified dimensions.
 */
@FXProcessorInfo(
    nodeType = "Resize",
    displayName = "Resize",
    category = "Transform",
    description = "Resize image\nImgproc.resize(src, dst, dsize)"
)
public class ResizeProcessor extends FXProcessorBase {

    // Properties with defaults
    private int width = 640;
    private int height = 480;

    @Override
    public String getNodeType() {
        return "Resize";
    }

    @Override
    public String getCategory() {
        return "Transform";
    }

    @Override
    public String getDescription() {
        return "Resize\nImgproc.resize(src, dst, size)";
    }

    @Override
    public Mat process(Mat input) {
        if (isInvalidInput(input)) {
            return input;
        }

        int w = width > 0 ? width : 640;
        int h = height > 0 ? height : 480;

        Mat output = new Mat();
        Imgproc.resize(input, output, new Size(w, h));
        return output;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        Spinner<Integer> widthSpinner = dialog.addSpinner("Width:", 1, 8192, width);
        Spinner<Integer> heightSpinner = dialog.addSpinner("Height:", 1, 8192, height);

        // Save callback
        dialog.setOnOk(() -> {
            width = widthSpinner.getValue();
            height = heightSpinner.getValue();
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("width", width);
        json.addProperty("height", height);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        width = getJsonInt(json, "width", 640);
        height = getJsonInt(json, "height", 480);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        width = getInt(node.properties, "width", 640);
        height = getInt(node.properties, "height", 480);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("width", width);
        node.properties.put("height", height);
    }
}
