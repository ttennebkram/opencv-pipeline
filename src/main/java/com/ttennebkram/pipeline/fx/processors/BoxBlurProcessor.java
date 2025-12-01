package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.Slider;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Box Blur processor.
 * Applies a normalized box filter (averaging blur) to the image.
 */
@FXProcessorInfo(
    nodeType = "BoxBlur",
    displayName = "Box Blur",
    category = "Blur",
    description = "Box blur (average)\nImgproc.blur(src, dst, ksize)"
)
public class BoxBlurProcessor extends FXProcessorBase {

    // Properties with defaults
    private int kernelSizeX = 5;
    private int kernelSizeY = 5;

    @Override
    public String getNodeType() {
        return "BoxBlur";
    }

    @Override
    public String getCategory() {
        return "Blur";
    }

    @Override
    public String getDescription() {
        return "Box Blur (Averaging)\nImgproc.blur(src, dst, ksize)";
    }

    @Override
    public Mat process(Mat input) {
        if (isInvalidInput(input)) {
            return input;
        }

        // Kernel sizes must be positive odd numbers
        int kx = Math.max(1, kernelSizeX);
        int ky = Math.max(1, kernelSizeY);
        if (kx % 2 == 0) kx++;
        if (ky % 2 == 0) ky++;

        Mat output = new Mat();
        Imgproc.blur(input, output, new Size(kx, ky));
        return output;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        // Kernel size sliders (odd values only)
        Slider kxSlider = dialog.addOddKernelSlider("Kernel Size X:", kernelSizeX, 255);
        Slider kySlider = dialog.addOddKernelSlider("Kernel Size Y:", kernelSizeY, 255);

        // Save callback
        dialog.setOnOk(() -> {
            kernelSizeX = (int) kxSlider.getValue();
            kernelSizeY = (int) kySlider.getValue();
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("kernelSizeX", kernelSizeX);
        json.addProperty("kernelSizeY", kernelSizeY);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        kernelSizeX = getJsonInt(json, "kernelSizeX", getJsonInt(json, "ksize", 5));
        kernelSizeY = getJsonInt(json, "kernelSizeY", getJsonInt(json, "ksize", 5));
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        kernelSizeX = getInt(node.properties, "kernelSizeX", getInt(node.properties, "ksize", 5));
        kernelSizeY = getInt(node.properties, "kernelSizeY", getInt(node.properties, "ksize", 5));
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("kernelSizeX", kernelSizeX);
        node.properties.put("kernelSizeY", kernelSizeY);
        node.properties.remove("ksize");  // Remove old property name
    }
}
