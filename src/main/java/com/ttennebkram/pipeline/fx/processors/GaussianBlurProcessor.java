package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.Slider;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Gaussian Blur processor.
 * Applies a Gaussian blur filter to smooth/blur the image.
 */
@FXProcessorInfo(
    nodeType = "GaussianBlur",
    displayName = "Gaussian Blur",
    category = "Blur",
    description = "Gaussian blur\nImgproc.GaussianBlur(src, dst, ksize, sigma)"
)
public class GaussianBlurProcessor extends FXProcessorBase {

    // Properties with defaults
    private int kernelSizeX = 15;
    private int kernelSizeY = 15;
    private double sigmaX = 0.0;

    @Override
    public String getNodeType() {
        return "GaussianBlur";
    }

    @Override
    public String getCategory() {
        return "Blur";
    }

    @Override
    public String getDescription() {
        return "Gaussian Blur\nImgproc.GaussianBlur(src, dst, ksize, sigmaX)";
    }

    @Override
    public Mat process(Mat input) {
        if (isInvalidInput(input)) {
            return input;
        }

        // Kernel sizes must be odd
        int ksizeX = kernelSizeX;
        if (ksizeX % 2 == 0) ksizeX++;
        if (ksizeX < 1) ksizeX = 1;

        int ksizeY = kernelSizeY;
        if (ksizeY % 2 == 0) ksizeY++;
        if (ksizeY < 1) ksizeY = 1;

        Mat output = new Mat();
        Imgproc.GaussianBlur(input, output, new Size(ksizeX, ksizeY), sigmaX);
        return output;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        // Kernel size X slider (odd values only, 1-255)
        Slider ksizeXSlider = dialog.addOddKernelSlider("Kernel Size X:", kernelSizeX, 255);

        // Kernel size Y slider (odd values only, 1-255)
        Slider ksizeYSlider = dialog.addOddKernelSlider("Kernel Size Y:", kernelSizeY, 255);

        // Sigma X slider (0 = auto-compute from kernel size)
        Slider sigmaSlider = dialog.addSliderWithConverter("Sigma X:", 0, 100, sigmaX * 10,
                val -> {
                    double sigma = val / 10.0;
                    return sigma == 0 ? "auto" : String.format("%.1f", sigma);
                });

        // Save callback
        dialog.setOnOk(() -> {
            kernelSizeX = (int) ksizeXSlider.getValue();
            kernelSizeY = (int) ksizeYSlider.getValue();
            sigmaX = sigmaSlider.getValue() / 10.0;
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("kernelSizeX", kernelSizeX);
        json.addProperty("kernelSizeY", kernelSizeY);
        json.addProperty("sigmaX", sigmaX);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        // Support legacy "kernelSize" property for backwards compatibility
        if (json.has("kernelSize") && !json.has("kernelSizeX")) {
            int legacySize = getJsonInt(json, "kernelSize", 15);
            kernelSizeX = legacySize;
            kernelSizeY = legacySize;
        } else {
            kernelSizeX = getJsonInt(json, "kernelSizeX", 15);
            kernelSizeY = getJsonInt(json, "kernelSizeY", 15);
        }
        sigmaX = getJsonDouble(json, "sigmaX", 0.0);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        // Support legacy "kernelSize" property for backwards compatibility
        if (node.properties.containsKey("kernelSize") && !node.properties.containsKey("kernelSizeX")) {
            int legacySize = getInt(node.properties, "kernelSize", 15);
            kernelSizeX = legacySize;
            kernelSizeY = legacySize;
        } else {
            kernelSizeX = getInt(node.properties, "kernelSizeX", 15);
            kernelSizeY = getInt(node.properties, "kernelSizeY", 15);
        }
        sigmaX = getDouble(node.properties, "sigmaX", 0.0);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("kernelSizeX", kernelSizeX);
        node.properties.put("kernelSizeY", kernelSizeY);
        node.properties.put("sigmaX", sigmaX);
        // Remove legacy property if present
        node.properties.remove("kernelSize");
    }

    // Getters/setters for direct access
    public int getKernelSizeX() {
        return kernelSizeX;
    }

    public void setKernelSizeX(int kernelSizeX) {
        this.kernelSizeX = kernelSizeX;
    }

    public int getKernelSizeY() {
        return kernelSizeY;
    }

    public void setKernelSizeY(int kernelSizeY) {
        this.kernelSizeY = kernelSizeY;
    }

    public double getSigmaX() {
        return sigmaX;
    }

    public void setSigmaX(double sigmaX) {
        this.sigmaX = sigmaX;
    }
}
