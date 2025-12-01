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
@FXProcessorInfo(nodeType = "GaussianBlur", category = "Blur")
public class GaussianBlurProcessor extends FXProcessorBase {

    // Properties with defaults
    private int kernelSize = 15;
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

        // Kernel size must be odd
        int ksize = kernelSize;
        if (ksize % 2 == 0) {
            ksize++;
        }
        if (ksize < 1) {
            ksize = 1;
        }

        Mat output = new Mat();
        Imgproc.GaussianBlur(input, output, new Size(ksize, ksize), sigmaX);
        return output;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        // Kernel size slider (odd values only, 1-255)
        Slider ksizeSlider = dialog.addOddKernelSlider("Kernel Size:", kernelSize, 255);

        // Sigma X slider (0 = auto-compute from kernel size)
        Slider sigmaSlider = dialog.addSliderWithConverter("Sigma X:", 0, 100, sigmaX * 10,
                val -> {
                    double sigma = val / 10.0;
                    return sigma == 0 ? "auto" : String.format("%.1f", sigma);
                });

        // Save callback
        dialog.setOnOk(() -> {
            kernelSize = (int) ksizeSlider.getValue();
            sigmaX = sigmaSlider.getValue() / 10.0;
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("kernelSize", kernelSize);
        json.addProperty("sigmaX", sigmaX);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        kernelSize = getJsonInt(json, "kernelSize", 15);
        sigmaX = getJsonDouble(json, "sigmaX", 0.0);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        kernelSize = getInt(node.properties, "kernelSize", 15);
        sigmaX = getDouble(node.properties, "sigmaX", 0.0);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("kernelSize", kernelSize);
        node.properties.put("sigmaX", sigmaX);
    }

    // Getters/setters for direct access
    public int getKernelSize() {
        return kernelSize;
    }

    public void setKernelSize(int kernelSize) {
        this.kernelSize = kernelSize;
    }

    public double getSigmaX() {
        return sigmaX;
    }

    public void setSigmaX(double sigmaX) {
        this.sigmaX = sigmaX;
    }
}
