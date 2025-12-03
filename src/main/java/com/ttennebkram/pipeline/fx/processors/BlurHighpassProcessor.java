package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.Slider;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Highpass filter using Gaussian blur subtraction.
 * Extracts high-frequency details by:
 * 1. Apply Gaussian blur to get low-frequency (background)
 * 2. Subtract blur from original: diff = input - blur
 * 3. Take absolute value: output = |diff|
 *
 * The result shows edges and fine details while removing smooth areas.
 */
@FXProcessorInfo(
    nodeType = "BlurHighpass",
    displayName = "Blur Highpass",
    category = "Filter",
    description = "Highpass filter via blur subtraction\nout = |input - GaussianBlur(input)|"
)
public class BlurHighpassProcessor extends FXProcessorBase {

    // Properties with defaults
    private int kernelSizeX = 15;
    private int kernelSizeY = 15;
    private double sigmaX = 0.0;

    @Override
    public String getNodeType() {
        return "BlurHighpass";
    }

    @Override
    public String getCategory() {
        return "Filter";
    }

    @Override
    public String getDescription() {
        return "Highpass Filter (Blur Subtraction)\nout = |input - GaussianBlur(input)|";
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

        Mat blurred = null;
        Mat diff = null;
        Mat output = null;

        try {
            // Step 1: Apply Gaussian blur to get low-frequency component
            blurred = new Mat();
            Imgproc.GaussianBlur(input, blurred, new Size(ksizeX, ksizeY), sigmaX);

            // Step 2: Subtract blur from original (need signed type for negative values)
            // Convert to signed 16-bit for subtraction
            Mat inputS16 = new Mat();
            Mat blurredS16 = new Mat();
            input.convertTo(inputS16, CvType.CV_16S);
            blurred.convertTo(blurredS16, CvType.CV_16S);

            diff = new Mat();
            Core.subtract(inputS16, blurredS16, diff);

            // Release intermediate S16 mats
            inputS16.release();
            blurredS16.release();

            // Step 3: Take absolute value and convert back to 8-bit
            output = new Mat();
            Core.convertScaleAbs(diff, output);

            return output;

        } finally {
            // Clean up temporary mats
            if (blurred != null) blurred.release();
            if (diff != null) diff.release();
        }
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
        kernelSizeX = getJsonInt(json, "kernelSizeX", 15);
        kernelSizeY = getJsonInt(json, "kernelSizeY", 15);
        sigmaX = getJsonDouble(json, "sigmaX", 0.0);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        kernelSizeX = getInt(node.properties, "kernelSizeX", 15);
        kernelSizeY = getInt(node.properties, "kernelSizeY", 15);
        sigmaX = getDouble(node.properties, "sigmaX", 0.0);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("kernelSizeX", kernelSizeX);
        node.properties.put("kernelSizeY", kernelSizeY);
        node.properties.put("sigmaX", sigmaX);
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
