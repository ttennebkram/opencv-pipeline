package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.Slider;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Divide by Background processor - divides input by a blurred version of itself.
 * Useful for removing uneven illumination.
 *
 * Algorithm:
 * 1. Create background by applying Gaussian blur to input
 * 2. Convert both to float32
 * 3. Divide input / (background + epsilon) to avoid division by zero
 * 4. Normalize result to 0-255
 * 5. Convert back to uint8
 */
@FXProcessorInfo(
    nodeType = "DivideByBackground",
    displayName = "Divide by Background",
    category = "Filter",
    description = "Divide by blurred background\nout = normalize(input / (blur(input) + eps))"
)
public class DivideByBackgroundProcessor extends FXProcessorBase {

    // Properties with defaults
    private int blurSize = 51;     // Kernel size for background blur (must be odd)
    private double epsilon = 1.0;  // Small value to avoid divide-by-zero

    @Override
    public String getNodeType() {
        return "DivideByBackground";
    }

    @Override
    public String getCategory() {
        return "Filter";
    }

    @Override
    public String getDescription() {
        return "Divide by Background\nout = normalize(input / (blur(input) + eps))";
    }

    @Override
    public Mat process(Mat input) {
        if (isInvalidInput(input)) {
            return input;
        }

        // Ensure blur size is odd
        int ksize = blurSize;
        if (ksize % 2 == 0) ksize++;
        if (ksize < 1) ksize = 1;

        // Track all temporary Mats for cleanup
        Mat background = null;
        Mat inputFloat = null;
        Mat bgFloat = null;
        Mat divided = null;
        List<Mat> channels = new ArrayList<>();
        List<Mat> normalizedChannels = new ArrayList<>();

        try {
            // Create background by blurring the input
            background = new Mat();
            Imgproc.GaussianBlur(input, background, new Size(ksize, ksize), 0);

            // Convert both to float32
            inputFloat = new Mat();
            bgFloat = new Mat();
            input.convertTo(inputFloat, CvType.CV_32F);
            background.convertTo(bgFloat, CvType.CV_32F);

            // Add epsilon to background to avoid divide-by-zero
            Core.add(bgFloat, new Scalar(epsilon, epsilon, epsilon), bgFloat);

            // Divide input / background
            // Result will be around 1.0 for areas where input ≈ background
            divided = new Mat();
            Core.divide(inputFloat, bgFloat, divided);

            // Normalize each channel independently to 0-255 using min/max scaling
            Mat output;
            if (divided.channels() == 1) {
                // Single channel - find min/max and scale
                Core.MinMaxLocResult minMax = Core.minMaxLoc(divided);
                double minVal = minMax.minVal;
                double maxVal = minMax.maxVal;
                double range = maxVal - minVal;
                if (range < 0.0001) range = 1.0;

                double scale = 255.0 / range;
                double offset = -minVal * scale;

                Mat scaled = new Mat();
                try {
                    Core.multiply(divided, new Scalar(scale), scaled);
                    Core.add(scaled, new Scalar(offset), scaled);
                    output = new Mat();
                    scaled.convertTo(output, CvType.CV_8U);
                } finally {
                    scaled.release();
                }
            } else {
                // Multi-channel - split, normalize each channel with its own min/max, merge
                Core.split(divided, channels);

                for (Mat channel : channels) {
                    Core.MinMaxLocResult minMax = Core.minMaxLoc(channel);
                    double minVal = minMax.minVal;
                    double maxVal = minMax.maxVal;
                    double range = maxVal - minVal;
                    if (range < 0.0001) range = 1.0;

                    double scale = 255.0 / range;
                    double offset = -minVal * scale;

                    Mat scaled = new Mat();
                    try {
                        Core.multiply(channel, new Scalar(scale), scaled);
                        Core.add(scaled, new Scalar(offset), scaled);

                        Mat normCh8U = new Mat();
                        scaled.convertTo(normCh8U, CvType.CV_8U);
                        normalizedChannels.add(normCh8U);
                    } finally {
                        scaled.release();
                    }
                }

                output = new Mat();
                Core.merge(normalizedChannels, output);
            }

            return output;

        } finally {
            // Always clean up temporary Mats
            if (background != null) background.release();
            if (inputFloat != null) inputFloat.release();
            if (bgFloat != null) bgFloat.release();
            if (divided != null) divided.release();
            for (Mat ch : channels) ch.release();
            for (Mat ch : normalizedChannels) ch.release();
        }
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        // Blur size slider (odd values only, 1-255)
        Slider blurSlider = dialog.addOddKernelSlider("Blur Size:", blurSize, 255);

        // Epsilon slider (0.001 to 10.0, displayed as value)
        Slider epsilonSlider = dialog.addSliderWithConverter("Epsilon:", 1, 10000, epsilon * 1000,
                val -> String.format("%.3f", val / 1000.0));

        // Save callback
        dialog.setOnOk(() -> {
            blurSize = (int) blurSlider.getValue();
            epsilon = epsilonSlider.getValue() / 1000.0;
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("blurSize", blurSize);
        json.addProperty("epsilon", epsilon);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        blurSize = getJsonInt(json, "blurSize", 51);
        epsilon = getJsonDouble(json, "epsilon", 1.0);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        blurSize = getInt(node.properties, "blurSize", 51);
        epsilon = getDouble(node.properties, "epsilon", 1.0);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("blurSize", blurSize);
        node.properties.put("epsilon", epsilon);
    }

    // Getters/setters
    public int getBlurSize() {
        return blurSize;
    }

    public void setBlurSize(int blurSize) {
        this.blurSize = blurSize;
    }

    public double getEpsilon() {
        return epsilon;
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }
}
