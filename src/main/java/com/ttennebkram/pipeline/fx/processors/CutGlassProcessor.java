package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Slider;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

/**
 * Cut Glass Refraction Processor
 *
 * Applies a cut-glass/prismatic refraction effect using a displacement template.
 * Input 1: Source image to distort
 * Input 2: Displacement template (grayscale) defining the refraction pattern
 */
@FXProcessorInfo(
    nodeType = "CutGlass",
    displayName = "Cut Glass",
    buttonName = "Glass",
    category = "Filter",
    description = "Cut glass refraction effect\nUses displacement template for prismatic distortion",
    dualInput = true
)
public class CutGlassProcessor extends FXDualInputProcessor {

    // Embossment mode
    public enum EmbossMode {
        HARD("Hard - sharp facet edges"),
        SOFT("Soft - smooth transitions"),
        CRYSTAL("Crystal - extra refraction");

        private final String description;
        EmbossMode(String description) { this.description = description; }
        @Override public String toString() { return description; }
    }

    private EmbossMode embossMode = EmbossMode.HARD;
    private double strength = 50.0;      // Displacement strength (pixels)
    private int blurSize = 3;            // Blur kernel for soft mode
    private double alpha = 0.8;          // Blend alpha with original

    // Cached maps to avoid reallocation
    private Mat mapX = new Mat();
    private Mat mapY = new Mat();
    private int lastWidth = -1;
    private int lastHeight = -1;

    @Override
    public String getNodeType() {
        return "CutGlass";
    }

    @Override
    public String getCategory() {
        return "Filter";
    }

    @Override
    public String getDescription() {
        return "Cut Glass Refraction\nPrismatic distortion using displacement template";
    }

    @Override
    public Mat processDual(Mat input1, Mat input2) {
        // Handle null inputs
        if (input1 == null) {
            return input2 != null ? input2.clone() : null;
        }
        if (input2 == null) {
            return input1.clone();
        }

        int width = input1.cols();
        int height = input1.rows();

        // Prepare displacement template (resize to match input1, convert to grayscale float)
        Mat displacement = prepareDisplacementMap(input1, input2);

        // Apply blur for soft/crystal modes
        if (embossMode == EmbossMode.SOFT || embossMode == EmbossMode.CRYSTAL) {
            int ksize = blurSize | 1; // ensure odd
            Imgproc.GaussianBlur(displacement, displacement, new Size(ksize, ksize), 0);
        }

        // Build remap coordinate maps
        buildRemapMaps(displacement, width, height);

        // Apply remap
        Mat refracted = new Mat();
        Imgproc.remap(input1, refracted, mapX, mapY, Imgproc.INTER_LINEAR,
                      Core.BORDER_REFLECT, new Scalar(0, 0, 0));

        // Crystal mode: extra refraction pass
        if (embossMode == EmbossMode.CRYSTAL) {
            Mat temp = new Mat();
            Imgproc.remap(refracted, temp, mapX, mapY, Imgproc.INTER_LINEAR,
                          Core.BORDER_REFLECT, new Scalar(0, 0, 0));
            refracted.release();
            refracted = temp;
        }

        // Blend with original
        Mat output = new Mat();
        if (alpha < 1.0) {
            Core.addWeighted(refracted, alpha, input1, 1.0 - alpha, 0, output);
            refracted.release();
        } else {
            output = refracted;
        }

        displacement.release();
        return output;
    }

    private Mat prepareDisplacementMap(Mat input1, Mat input2) {
        // Resize input2 to match input1
        Mat resized = new Mat();
        if (input2.cols() != input1.cols() || input2.rows() != input1.rows()) {
            Imgproc.resize(input2, resized, new Size(input1.cols(), input1.rows()));
        } else {
            input2.copyTo(resized);
        }

        // Convert to grayscale float [0, 1]
        Mat gray = new Mat();
        if (resized.channels() > 1) {
            Imgproc.cvtColor(resized, gray, Imgproc.COLOR_BGR2GRAY);
            resized.release();
        } else {
            gray = resized;
        }

        Mat floatMap = new Mat();
        gray.convertTo(floatMap, CvType.CV_32F, 1.0 / 255.0);
        gray.release();

        return floatMap;
    }

    private void buildRemapMaps(Mat displacement, int width, int height) {
        // Reallocate if size changed
        if (width != lastWidth || height != lastHeight) {
            mapX.release();
            mapY.release();
            mapX = new Mat(height, width, CvType.CV_32FC1);
            mapY = new Mat(height, width, CvType.CV_32FC1);
            lastWidth = width;
            lastHeight = height;
        }

        // Compute displacement gradients for X and Y offsets
        Mat gradX = new Mat();
        Mat gradY = new Mat();
        Imgproc.Sobel(displacement, gradX, CvType.CV_32F, 1, 0, 3);
        Imgproc.Sobel(displacement, gradY, CvType.CV_32F, 0, 1, 3);

        // Scale gradients by strength
        Core.multiply(gradX, new Scalar(strength), gradX);
        Core.multiply(gradY, new Scalar(strength), gradY);

        // Build coordinate maps: mapX[y,x] = x + gradX, mapY[y,x] = y + gradY
        float[] gxRow = new float[width];
        float[] gyRow = new float[width];
        float[] mxRow = new float[width];
        float[] myRow = new float[width];

        for (int y = 0; y < height; y++) {
            gradX.get(y, 0, gxRow);
            gradY.get(y, 0, gyRow);

            for (int x = 0; x < width; x++) {
                float sx = x + gxRow[x];
                float sy = y + gyRow[x];

                // Clamp to valid range
                sx = Math.max(0, Math.min(width - 1, sx));
                sy = Math.max(0, Math.min(height - 1, sy));

                mxRow[x] = sx;
                myRow[x] = sy;
            }

            mapX.put(y, 0, mxRow);
            mapY.put(y, 0, myRow);
        }

        gradX.release();
        gradY.release();
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        // Emboss mode combo
        ComboBox<EmbossMode> modeCombo = dialog.addComboBox("Emboss Mode:",
            EmbossMode.values(), embossMode);

        // Strength slider
        Slider strengthSlider = dialog.addSlider("Strength:", 0, 200, strength, "%.0f px");

        // Blur size slider (for soft mode)
        Slider blurSlider = dialog.addOddKernelSlider("Blur Size:", blurSize, 15);

        // Alpha blend slider
        Slider alphaSlider = dialog.addSlider("Blend Alpha:", 0, 100, alpha * 100, "%.0f%%");

        dialog.setOnOk(() -> {
            embossMode = modeCombo.getValue();
            strength = strengthSlider.getValue();
            blurSize = (int) blurSlider.getValue();
            alpha = alphaSlider.getValue() / 100.0;
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("embossMode", embossMode.name());
        json.addProperty("strength", strength);
        json.addProperty("blurSize", blurSize);
        json.addProperty("alpha", alpha);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        String modeName = getJsonString(json, "embossMode", "HARD");
        try {
            embossMode = EmbossMode.valueOf(modeName);
        } catch (IllegalArgumentException e) {
            embossMode = EmbossMode.HARD;
        }
        strength = getJsonDouble(json, "strength", 50.0);
        blurSize = getJsonInt(json, "blurSize", 3);
        alpha = getJsonDouble(json, "alpha", 0.8);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        String modeName = (String) node.properties.getOrDefault("embossMode", "HARD");
        try {
            embossMode = EmbossMode.valueOf(modeName);
        } catch (IllegalArgumentException e) {
            embossMode = EmbossMode.HARD;
        }
        strength = getDouble(node.properties, "strength", 50.0);
        blurSize = getInt(node.properties, "blurSize", 3);
        alpha = getDouble(node.properties, "alpha", 0.8);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("embossMode", embossMode.name());
        node.properties.put("strength", strength);
        node.properties.put("blurSize", blurSize);
        node.properties.put("alpha", alpha);
    }

    // Getters/setters
    public EmbossMode getEmbossMode() { return embossMode; }
    public void setEmbossMode(EmbossMode mode) { this.embossMode = mode; }
    public double getStrength() { return strength; }
    public void setStrength(double strength) { this.strength = strength; }
    public int getBlurSize() { return blurSize; }
    public void setBlurSize(int blurSize) { this.blurSize = blurSize; }
    public double getAlpha() { return alpha; }
    public void setAlpha(double alpha) { this.alpha = alpha; }
}
