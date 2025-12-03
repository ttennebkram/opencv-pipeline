package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.Slider;
import org.opencv.core.Core;
import org.opencv.core.Mat;

/**
 * AddWeighted processor - blends two images with adjustable weights.
 * output = input1 * alpha + input2 * beta + gamma
 *
 * Demonstrates dual-input processing with multiple sliders.
 */
@FXProcessorInfo(
    nodeType = "AddWeighted",
    displayName = "Add Weighted",
    category = "Dual Input",
    description = "Weighted addition (blend)\nCore.addWeighted(src1, alpha, src2, beta, gamma, dst)",
    dualInput = true
)
public class AddWeightedProcessor extends FXDualInputProcessor {

    // Properties with defaults
    private double alpha = 0.5;
    private double beta = 0.5;
    private double gamma = 0.0;

    @Override
    public String getNodeType() {
        return "AddWeighted";
    }

    @Override
    public String getCategory() {
        return "Dual Input";
    }

    @Override
    public String getDescription() {
        return "Weighted Addition\nCore.addWeighted(src1, alpha, src2, beta, gamma, dst)";
    }

    @Override
    public Mat processDual(Mat input1, Mat input2) {
        // Handle null inputs
        if (input1 == null && input2 == null) {
            return null;
        }
        if (input1 == null) {
            return input2.clone();
        }
        if (input2 == null) {
            return input1.clone();
        }

        // Prepare input2 to match input1 (resize and convert type)
        PreparedInput prep = prepareInput2(input1, input2);

        // Perform weighted addition
        Mat output = new Mat();
        Core.addWeighted(input1, alpha, prep.mat, beta, gamma, output);

        // Clean up
        prep.releaseIfNeeded();

        return output;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        // Alpha slider (weight for input1)
        Slider alphaSlider = dialog.addSlider("Alpha (img1 weight):", 0, 100, alpha * 100, "%.0f%%");

        // Beta slider (weight for input2)
        Slider betaSlider = dialog.addSlider("Beta (img2 weight):", 0, 100, beta * 100, "%.0f%%");

        // Gamma slider (brightness adjustment)
        Slider gammaSlider = dialog.addSlider("Gamma (brightness):", -128, 128, gamma, "%.0f");

        // Save callback
        dialog.setOnOk(() -> {
            alpha = alphaSlider.getValue() / 100.0;
            beta = betaSlider.getValue() / 100.0;
            gamma = gammaSlider.getValue();
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("alpha", alpha);
        json.addProperty("beta", beta);
        json.addProperty("gamma", gamma);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        alpha = getJsonDouble(json, "alpha", 0.5);
        beta = getJsonDouble(json, "beta", 0.5);
        gamma = getJsonDouble(json, "gamma", 0.0);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        alpha = getDouble(node.properties, "alpha", 0.5);
        beta = getDouble(node.properties, "beta", 0.5);
        gamma = getDouble(node.properties, "gamma", 0.0);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("alpha", alpha);
        node.properties.put("beta", beta);
        node.properties.put("gamma", gamma);
    }

    // Getters/setters
    public double getAlpha() {
        return alpha;
    }

    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }

    public double getBeta() {
        return beta;
    }

    public void setBeta(double beta) {
        this.beta = beta;
    }

    public double getGamma() {
        return gamma;
    }

    public void setGamma(double gamma) {
        this.gamma = gamma;
    }
}
