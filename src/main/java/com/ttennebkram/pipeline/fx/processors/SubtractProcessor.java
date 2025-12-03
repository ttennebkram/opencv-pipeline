package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import org.opencv.core.Core;
import org.opencv.core.Mat;

/**
 * Subtract processor - subtracts second image from first with clamping.
 * output = clamp(input1 - input2, 0, 255)
 *
 * Uses Core.subtract which automatically saturates/clamps the result (negative values become 0).
 */
@FXProcessorInfo(
    nodeType = "SubtractClamp",
    displayName = "Subtract",
    category = "Dual Input",
    description = "Subtract with saturation\nCore.subtract(src1, src2, dst)",
    dualInput = true
)
public class SubtractProcessor extends FXDualInputProcessor {

    @Override
    public String getNodeType() {
        return "SubtractClamp";
    }

    @Override
    public String getCategory() {
        return "Dual Input";
    }

    @Override
    public String getDescription() {
        return "Subtract\nCore.subtract(src1, src2, dst)\nResult saturated to 0-255";
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

        // Perform subtraction with saturation (clamp)
        Mat output = new Mat();
        Core.subtract(input1, prep.mat, output);

        // Clean up
        prep.releaseIfNeeded();

        return output;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());
        // No configurable properties - simple subtract
    }

    @Override
    public void serializeProperties(JsonObject json) {
        // No properties to serialize
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        // No properties to deserialize
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        // No properties to sync
    }

    @Override
    public void syncToFXNode(FXNode node) {
        // No properties to sync
    }
}
