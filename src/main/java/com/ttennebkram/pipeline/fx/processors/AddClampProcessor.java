package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import org.opencv.core.Core;
import org.opencv.core.Mat;

/**
 * Add with Clamp processor - adds two images with automatic clamping to 0-255.
 * output = clamp(input1 + input2, 0, 255)
 *
 * Uses Core.add which automatically saturates/clamps the result.
 */
@FXProcessorInfo(
    nodeType = "AddClamp",
    displayName = "Add w/Clamp",
    category = "Dual Input",
    description = "Add with saturation/clamp\nCore.add(src1, src2, dst)",
    dualInput = true
)
public class AddClampProcessor extends FXDualInputProcessor {

    @Override
    public String getNodeType() {
        return "AddClamp";
    }

    @Override
    public String getCategory() {
        return "Dual Input";
    }

    @Override
    public String getDescription() {
        return "Add with Clamp\nCore.add(src1, src2, dst)\nResult saturated to 0-255";
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

        // Perform addition with saturation (clamp)
        Mat output = new Mat();
        Core.add(input1, prep.mat, output);

        // Clean up
        prep.releaseIfNeeded();

        return output;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());
        // No configurable properties - simple add with clamp
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
