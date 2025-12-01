package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import org.opencv.core.Core;
import org.opencv.core.Mat;

/**
 * Invert processor.
 * Inverts (negates) all pixel values in the image.
 * This is a simple processor with no configurable properties.
 */
@FXProcessorInfo(nodeType = "Invert", category = "Basic")
public class InvertProcessor extends FXProcessorBase {

    @Override
    public String getNodeType() {
        return "Invert";
    }

    @Override
    public String getCategory() {
        return "Basic";
    }

    @Override
    public String getDescription() {
        return "Invert (Negate)\nCore.bitwise_not(src, dst)";
    }

    @Override
    public Mat process(Mat input) {
        if (isInvalidInput(input)) {
            return input;
        }

        Mat output = new Mat();
        Core.bitwise_not(input, output);
        return output;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());
        // No configurable properties for Invert
    }

    @Override
    public boolean hasProperties() {
        // Invert has no configurable properties, but we still show the dialog
        // with the description for consistency
        return true;
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
