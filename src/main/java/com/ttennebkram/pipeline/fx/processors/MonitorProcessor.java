package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import org.opencv.core.Mat;

/**
 * Monitor/Passthrough processor.
 * Simply clones the input image - useful for monitoring a pipeline stage
 * without modifying the image.
 */
@FXProcessorInfo(
    nodeType = "Monitor",
    displayName = "Monitor/Passthrough",
    buttonName = "Monitor/Passthrough",
    category = "Utility",
    description = "Passthrough\nMat.clone()"
)
public class MonitorProcessor extends FXProcessorBase {

    @Override
    public String getNodeType() {
        return "Monitor";
    }

    @Override
    public String getCategory() {
        return "Utility";
    }

    @Override
    public String getDescription() {
        return "Monitor/Passthrough\nPasses through the input unchanged.\nUseful for viewing intermediate pipeline stages.";
    }

    @Override
    public Mat process(Mat input) {
        if (isInvalidInput(input)) {
            return input;
        }
        return input.clone();
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());
        dialog.addDescription("\nNo configurable properties.\n\n" +
            "Use this node to view the output of any pipeline stage\n" +
            "without modifying the image.");
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
