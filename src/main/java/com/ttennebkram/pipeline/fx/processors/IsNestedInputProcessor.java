package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import org.opencv.core.Mat;

/**
 * Is-Nested Input processor - routes frames based on execution context.
 *
 * - When running NESTED (inside a container/subprocess): passes Input 1 to output
 * - When running at ROOT level (top-level diagram): passes Input 2 to output
 *
 * This allows pipelines to behave differently when used standalone vs. when
 * nested as a sub-pipeline inside a container node.
 *
 * The isEmbedded flag is set at runtime by the executor when setting up
 * container internals, so the same saved pipeline can behave differently
 * depending on how it's being executed.
 */
@FXProcessorInfo(
    nodeType = "IsNestedInput",
    displayName = "Is-Nested Input",
    category = "Nested Pipelines",
    description = "Route by execution context\nNested: Input1 -> Out\nRoot: Input2 -> Out",
    dualInput = true,
    canBeDisabled = false
)
public class IsNestedInputProcessor extends FXDualInputProcessor {

    @Override
    public String getNodeType() {
        return "IsNestedInput";
    }

    @Override
    public String getCategory() {
        return "Nested Pipelines";
    }

    @Override
    public String getDescription() {
        return "Route by execution context\n" +
               "If nested (in container): Input1 -> Output\n" +
               "If root (top-level): Input2 -> Output";
    }

    @Override
    public Mat processDual(Mat input1, Mat input2) {
        // Check if we're running nested (inside a container)
        boolean isNested = true;  // Default to nested (safe assumption)
        if (fxNode != null) {
            isNested = fxNode.isEmbedded;
        }

        if (isNested) {
            // Running inside a container - pass through Input 1
            if (input1 != null) {
                return input1.clone();
            }
            return null;
        } else {
            // Running at root level - pass through Input 2
            if (input2 != null) {
                return input2.clone();
            }
            return null;
        }
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription("Allows a pipeline to work both as a nested\n" +
            "sub-pipeline and as a standalone root pipeline.\n\n" +
            "Input 1 (Nested): Connect to Nested Pipeline Input.\n" +
            "  Used when this pipeline runs inside a Container,\n" +
            "  receiving frames from the parent pipeline.\n\n" +
            "Input 2 (Root): Connect to a source (File, Webcam).\n" +
            "  Used when this pipeline runs standalone,\n" +
            "  getting frames directly from the source.\n\n" +
            "The node automatically selects the correct input\n" +
            "based on execution context.");
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
        // No properties to sync - we use fxNode.isEmbedded directly in processDual()
    }

    @Override
    public void syncToFXNode(FXNode node) {
        // No properties to sync
    }
}
