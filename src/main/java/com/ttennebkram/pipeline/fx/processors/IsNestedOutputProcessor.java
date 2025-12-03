package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import org.opencv.core.Mat;

/**
 * Is-Nested Output processor - routes frames based on execution context.
 *
 * - Out 1: Receives frames when running NESTED (inside a container)
 * - Out 2: Receives frames when running at ROOT level (top-level diagram)
 *
 * This allows pipelines to have conditional routing based on how they are executed.
 * When running standalone (root), output goes to Out 2.
 * When running as a sub-diagram inside a container, output goes to Out 1.
 *
 * If an output is not connected, frames are simply dropped (released).
 *
 * The isEmbedded flag is set at runtime by the executor when setting up
 * container internals, so the same saved pipeline can behave differently
 * depending on how it's being executed.
 */
@FXProcessorInfo(
    nodeType = "IsNestedOutput",
    displayName = "Is-Nested Output",
    category = "Nested Pipelines",
    description = "Routes based on context\nOut 1: if nested\nOut 2: if root",
    outputCount = 2,
    canBeDisabled = false
)
public class IsNestedOutputProcessor extends FXProcessorBase implements FXMultiOutputProcessor {

    private static final String[] OUTPUT_LABELS = {"Nested", "Root"};

    @Override
    public String getNodeType() {
        return "IsNestedOutput";
    }

    @Override
    public String getCategory() {
        return "Nested Pipelines";
    }

    @Override
    public String getDescription() {
        return "Routes frames based on execution context\n" +
               "Out 1 (Nested): if running inside a container\n" +
               "Out 2 (Root): if running at top level";
    }

    @Override
    public int getOutputCount() {
        return 2;
    }

    @Override
    public String[] getOutputLabels() {
        return OUTPUT_LABELS;
    }

    @Override
    public Mat[] processMultiOutput(Mat input) {
        Mat[] outputs = new Mat[2];

        if (input == null || input.empty()) {
            return outputs;  // Both null
        }

        // Check if we're running nested (inside a container)
        boolean isNested = true;  // Default to nested (safe assumption)
        if (fxNode != null) {
            isNested = fxNode.isEmbedded;
        }

        if (isNested) {
            // Running nested - output to Out 1 only
            outputs[0] = input.clone();
            outputs[1] = null;  // Nothing to Out 2
        } else {
            // Running at root level - output to Out 2 only
            outputs[0] = null;  // Nothing to Out 1
            outputs[1] = input.clone();
        }

        return outputs;
    }

    @Override
    public Mat process(Mat input) {
        // Default implementation returns first non-null output
        Mat[] outputs = processMultiOutput(input);
        if (outputs[0] != null) {
            if (outputs[1] != null) outputs[1].release();
            return outputs[0];
        }
        return outputs[1];
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription("Allows a pipeline to work both as a nested\n" +
            "sub-pipeline and as a standalone root pipeline.\n\n" +
            "Output 1 (Nested): Connect to Nested Pipeline Output.\n" +
            "  Used when this pipeline runs inside a Container,\n" +
            "  sending frames back to the parent pipeline.\n\n" +
            "Output 2 (Root): Connect to a Monitor or endpoint.\n" +
            "  Used when this pipeline runs standalone,\n" +
            "  for viewing/testing the results directly.\n\n" +
            "The node automatically routes to the correct output\n" +
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
        // No properties to sync - we use fxNode.isEmbedded directly in processMultiOutput()
    }

    @Override
    public void syncToFXNode(FXNode node) {
        // No properties to sync
    }
}
