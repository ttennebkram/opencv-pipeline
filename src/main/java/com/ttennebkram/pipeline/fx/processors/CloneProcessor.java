package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.ComboBox;
import org.opencv.core.Mat;

/**
 * Clone processor.
 * Clones/duplicates the input image (useful for multiple outputs).
 */
@FXProcessorInfo(
    nodeType = "Clone",
    displayName = "Clone",
    category = "Utility",
    description = "Clone to multiple outputs\nMat.clone()",
    outputCount = 2
)
public class CloneProcessor extends FXProcessorBase {

    // Properties with defaults
    private int numOutputs = 2;

    private static final String[] OUTPUT_OPTIONS = {"2", "3", "4"};

    @Override
    public String getNodeType() {
        return "Clone";
    }

    @Override
    public String getCategory() {
        return "Utility";
    }

    @Override
    public String getDescription() {
        return "Clone/Duplicate\nDuplicates the input image for multiple outputs";
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

        ComboBox<String> outputCombo = dialog.addComboBox("Number of Outputs:", OUTPUT_OPTIONS,
                OUTPUT_OPTIONS[Math.max(0, Math.min(numOutputs - 2, OUTPUT_OPTIONS.length - 1))]);

        // Save callback
        dialog.setOnOk(() -> {
            numOutputs = outputCombo.getSelectionModel().getSelectedIndex() + 2;
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("numOutputs", numOutputs);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        numOutputs = getJsonInt(json, "numOutputs", 2);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        numOutputs = getInt(node.properties, "numOutputs", 2);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("numOutputs", numOutputs);
    }
}
