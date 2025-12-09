package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import org.opencv.core.Mat;

/**
 * Webcam Source processor.
 * Configures webcam/camera input source properties.
 * Note: The actual camera capture is handled by the pipeline executor.
 */
@FXProcessorInfo(
    nodeType = "WebcamSource",
    displayName = "Webcam Source",
    category = "Sources",
    description = "Webcam capture\nVideoCapture.read()",
    isSource = true
)
public class WebcamSourceProcessor extends FXCameraProcessorBase {

    private CameraSettingsUI cameraUI;

    @Override
    public String getNodeType() {
        return "WebcamSource";
    }

    @Override
    public String getCategory() {
        return "Source";
    }

    @Override
    public String getDescription() {
        return "Webcam Source\nCaptures frames from camera";
    }

    @Override
    public Mat process(Mat input) {
        // Source nodes don't process - they generate
        // Actual capture is handled by the pipeline executor
        return input;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        cameraUI = addCameraSettingsUI(dialog, "");

        dialog.setOnOk(() -> {
            applyCameraSettingsFromUI(cameraUI);
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        serializeCameraProperties(json);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        deserializeCameraProperties(json);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        syncCameraPropertiesFromFXNode(node);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        syncCameraPropertiesToFXNode(node);
    }
}
