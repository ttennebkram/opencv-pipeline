package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Spinner;
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
public class WebcamSourceProcessor extends FXProcessorBase {

    // Properties with defaults
    private int cameraIndex = -1;
    private int resolutionIndex = 1;
    private boolean mirrorHorizontal = true;
    private int fpsIndex = 0;

    private static final String[] RESOLUTIONS = {"320x240", "640x480", "1280x720", "1920x1080"};
    private static final String[] FPS_OPTIONS = {"1 fps", "5 fps", "10 fps", "15 fps", "30 fps"};

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

        Spinner<Integer> camSpinner = dialog.addSpinner("Camera Index (-1=auto):", -1, 10, cameraIndex);
        ComboBox<String> resCombo = dialog.addComboBox("Resolution:", RESOLUTIONS,
                RESOLUTIONS[Math.min(resolutionIndex, RESOLUTIONS.length - 1)]);
        CheckBox mirrorCheck = dialog.addCheckbox("Mirror Horizontal", mirrorHorizontal);
        ComboBox<String> fpsCombo = dialog.addComboBox("FPS:", FPS_OPTIONS,
                FPS_OPTIONS[Math.min(fpsIndex, FPS_OPTIONS.length - 1)]);

        // Save callback
        dialog.setOnOk(() -> {
            cameraIndex = camSpinner.getValue();
            resolutionIndex = resCombo.getSelectionModel().getSelectedIndex();
            mirrorHorizontal = mirrorCheck.isSelected();
            fpsIndex = fpsCombo.getSelectionModel().getSelectedIndex();
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("cameraIndex", cameraIndex);
        json.addProperty("resolutionIndex", resolutionIndex);
        json.addProperty("mirrorHorizontal", mirrorHorizontal);
        json.addProperty("fpsIndex", fpsIndex);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        cameraIndex = getJsonInt(json, "cameraIndex", -1);
        resolutionIndex = getJsonInt(json, "resolutionIndex", 1);
        mirrorHorizontal = json.has("mirrorHorizontal") ? json.get("mirrorHorizontal").getAsBoolean() : true;
        fpsIndex = getJsonInt(json, "fpsIndex", 0);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        cameraIndex = getInt(node.properties, "cameraIndex", -1);
        resolutionIndex = getInt(node.properties, "resolutionIndex", 1);
        mirrorHorizontal = getBool(node.properties, "mirrorHorizontal", true);
        fpsIndex = getInt(node.properties, "fpsIndex", 0);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("cameraIndex", cameraIndex);
        node.properties.put("resolutionIndex", resolutionIndex);
        node.properties.put("mirrorHorizontal", mirrorHorizontal);
        node.properties.put("fpsIndex", fpsIndex);
    }

    private boolean getBool(java.util.Map<String, Object> props, String key, boolean defaultValue) {
        if (props.containsKey(key)) {
            Object val = props.get(key);
            if (val instanceof Boolean) return (Boolean) val;
        }
        return defaultValue;
    }
}
