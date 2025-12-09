package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Spinner;

import java.util.Map;

/**
 * Base class for processors that use camera input.
 * Provides common camera settings (index, resolution, FPS, mirror)
 * and serialization/deserialization support.
 */
public abstract class FXCameraProcessorBase extends FXProcessorBase {

    // Camera settings with defaults
    protected int cameraIndex = -1;         // -1 = auto-detect
    protected int resolutionIndex = 1;       // 640x480
    protected int fpsIndex = 2;              // 10 fps
    protected boolean mirrorHorizontal = true;

    // Common constants
    protected static final String[] RESOLUTIONS = {"320x240", "640x480", "1280x720", "1920x1080"};
    protected static final String[] FPS_OPTIONS = {"1 fps", "5 fps", "10 fps", "15 fps", "30 fps"};
    protected static final double[] FPS_VALUES = {1.0, 5.0, 10.0, 15.0, 30.0};

    /**
     * Get the actual FPS value from the index.
     */
    public double getCameraFps() {
        if (fpsIndex >= 0 && fpsIndex < FPS_VALUES.length) {
            return FPS_VALUES[fpsIndex];
        }
        return 10.0;  // default
    }

    /**
     * Set FPS by value (finds closest index).
     */
    public void setCameraFps(double fps) {
        int closest = 2;  // default to 10 fps
        double minDiff = Double.MAX_VALUE;
        for (int i = 0; i < FPS_VALUES.length; i++) {
            double diff = Math.abs(FPS_VALUES[i] - fps);
            if (diff < minDiff) {
                minDiff = diff;
                closest = i;
            }
        }
        fpsIndex = closest;
    }

    // Getters for pipeline executor
    public int getCameraIndex() { return cameraIndex; }
    public int getResolutionIndex() { return resolutionIndex; }
    public int getFpsIndex() { return fpsIndex; }
    public boolean isMirrorHorizontal() { return mirrorHorizontal; }

    // Setters
    public void setCameraIndex(int index) { this.cameraIndex = index; }
    public void setResolutionIndex(int index) { this.resolutionIndex = index; }
    public void setFpsIndex(int index) { this.fpsIndex = index; }
    public void setMirrorHorizontal(boolean mirror) { this.mirrorHorizontal = mirror; }

    /**
     * Parse resolution string (e.g., "1280x720") and return width.
     */
    public int getResolutionWidth() {
        String res = RESOLUTIONS[Math.min(resolutionIndex, RESOLUTIONS.length - 1)];
        return Integer.parseInt(res.split("x")[0]);
    }

    /**
     * Parse resolution string and return height.
     */
    public int getResolutionHeight() {
        String res = RESOLUTIONS[Math.min(resolutionIndex, RESOLUTIONS.length - 1)];
        return Integer.parseInt(res.split("x")[1]);
    }

    /**
     * Add camera settings UI to a properties dialog.
     * Returns the created UI components for subclass customization.
     */
    protected CameraSettingsUI addCameraSettingsUI(FXPropertiesDialog dialog) {
        return addCameraSettingsUI(dialog, "Camera Settings");
    }

    /**
     * Add camera settings UI with custom section title.
     */
    protected CameraSettingsUI addCameraSettingsUI(FXPropertiesDialog dialog, String sectionTitle) {
        if (sectionTitle != null && !sectionTitle.isEmpty()) {
            dialog.addDescription("\n" + sectionTitle);
        }

        Spinner<Integer> camSpinner = dialog.addSpinner("Camera Index (-1=auto):", -1, 10, cameraIndex);
        ComboBox<String> resCombo = dialog.addComboBox("Resolution:", RESOLUTIONS,
                RESOLUTIONS[Math.min(resolutionIndex, RESOLUTIONS.length - 1)]);
        CheckBox mirrorCheck = dialog.addCheckbox("Mirror Horizontal", mirrorHorizontal);
        ComboBox<String> fpsCombo = dialog.addComboBox("FPS:", FPS_OPTIONS,
                FPS_OPTIONS[Math.min(fpsIndex, FPS_OPTIONS.length - 1)]);

        return new CameraSettingsUI(camSpinner, resCombo, mirrorCheck, fpsCombo);
    }

    /**
     * Apply camera settings from UI components.
     */
    protected void applyCameraSettingsFromUI(CameraSettingsUI ui) {
        cameraIndex = ui.camSpinner.getValue();
        resolutionIndex = ui.resCombo.getSelectionModel().getSelectedIndex();
        mirrorHorizontal = ui.mirrorCheck.isSelected();
        fpsIndex = ui.fpsCombo.getSelectionModel().getSelectedIndex();
    }

    /**
     * Serialize camera properties to JSON.
     */
    protected void serializeCameraProperties(JsonObject json) {
        json.addProperty("cameraIndex", cameraIndex);
        json.addProperty("resolutionIndex", resolutionIndex);
        json.addProperty("mirrorHorizontal", mirrorHorizontal);
        json.addProperty("fpsIndex", fpsIndex);
    }

    /**
     * Deserialize camera properties from JSON.
     */
    protected void deserializeCameraProperties(JsonObject json) {
        cameraIndex = getJsonInt(json, "cameraIndex", -1);
        resolutionIndex = getJsonInt(json, "resolutionIndex", 1);
        mirrorHorizontal = json.has("mirrorHorizontal") ? json.get("mirrorHorizontal").getAsBoolean() : true;
        fpsIndex = getJsonInt(json, "fpsIndex", 2);

        // Handle legacy cameraFps property (convert to fpsIndex)
        if (json.has("cameraFps") && !json.has("fpsIndex")) {
            setCameraFps(json.get("cameraFps").getAsDouble());
        }
    }

    /**
     * Sync camera properties from FXNode.
     */
    protected void syncCameraPropertiesFromFXNode(FXNode node) {
        cameraIndex = getInt(node.properties, "cameraIndex", -1);
        resolutionIndex = getInt(node.properties, "resolutionIndex", 1);
        mirrorHorizontal = getBool(node.properties, "mirrorHorizontal", true);
        fpsIndex = getInt(node.properties, "fpsIndex", 2);
    }

    /**
     * Sync camera properties to FXNode.
     */
    protected void syncCameraPropertiesToFXNode(FXNode node) {
        node.properties.put("cameraIndex", cameraIndex);
        node.properties.put("resolutionIndex", resolutionIndex);
        node.properties.put("mirrorHorizontal", mirrorHorizontal);
        node.properties.put("fpsIndex", fpsIndex);
    }

    // Helper methods
    protected boolean getBool(Map<String, Object> props, String key, boolean defaultValue) {
        if (props.containsKey(key)) {
            Object val = props.get(key);
            if (val instanceof Boolean) return (Boolean) val;
        }
        return defaultValue;
    }

    protected double getDouble(Map<String, Object> props, String key, double defaultValue) {
        if (props.containsKey(key)) {
            Object val = props.get(key);
            if (val instanceof Number) return ((Number) val).doubleValue();
        }
        return defaultValue;
    }

    /**
     * Container for camera settings UI components.
     */
    protected static class CameraSettingsUI {
        public final Spinner<Integer> camSpinner;
        public final ComboBox<String> resCombo;
        public final CheckBox mirrorCheck;
        public final ComboBox<String> fpsCombo;

        public CameraSettingsUI(Spinner<Integer> camSpinner, ComboBox<String> resCombo,
                               CheckBox mirrorCheck, ComboBox<String> fpsCombo) {
            this.camSpinner = camSpinner;
            this.resCombo = resCombo;
            this.mirrorCheck = mirrorCheck;
            this.fpsCombo = fpsCombo;
        }
    }
}
