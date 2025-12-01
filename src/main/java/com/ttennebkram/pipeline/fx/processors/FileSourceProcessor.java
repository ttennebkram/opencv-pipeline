package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.layout.HBox;
import javafx.stage.FileChooser;
import javafx.stage.Window;
import org.opencv.core.Mat;

import java.io.File;

/**
 * File Source processor.
 * Configures image/video file input source properties.
 * Note: The actual file loading is handled by the pipeline executor.
 */
@FXProcessorInfo(
    nodeType = "FileSource",
    displayName = "File Source",
    category = "Sources",
    description = "Image/Video file source\nImgcodecs.imread() / VideoCapture",
    isSource = true
)
public class FileSourceProcessor extends FXProcessorBase {

    // Properties with defaults
    private String imagePath = "";
    private int fpsMode = 1; // 0=just once, 1=auto, 2+=specific fps
    private boolean loopVideo = true;

    private static final String[] FPS_MODES = {
        "Just Once", "Automatic", "1 fps", "5 fps", "10 fps", "15 fps", "24 fps", "30 fps", "60 fps"
    };

    @Override
    public String getNodeType() {
        return "FileSource";
    }

    @Override
    public String getCategory() {
        return "Source";
    }

    @Override
    public String getDescription() {
        return "File Source\nLoads images or videos from file";
    }

    @Override
    public Mat process(Mat input) {
        // Source nodes don't process - they generate
        // Actual loading is handled by the pipeline executor
        return input;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        // Image/Video File with Browse button
        HBox pathRow = new HBox(10);
        Label pathLabel = new Label("Image/Video File:");
        TextField pathField = new TextField(imagePath);
        pathField.setPrefWidth(200);
        Button browseBtn = new Button("Browse...");
        browseBtn.setOnAction(e -> {
            FileChooser fileChooser = new FileChooser();
            fileChooser.setTitle("Select Image or Video File");
            fileChooser.getExtensionFilters().addAll(
                new FileChooser.ExtensionFilter("All Supported", "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif", "*.tiff", "*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"),
                new FileChooser.ExtensionFilter("Images", "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif", "*.tiff"),
                new FileChooser.ExtensionFilter("Videos", "*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"),
                new FileChooser.ExtensionFilter("All Files", "*.*")
            );
            // Set initial directory from current path if valid
            String currentPath = pathField.getText();
            if (currentPath != null && !currentPath.isEmpty()) {
                File currentFile = new File(currentPath);
                if (currentFile.getParentFile() != null && currentFile.getParentFile().exists()) {
                    fileChooser.setInitialDirectory(currentFile.getParentFile());
                }
            }
            Window window = dialog.getDialogPane().getScene().getWindow();
            File selectedFile = fileChooser.showOpenDialog(window);
            if (selectedFile != null) {
                pathField.setText(selectedFile.getAbsolutePath());
            }
        });
        pathRow.getChildren().addAll(pathLabel, pathField, browseBtn);
        dialog.addCustomContent(pathRow);

        // FPS Mode
        ComboBox<String> fpsCombo = dialog.addComboBox("FPS Mode:", FPS_MODES,
                FPS_MODES[Math.min(fpsMode, FPS_MODES.length - 1)]);

        // Loop Video
        CheckBox loopCheck = dialog.addCheckbox("Loop Video", loopVideo);

        // Save callback
        dialog.setOnOk(() -> {
            imagePath = pathField.getText();
            fpsMode = fpsCombo.getSelectionModel().getSelectedIndex();
            loopVideo = loopCheck.isSelected();
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("imagePath", imagePath);
        json.addProperty("fpsMode", fpsMode);
        json.addProperty("loopVideo", loopVideo);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        imagePath = json.has("imagePath") ? json.get("imagePath").getAsString() : "";
        fpsMode = getJsonInt(json, "fpsMode", 1);
        loopVideo = json.has("loopVideo") ? json.get("loopVideo").getAsBoolean() : true;
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        Object pathObj = node.properties.get("imagePath");
        imagePath = pathObj != null ? pathObj.toString() : "";
        fpsMode = getInt(node.properties, "fpsMode", 1);
        loopVideo = getBool(node.properties, "loopVideo", true);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("imagePath", imagePath);
        node.properties.put("fpsMode", fpsMode);
        node.properties.put("loopVideo", loopVideo);
    }

    private boolean getBool(java.util.Map<String, Object> props, String key, boolean defaultValue) {
        if (props.containsKey(key)) {
            Object val = props.get(key);
            if (val instanceof Boolean) return (Boolean) val;
        }
        return defaultValue;
    }
}
