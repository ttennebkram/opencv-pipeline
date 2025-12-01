package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.geometry.Insets;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Filter2D processor with custom convolution kernel.
 * Allows defining arbitrary convolution kernels via a grid UI.
 */
@FXProcessorInfo(nodeType = "Filter2D", category = "Filter")
public class Filter2DProcessor extends FXProcessorBase {

    // Properties with defaults
    private int kernelSize = 3;
    private int[] kernelValues;

    private static final String[] SIZE_OPTIONS = {"3x3", "5x5", "7x7", "9x9"};

    // Transient UI references (not serialized)
    private transient VBox gridContainer;
    private transient List<TextField> kernelFields;
    private transient FXPropertiesDialog currentDialog;

    public Filter2DProcessor() {
        // Default identity kernel
        kernelValues = new int[9];
        kernelValues[4] = 1; // Center element
    }

    @Override
    public String getNodeType() {
        return "Filter2D";
    }

    @Override
    public String getCategory() {
        return "Filter";
    }

    @Override
    public String getDescription() {
        return "Custom 2D Convolution Filter\nImgproc.filter2D(src, dst, -1, kernel)";
    }

    @Override
    public Mat process(Mat input) {
        if (isInvalidInput(input)) {
            return input;
        }

        // Build the kernel matrix
        Mat kernel = new Mat(kernelSize, kernelSize, CvType.CV_32F);
        if (kernelValues != null && kernelValues.length == kernelSize * kernelSize) {
            for (int i = 0; i < kernelSize; i++) {
                for (int j = 0; j < kernelSize; j++) {
                    kernel.put(i, j, (float) kernelValues[i * kernelSize + j]);
                }
            }
        } else {
            // Default identity kernel
            kernel.setTo(new Scalar(0));
            kernel.put(kernelSize / 2, kernelSize / 2, 1.0f);
        }

        Mat output = new Mat();
        Imgproc.filter2D(input, output, -1, kernel);
        kernel.release();

        return output;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());
        currentDialog = dialog;

        // Ensure kernelSize is valid
        if (kernelSize < 3) kernelSize = 3;
        if (kernelSize > 9) kernelSize = 9;
        if (kernelSize % 2 == 0) kernelSize++;

        // Ensure kernelValues array is correct size
        if (kernelValues == null || kernelValues.length != kernelSize * kernelSize) {
            kernelValues = new int[kernelSize * kernelSize];
            kernelValues[kernelSize * kernelSize / 2] = 1; // Identity
        }

        // Kernel size selector
        int sizeIndex = (kernelSize - 3) / 2;
        if (sizeIndex < 0 || sizeIndex > 3) sizeIndex = 0;
        ComboBox<String> sizeCombo = dialog.addComboBox("Kernel Size:", SIZE_OPTIONS, SIZE_OPTIONS[sizeIndex]);

        // Create a VBox to hold the kernel grid
        gridContainer = new VBox(5);
        gridContainer.setPadding(new Insets(10));
        dialog.addCustomContent(gridContainer);

        // Initialize kernel fields list
        kernelFields = new ArrayList<>();

        // Build the initial grid
        rebuildGrid();

        // Size change listener
        sizeCombo.setOnAction(e -> {
            int newSize = 3 + sizeCombo.getSelectionModel().getSelectedIndex() * 2;

            // Save current field values before rebuilding
            int[] oldValues = readFieldValues();
            int oldSize = kernelSize;

            // Resize values array (centered copy)
            kernelValues = resizeKernel(oldValues, oldSize, newSize);
            kernelSize = newSize;

            rebuildGrid();

            // Resize dialog to fit new grid
            if (currentDialog != null) {
                currentDialog.getDialogPane().getScene().getWindow().sizeToScene();
            }
        });

        // Preset buttons
        HBox presetBox = new HBox(10);
        presetBox.setPadding(new Insets(5, 0, 0, 0));

        Button identityBtn = new Button("Identity");
        identityBtn.setOnAction(e -> applyPreset("identity"));

        Button sharpenBtn = new Button("Sharpen/Edge");
        sharpenBtn.setOnAction(e -> applyPreset("sharpen"));

        Button boxBtn = new Button("Box Blur");
        boxBtn.setOnAction(e -> applyPreset("box"));

        presetBox.getChildren().addAll(new Label("Presets:"), identityBtn, sharpenBtn, boxBtn);
        dialog.addCustomContent(presetBox);

        // Save callback
        dialog.setOnOk(() -> {
            kernelValues = readFieldValues();
        });
    }

    private void rebuildGrid() {
        gridContainer.getChildren().clear();
        kernelFields.clear();

        GridPane grid = new GridPane();
        grid.setHgap(5);
        grid.setVgap(5);

        for (int i = 0; i < kernelSize; i++) {
            for (int j = 0; j < kernelSize; j++) {
                TextField field = new TextField();
                field.setPrefWidth(50);
                field.setStyle("-fx-alignment: center;");
                int idx = i * kernelSize + j;
                field.setText(idx < kernelValues.length ? String.valueOf(kernelValues[idx]) : "0");

                // Validate integer input
                field.textProperty().addListener((obs, oldVal, newVal) -> {
                    if (!newVal.isEmpty() && !newVal.equals("-")) {
                        try {
                            Integer.parseInt(newVal);
                        } catch (NumberFormatException ex) {
                            field.setText(oldVal);
                        }
                    }
                });

                grid.add(field, j, i);
                kernelFields.add(field);
            }
        }

        gridContainer.getChildren().add(grid);
    }

    private int[] readFieldValues() {
        int[] values = new int[kernelSize * kernelSize];
        for (int i = 0; i < kernelFields.size() && i < values.length; i++) {
            try {
                values[i] = Integer.parseInt(kernelFields.get(i).getText());
            } catch (NumberFormatException e) {
                values[i] = 0;
            }
        }
        return values;
    }

    private int[] resizeKernel(int[] oldValues, int oldSize, int newSize) {
        int[] newValues = new int[newSize * newSize];

        if (oldSize == newSize) {
            System.arraycopy(oldValues, 0, newValues, 0, Math.min(oldValues.length, newValues.length));
            return newValues;
        }

        // Copy centered
        int offset = (newSize - oldSize) / 2;
        for (int i = 0; i < oldSize && i < newSize; i++) {
            for (int j = 0; j < oldSize && j < newSize; j++) {
                int newI = i + (offset > 0 ? offset : 0);
                int newJ = j + (offset > 0 ? offset : 0);
                int oldI = i + (offset < 0 ? -offset : 0);
                int oldJ = j + (offset < 0 ? -offset : 0);
                if (newI < newSize && newJ < newSize && oldI < oldSize && oldJ < oldSize) {
                    newValues[newI * newSize + newJ] = oldValues[oldI * oldSize + oldJ];
                }
            }
        }

        return newValues;
    }

    private void applyPreset(String preset) {
        int center = kernelSize / 2;
        for (int i = 0; i < kernelSize; i++) {
            for (int j = 0; j < kernelSize; j++) {
                int idx = i * kernelSize + j;
                String value;
                switch (preset) {
                    case "identity":
                        value = (i == center && j == center) ? "1" : "0";
                        break;
                    case "sharpen":
                        int neighbors = (kernelSize - 1) * 2;
                        if (i == center && j == center) {
                            value = String.valueOf(neighbors + 1);
                        } else if (i == center || j == center) {
                            value = "-1";
                        } else {
                            value = "0";
                        }
                        break;
                    case "box":
                        value = "1";
                        break;
                    default:
                        value = "0";
                }
                if (idx < kernelFields.size()) {
                    kernelFields.get(idx).setText(value);
                }
            }
        }
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("kernelSize", kernelSize);
        JsonArray arr = new JsonArray();
        if (kernelValues != null) {
            for (int v : kernelValues) {
                arr.add(v);
            }
        }
        json.add("kernelValues", arr);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        kernelSize = getJsonInt(json, "kernelSize", 3);

        // Ensure valid size
        if (kernelSize < 3) kernelSize = 3;
        if (kernelSize > 9) kernelSize = 9;
        if (kernelSize % 2 == 0) kernelSize++;

        if (json.has("kernelValues") && json.get("kernelValues").isJsonArray()) {
            JsonArray arr = json.getAsJsonArray("kernelValues");
            kernelValues = new int[arr.size()];
            for (int i = 0; i < arr.size(); i++) {
                kernelValues[i] = arr.get(i).getAsInt();
            }
        } else {
            // Default identity kernel
            kernelValues = new int[kernelSize * kernelSize];
            kernelValues[kernelSize * kernelSize / 2] = 1;
        }
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        kernelSize = getInt(node.properties, "kernelSize", 3);

        // Ensure valid size
        if (kernelSize < 3) kernelSize = 3;
        if (kernelSize > 9) kernelSize = 9;
        if (kernelSize % 2 == 0) kernelSize++;

        Object kvObj = node.properties.get("kernelValues");
        if (kvObj instanceof int[]) {
            kernelValues = (int[]) kvObj;
        } else if (kvObj instanceof double[]) {
            double[] darr = (double[]) kvObj;
            kernelValues = new int[darr.length];
            for (int i = 0; i < darr.length; i++) {
                kernelValues[i] = (int) darr[i];
            }
        } else if (kvObj instanceof List) {
            @SuppressWarnings("unchecked")
            List<Number> list = (List<Number>) kvObj;
            kernelValues = new int[list.size()];
            for (int i = 0; i < list.size(); i++) {
                kernelValues[i] = list.get(i).intValue();
            }
        } else {
            // Default identity kernel
            kernelValues = new int[kernelSize * kernelSize];
            kernelValues[kernelSize * kernelSize / 2] = 1;
        }
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("kernelSize", kernelSize);
        node.properties.put("kernelValues", kernelValues);
    }
}
