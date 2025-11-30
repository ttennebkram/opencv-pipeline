package com.ttennebkram.pipeline.fx;

import javafx.geometry.Insets;
import javafx.scene.Node;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.stage.Modality;
import javafx.stage.Stage;
import javafx.stage.Window;

import java.util.Optional;

/**
 * JavaFX-based properties dialog for pipeline nodes.
 * Provides a standard dialog layout with:
 * - Type label (read-only)
 * - Name field
 * - Custom content area (populated by subclasses/callbacks)
 * - OK/Cancel buttons
 */
public class FXPropertiesDialog extends Dialog<Boolean> {

    private final TextField nameField;
    private final VBox contentArea;
    private Runnable onOk;

    /**
     * Create a new properties dialog.
     * @param owner The owner window (for modality)
     * @param title Dialog title
     * @param nodeType The node type name (displayed read-only)
     * @param currentName Current name of the node
     */
    public FXPropertiesDialog(Window owner, String title, String nodeType, String currentName) {
        setTitle(title);
        initOwner(owner);
        initModality(Modality.APPLICATION_MODAL);
        setResizable(true);

        // Create main content
        GridPane grid = new GridPane();
        grid.setHgap(10);
        grid.setVgap(10);
        grid.setPadding(new Insets(20));

        int row = 0;

        // Type label (read-only)
        grid.add(new Label("Type:"), 0, row);
        Label typeLabel = new Label(nodeType);
        typeLabel.setStyle("-fx-font-weight: bold;");
        grid.add(typeLabel, 1, row++);

        // Name field
        grid.add(new Label("Name:"), 0, row);
        nameField = new TextField(currentName);
        nameField.setPrefWidth(200);
        grid.add(nameField, 1, row++);

        // Separator
        Separator sep = new Separator();
        GridPane.setColumnSpan(sep, 2);
        grid.add(sep, 0, row++);

        // Content area for custom properties
        contentArea = new VBox(10);
        GridPane.setColumnSpan(contentArea, 2);
        grid.add(contentArea, 0, row++);

        getDialogPane().setContent(grid);

        // Set minimum size to ensure content is visible
        getDialogPane().setMinWidth(350);
        getDialogPane().setMinHeight(200);

        // Buttons
        ButtonType okButton = new ButtonType("OK", ButtonBar.ButtonData.OK_DONE);
        ButtonType cancelButton = new ButtonType("Cancel", ButtonBar.ButtonData.CANCEL_CLOSE);
        getDialogPane().getButtonTypes().addAll(okButton, cancelButton);

        // Result converter
        setResultConverter(buttonType -> buttonType == okButton);

        // Handle OK action
        final Button okBtn = (Button) getDialogPane().lookupButton(okButton);
        okBtn.addEventFilter(javafx.event.ActionEvent.ACTION, event -> {
            if (onOk != null) {
                onOk.run();
            }
        });
    }

    /**
     * Get the name field value.
     */
    public String getNameValue() {
        return nameField.getText().trim();
    }

    /**
     * Add a labeled row to the content area.
     * @param label The label text
     * @param control The control to add
     */
    public void addRow(String label, Node control) {
        HBox row = new HBox(10);
        row.getChildren().addAll(new Label(label), control);
        contentArea.getChildren().add(row);
    }

    /**
     * Add a description label spanning full width.
     * @param description The description text
     */
    public void addDescription(String description) {
        Label descLabel = new Label(description);
        descLabel.setStyle("-fx-text-fill: gray;");
        descLabel.setWrapText(true);
        contentArea.getChildren().add(descLabel);
    }

    /**
     * Add a slider with label display.
     * @param label The field label
     * @param min Minimum value
     * @param max Maximum value
     * @param currentValue Current value
     * @param formatString Format string for displaying value (e.g., "%.2f")
     * @return The slider for later value retrieval
     */
    public Slider addSlider(String label, double min, double max, double currentValue, String formatString) {
        HBox row = new HBox(10);
        row.getChildren().add(new Label(label));

        Slider slider = new Slider(min, max, currentValue);
        slider.setPrefWidth(200);
        slider.setShowTickMarks(true);
        slider.setShowTickLabels(true);

        // Configure sensible tick spacing based on range
        double range = max - min;
        if (range <= 10) {
            slider.setMajorTickUnit(1);
        } else if (range <= 50) {
            slider.setMajorTickUnit(10);
        } else if (range <= 100) {
            slider.setMajorTickUnit(25);
        } else if (range <= 255) {
            slider.setMajorTickUnit(50);
        } else {
            slider.setMajorTickUnit(range / 4);
        }
        slider.setMinorTickCount(0);

        Label valueLabel = new Label(String.format(formatString, currentValue));
        valueLabel.setMinWidth(50);

        slider.valueProperty().addListener((obs, oldVal, newVal) -> {
            valueLabel.setText(String.format(formatString, newVal.doubleValue()));
        });

        row.getChildren().addAll(slider, valueLabel);
        contentArea.getChildren().add(row);

        return slider;
    }

    /**
     * Add a slider for kernel sizes that must be odd values (1, 3, 5, ..., 31).
     * Snaps to odd values and shows only odd tick labels.
     * @param label The field label
     * @param currentValue Current value (will be rounded to nearest odd)
     * @return The slider for later value retrieval
     */
    public Slider addOddKernelSlider(String label, int currentValue) {
        HBox row = new HBox(10);
        row.getChildren().add(new Label(label));

        // Ensure current value is odd
        if (currentValue % 2 == 0) currentValue++;
        if (currentValue < 1) currentValue = 1;
        if (currentValue > 31) currentValue = 31;

        Slider slider = new Slider(1, 31, currentValue);
        slider.setPrefWidth(200);
        slider.setShowTickMarks(true);
        slider.setShowTickLabels(true);
        slider.setMajorTickUnit(10);
        slider.setMinorTickCount(4);
        slider.setSnapToTicks(true);
        slider.setBlockIncrement(2);  // Arrow keys move by 2

        Label valueLabel = new Label(String.valueOf(currentValue));
        valueLabel.setMinWidth(50);

        // Snap to odd values
        slider.valueProperty().addListener((obs, oldVal, newVal) -> {
            int val = newVal.intValue();
            if (val % 2 == 0) {
                // Snap to nearest odd
                val = val < oldVal.intValue() ? val - 1 : val + 1;
                if (val < 1) val = 1;
                if (val > 31) val = 31;
                slider.setValue(val);
            }
            valueLabel.setText(String.valueOf((int) slider.getValue()));
        });

        row.getChildren().addAll(slider, valueLabel);
        contentArea.getChildren().add(row);

        return slider;
    }

    /**
     * Add a spinner for integer values.
     * @param label The field label
     * @param min Minimum value
     * @param max Maximum value
     * @param currentValue Current value
     * @return The spinner for later value retrieval
     */
    public Spinner<Integer> addSpinner(String label, int min, int max, int currentValue) {
        HBox row = new HBox(10);
        row.getChildren().add(new Label(label));

        Spinner<Integer> spinner = new Spinner<>(min, max, currentValue);
        spinner.setEditable(true);
        spinner.setPrefWidth(100);

        row.getChildren().add(spinner);
        contentArea.getChildren().add(row);

        return spinner;
    }

    /**
     * Add a text field.
     * @param label The field label
     * @param currentValue Current value
     * @return The text field for later value retrieval
     */
    public TextField addTextField(String label, String currentValue) {
        HBox row = new HBox(10);
        row.getChildren().add(new Label(label));

        TextField textField = new TextField(currentValue);
        textField.setPrefWidth(200);

        row.getChildren().add(textField);
        contentArea.getChildren().add(row);

        return textField;
    }

    /**
     * Add a checkbox.
     * @param label The checkbox label
     * @param currentValue Current checked state
     * @return The checkbox for later value retrieval
     */
    public CheckBox addCheckbox(String label, boolean currentValue) {
        CheckBox checkBox = new CheckBox(label);
        checkBox.setSelected(currentValue);
        contentArea.getChildren().add(checkBox);
        return checkBox;
    }

    /**
     * Add a combo box (dropdown).
     * @param label The field label
     * @param options Available options
     * @param currentValue Current selection
     * @return The combo box for later value retrieval
     */
    public <T> ComboBox<T> addComboBox(String label, T[] options, T currentValue) {
        HBox row = new HBox(10);
        row.getChildren().add(new Label(label));

        ComboBox<T> comboBox = new ComboBox<>();
        comboBox.getItems().addAll(options);
        comboBox.setValue(currentValue);

        row.getChildren().add(comboBox);
        contentArea.getChildren().add(row);

        return comboBox;
    }

    /**
     * Add horizontal radio buttons for selecting from a small set of options.
     * @param label The field label
     * @param options Available options (displayed as button labels)
     * @param currentIndex Currently selected index (0-based)
     * @return The ToggleGroup for later value retrieval via getSelectedToggle()
     */
    public ToggleGroup addRadioButtons(String label, String[] options, int currentIndex) {
        HBox row = new HBox(10);
        row.getChildren().add(new Label(label));

        ToggleGroup group = new ToggleGroup();
        HBox buttonsBox = new HBox(8);

        for (int i = 0; i < options.length; i++) {
            RadioButton rb = new RadioButton(options[i]);
            rb.setToggleGroup(group);
            rb.setUserData(i);  // Store the index as user data
            if (i == currentIndex) {
                rb.setSelected(true);
            }
            buttonsBox.getChildren().add(rb);
        }

        row.getChildren().add(buttonsBox);
        contentArea.getChildren().add(row);

        return group;
    }

    /**
     * Add a custom node to the content area.
     * @param node The node to add
     */
    public void addCustomContent(Node node) {
        contentArea.getChildren().add(node);
    }

    /**
     * Set a callback to run when OK is clicked.
     * Use this to save values back to the node.
     * @param onOk The callback
     */
    public void setOnOk(Runnable onOk) {
        this.onOk = onOk;
    }

    /**
     * Show the dialog and wait for user response.
     * @return true if OK was clicked, false otherwise
     */
    public boolean showAndWaitForResult() {
        Optional<Boolean> result = showAndWait();
        return result.orElse(false);
    }
}
