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
