package com.ttennebkram.pipeline;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.geometry.Orientation;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.*;
import javafx.scene.image.ImageView;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyCodeCombination;
import javafx.scene.input.KeyCombination;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.stage.FileChooser;
import javafx.stage.Screen;
import javafx.stage.Stage;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.prefs.Preferences;

/**
 * Main JavaFX Application class for the Pipeline Editor.
 * This replaces the SWT-based PipelineEditor.
 */
public class PipelineEditorApp extends Application {

    // =========================== COLOR CONSTANTS ===========================
    private static final Color COLOR_TOOLBAR_BG = Color.rgb(160, 200, 160);
    private static final Color COLOR_BUTTON_NORMAL = Color.rgb(160, 160, 160);
    private static final Color COLOR_BUTTON_SELECTED = Color.rgb(100, 150, 255);
    private static final Color COLOR_STATUS_BAR_BG = Color.rgb(160, 160, 160);
    private static final Color COLOR_STATUS_STOPPED = Color.rgb(180, 0, 0);
    private static final Color COLOR_STATUS_RUNNING = Color.rgb(0, 128, 0);
    private static final Color COLOR_START_BUTTON = Color.rgb(100, 180, 100);
    private static final Color COLOR_STOP_BUTTON = Color.rgb(200, 100, 100);
    private static final Color COLOR_GRID_LINES = Color.rgb(230, 230, 230);
    private static final Color COLOR_SELECTION_BOX = Color.rgb(0, 120, 215);
    // ========================================================================

    private Stage primaryStage;
    private Canvas pipelineCanvas;
    private ScrollPane canvasScrollPane;
    private ImageView previewImageView;
    private Label statusBar;
    private Label nodeCountLabel;
    private ComboBox<String> zoomCombo;
    private TextField searchBox;
    private VBox toolbarContent;
    private Button startStopBtn;

    private double zoomLevel = 1.0;
    private static final int[] ZOOM_LEVELS = {25, 50, 75, 100, 125, 150, 200, 300, 400};

    // Pipeline state
    // TODO: These will be moved to a separate PipelineModel class
    // private List<PipelineNode> nodes = new ArrayList<>();
    // private List<Connection> connections = new ArrayList<>();
    private boolean pipelineRunning = false;
    private boolean isDirty = false;
    private String currentFilePath = null;

    // Recent files
    private static final int MAX_RECENT_FILES = 10;
    private static final String RECENT_FILES_KEY = "recentFiles";
    private static final String LAST_FILE_KEY = "lastFile";
    private Preferences prefs;
    private List<String> recentFiles = new ArrayList<>();
    private Menu openRecentMenu;

    @Override
    public void start(Stage primaryStage) {
        this.primaryStage = primaryStage;

        // Initialize preferences
        prefs = Preferences.userNodeForPackage(PipelineEditorApp.class);
        loadRecentFiles();

        // Create main layout
        BorderPane root = new BorderPane();

        // Menu bar at top
        root.setTop(createMenuBar());

        // Toolbar on left
        root.setLeft(createToolbar());

        // Main content - SplitPane with canvas and preview
        SplitPane splitPane = new SplitPane();
        splitPane.setOrientation(Orientation.HORIZONTAL);
        splitPane.getItems().addAll(createCanvasPane(), createPreviewPane());
        splitPane.setDividerPositions(0.75);
        root.setCenter(splitPane);

        // Status bar at bottom
        root.setBottom(createStatusBar());

        // Create scene
        Scene scene = new Scene(root, 1400, 800);

        // Apply any CSS styling
        // scene.getStylesheets().add(getClass().getResource("/styles.css").toExternalForm());

        primaryStage.setTitle("OpenCV Pipeline Editor - (untitled)");
        primaryStage.setScene(scene);

        // Center on screen
        primaryStage.centerOnScreen();

        // Handle close request
        primaryStage.setOnCloseRequest(event -> {
            if (!checkUnsavedChanges()) {
                event.consume(); // Cancel close
            } else {
                stopPipeline();
            }
        });

        primaryStage.show();

        // Initial canvas paint
        Platform.runLater(this::paintCanvas);

        System.out.println("JavaFX Pipeline Editor started");
        System.out.println("OpenCV version: " + org.opencv.core.Core.VERSION);
    }

    private MenuBar createMenuBar() {
        MenuBar menuBar = new MenuBar();

        // File menu
        Menu fileMenu = new Menu("File");

        MenuItem newItem = new MenuItem("New");
        newItem.setAccelerator(new KeyCodeCombination(KeyCode.N, KeyCombination.SHORTCUT_DOWN));
        newItem.setOnAction(e -> newDiagram());

        MenuItem openItem = new MenuItem("Open...");
        openItem.setAccelerator(new KeyCodeCombination(KeyCode.O, KeyCombination.SHORTCUT_DOWN));
        openItem.setOnAction(e -> loadDiagram());

        openRecentMenu = new Menu("Open Recent");
        updateOpenRecentMenu();

        MenuItem saveItem = new MenuItem("Save");
        saveItem.setAccelerator(new KeyCodeCombination(KeyCode.S, KeyCombination.SHORTCUT_DOWN));
        saveItem.setOnAction(e -> saveDiagram());

        MenuItem saveAsItem = new MenuItem("Save As...");
        saveAsItem.setAccelerator(new KeyCodeCombination(KeyCode.S, KeyCombination.SHORTCUT_DOWN, KeyCombination.SHIFT_DOWN));
        saveAsItem.setOnAction(e -> saveDiagramAs());

        fileMenu.getItems().addAll(newItem, openItem, openRecentMenu, new SeparatorMenuItem(), saveItem, saveAsItem);

        // Edit menu
        Menu editMenu = new Menu("Edit");

        MenuItem deleteItem = new MenuItem("Delete");
        deleteItem.setAccelerator(new KeyCodeCombination(KeyCode.BACK_SPACE));
        deleteItem.setOnAction(e -> deleteSelected());

        MenuItem selectAllItem = new MenuItem("Select All");
        selectAllItem.setAccelerator(new KeyCodeCombination(KeyCode.A, KeyCombination.SHORTCUT_DOWN));
        selectAllItem.setOnAction(e -> selectAll());

        editMenu.getItems().addAll(deleteItem, selectAllItem);

        // Help menu
        Menu helpMenu = new Menu("Help");
        MenuItem aboutItem = new MenuItem("About");
        aboutItem.setOnAction(e -> showAbout());
        helpMenu.getItems().add(aboutItem);

        menuBar.getMenus().addAll(fileMenu, editMenu, helpMenu);

        // macOS-specific menu bar integration
        menuBar.setUseSystemMenuBar(true);

        return menuBar;
    }

    private VBox createToolbar() {
        VBox toolbar = new VBox(5);
        toolbar.setPadding(new Insets(10));
        toolbar.setStyle("-fx-background-color: rgb(160, 200, 160);");
        toolbar.setPrefWidth(200);

        // Search box
        searchBox = new TextField();
        searchBox.setPromptText("Search nodes...");
        searchBox.textProperty().addListener((obs, oldVal, newVal) -> filterToolbarButtons());
        toolbar.getChildren().add(searchBox);

        // Scrollable content for node buttons
        ScrollPane scrollPane = new ScrollPane();
        scrollPane.setFitToWidth(true);
        scrollPane.setVbarPolicy(ScrollPane.ScrollBarPolicy.AS_NEEDED);
        scrollPane.setHbarPolicy(ScrollPane.ScrollBarPolicy.NEVER);
        VBox.setVgrow(scrollPane, Priority.ALWAYS);

        toolbarContent = new VBox(3);
        toolbarContent.setPadding(new Insets(5));
        scrollPane.setContent(toolbarContent);

        // Add node category buttons
        // TODO: Use NodeRegistry to populate these dynamically
        addToolbarCategory("Source");
        addToolbarButton("Webcam", () -> addNodeAt("WebcamSourceNode", 100, 100));
        addToolbarButton("File", () -> addNodeAt("FileSourceNode", 100, 100));
        addToolbarButton("Blank", () -> addNodeAt("BlankSourceNode", 100, 100));

        addToolbarCategory("Basic");
        addToolbarButton("Grayscale", () -> addNodeAt("GrayscaleNode", 100, 100));
        addToolbarButton("Invert", () -> addNodeAt("InvertNode", 100, 100));
        addToolbarButton("Threshold", () -> addNodeAt("ThresholdNode", 100, 100));

        addToolbarCategory("Blur");
        addToolbarButton("Gaussian", () -> addNodeAt("GaussianBlurNode", 100, 100));
        addToolbarButton("Median", () -> addNodeAt("MedianBlurNode", 100, 100));

        addToolbarCategory("Edge Detection");
        addToolbarButton("Canny", () -> addNodeAt("CannyEdgeNode", 100, 100));
        addToolbarButton("Sobel", () -> addNodeAt("SobelNode", 100, 100));

        toolbar.getChildren().add(scrollPane);

        // Pipeline control buttons
        HBox controlBox = new HBox(5);
        controlBox.setPadding(new Insets(10, 0, 0, 0));

        startStopBtn = new Button("Start Pipeline");
        startStopBtn.setStyle("-fx-background-color: rgb(100, 180, 100);");
        startStopBtn.setOnAction(e -> togglePipeline());
        startStopBtn.setMaxWidth(Double.MAX_VALUE);
        HBox.setHgrow(startStopBtn, Priority.ALWAYS);

        controlBox.getChildren().add(startStopBtn);
        toolbar.getChildren().add(controlBox);

        return toolbar;
    }

    private void addToolbarCategory(String name) {
        Label label = new Label(name);
        label.setStyle("-fx-font-weight: bold; -fx-padding: 5 0 2 0;");
        toolbarContent.getChildren().add(label);
    }

    private void addToolbarButton(String name, Runnable action) {
        Button btn = new Button(name);
        btn.setMaxWidth(Double.MAX_VALUE);
        btn.setOnAction(e -> action.run());
        toolbarContent.getChildren().add(btn);
    }

    private ScrollPane createCanvasPane() {
        // Create canvas with initial size
        pipelineCanvas = new Canvas(2000, 2000);

        // Wrap in a Pane for proper sizing
        Pane canvasContainer = new Pane(pipelineCanvas);
        canvasContainer.setStyle("-fx-background-color: white;");

        // Wrap in ScrollPane
        canvasScrollPane = new ScrollPane(canvasContainer);
        canvasScrollPane.setPannable(true);

        // Mouse event handlers
        pipelineCanvas.setOnMousePressed(this::handleMousePressed);
        pipelineCanvas.setOnMouseDragged(this::handleMouseDragged);
        pipelineCanvas.setOnMouseReleased(this::handleMouseReleased);
        pipelineCanvas.setOnMouseClicked(e -> {
            if (e.getClickCount() == 2) {
                handleDoubleClick(e);
            }
        });

        // Context menu
        ContextMenu contextMenu = new ContextMenu();
        MenuItem deleteMenuItem = new MenuItem("Delete");
        deleteMenuItem.setOnAction(e -> deleteSelected());
        contextMenu.getItems().add(deleteMenuItem);
        pipelineCanvas.setOnContextMenuRequested(e ->
            contextMenu.show(pipelineCanvas, e.getScreenX(), e.getScreenY()));

        return canvasScrollPane;
    }

    private VBox createPreviewPane() {
        VBox previewPane = new VBox(5);
        previewPane.setPadding(new Insets(10));
        previewPane.setStyle("-fx-background-color: #f0f0f0;");

        Label previewLabel = new Label("Preview");
        previewLabel.setStyle("-fx-font-weight: bold;");

        previewImageView = new ImageView();
        previewImageView.setPreserveRatio(true);
        previewImageView.setFitWidth(300);
        previewImageView.setFitHeight(300);

        // Placeholder for when no image
        StackPane imageContainer = new StackPane(previewImageView);
        imageContainer.setStyle("-fx-background-color: #cccccc; -fx-min-height: 200;");
        VBox.setVgrow(imageContainer, Priority.ALWAYS);

        // Zoom controls
        HBox zoomBox = new HBox(5);
        zoomBox.setPadding(new Insets(5, 0, 0, 0));

        Label zoomLabel = new Label("Zoom:");
        zoomCombo = new ComboBox<>();
        for (int level : ZOOM_LEVELS) {
            zoomCombo.getItems().add(level + "%");
        }
        zoomCombo.setValue("100%");
        zoomCombo.setOnAction(e -> {
            String selected = zoomCombo.getValue();
            if (selected != null) {
                zoomLevel = Integer.parseInt(selected.replace("%", "")) / 100.0;
                paintCanvas();
            }
        });

        zoomBox.getChildren().addAll(zoomLabel, zoomCombo);

        previewPane.getChildren().addAll(previewLabel, imageContainer, zoomBox);

        return previewPane;
    }

    private HBox createStatusBar() {
        HBox statusBarBox = new HBox(10);
        statusBarBox.setPadding(new Insets(5, 10, 5, 10));
        statusBarBox.setStyle("-fx-background-color: rgb(160, 160, 160);");

        statusBar = new Label("Pipeline stopped");
        statusBar.setTextFill(COLOR_STATUS_STOPPED);

        Region spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);

        nodeCountLabel = new Label("Nodes: 0");

        statusBarBox.getChildren().addAll(statusBar, spacer, nodeCountLabel);

        return statusBarBox;
    }

    // ========================= Canvas Painting =========================

    private void paintCanvas() {
        GraphicsContext gc = pipelineCanvas.getGraphicsContext2D();

        // Clear background
        gc.setFill(Color.WHITE);
        gc.fillRect(0, 0, pipelineCanvas.getWidth(), pipelineCanvas.getHeight());

        // Draw grid
        gc.setStroke(COLOR_GRID_LINES);
        gc.setLineWidth(1);
        double gridSize = 20 * zoomLevel;
        for (double x = 0; x < pipelineCanvas.getWidth(); x += gridSize) {
            gc.strokeLine(x, 0, x, pipelineCanvas.getHeight());
        }
        for (double y = 0; y < pipelineCanvas.getHeight(); y += gridSize) {
            gc.strokeLine(0, y, pipelineCanvas.getWidth(), y);
        }

        // TODO: Draw nodes and connections
        // This will be implemented as we migrate the node rendering

        // Placeholder text
        gc.setFill(Color.BLACK);
        gc.fillText("Pipeline Canvas - JavaFX Migration in Progress", 50, 50);
        gc.fillText("Nodes and connections will render here", 50, 70);
    }

    // ========================= Mouse Handlers =========================

    private void handleMousePressed(javafx.scene.input.MouseEvent e) {
        // TODO: Implement node selection and dragging
    }

    private void handleMouseDragged(javafx.scene.input.MouseEvent e) {
        // TODO: Implement node dragging and connection drawing
    }

    private void handleMouseReleased(javafx.scene.input.MouseEvent e) {
        // TODO: Implement connection completion
    }

    private void handleDoubleClick(javafx.scene.input.MouseEvent e) {
        // TODO: Open properties dialog for double-clicked node
    }

    // ========================= Actions =========================

    private void newDiagram() {
        if (checkUnsavedChanges()) {
            // Clear nodes and connections
            // nodes.clear();
            // connections.clear();
            currentFilePath = null;
            isDirty = false;
            updateTitle();
            paintCanvas();
        }
    }

    private void loadDiagram() {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Open Pipeline");
        fileChooser.getExtensionFilters().add(
            new FileChooser.ExtensionFilter("Pipeline Files", "*.json"));
        File file = fileChooser.showOpenDialog(primaryStage);
        if (file != null) {
            loadDiagramFromPath(file.getAbsolutePath());
        }
    }

    private void loadDiagramFromPath(String path) {
        // TODO: Implement pipeline loading using PipelineSerializer
        currentFilePath = path;
        isDirty = false;
        addToRecentFiles(path);
        updateTitle();
        paintCanvas();
    }

    private void saveDiagram() {
        if (currentFilePath != null) {
            saveDiagramToPath(currentFilePath);
        } else {
            saveDiagramAs();
        }
    }

    private void saveDiagramAs() {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Save Pipeline As");
        fileChooser.getExtensionFilters().add(
            new FileChooser.ExtensionFilter("Pipeline Files", "*.json"));
        File file = fileChooser.showSaveDialog(primaryStage);
        if (file != null) {
            String path = file.getAbsolutePath();
            if (!path.endsWith(".json")) {
                path += ".json";
            }
            saveDiagramToPath(path);
        }
    }

    private void saveDiagramToPath(String path) {
        // TODO: Implement pipeline saving using PipelineSerializer
        currentFilePath = path;
        isDirty = false;
        addToRecentFiles(path);
        updateTitle();
    }

    private void deleteSelected() {
        // TODO: Delete selected nodes and connections
        markDirty();
        paintCanvas();
    }

    private void selectAll() {
        // TODO: Select all nodes
        paintCanvas();
    }

    private void showAbout() {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle("About");
        alert.setHeaderText("OpenCV Pipeline Editor");
        alert.setContentText("A visual node-based editor for creating OpenCV image processing pipelines.\n\n" +
            "OpenCV Version: " + org.opencv.core.Core.VERSION + "\n" +
            "JavaFX Version: " + System.getProperty("javafx.version"));
        alert.showAndWait();
    }

    private void togglePipeline() {
        if (pipelineRunning) {
            stopPipeline();
        } else {
            startPipeline();
        }
    }

    private void startPipeline() {
        pipelineRunning = true;
        startStopBtn.setText("Stop Pipeline");
        startStopBtn.setStyle("-fx-background-color: rgb(200, 100, 100);");
        statusBar.setText("Pipeline running");
        statusBar.setTextFill(COLOR_STATUS_RUNNING);
        // TODO: Actually start the pipeline threads
    }

    private void stopPipeline() {
        pipelineRunning = false;
        startStopBtn.setText("Start Pipeline");
        startStopBtn.setStyle("-fx-background-color: rgb(100, 180, 100);");
        statusBar.setText("Pipeline stopped");
        statusBar.setTextFill(COLOR_STATUS_STOPPED);
        // TODO: Actually stop the pipeline threads
    }

    private void addNodeAt(String nodeType, int x, int y) {
        // TODO: Create node instance and add to canvas
        markDirty();
        paintCanvas();
    }

    private void filterToolbarButtons() {
        // TODO: Filter toolbar buttons based on search text
    }

    // ========================= Helper Methods =========================

    private void markDirty() {
        isDirty = true;
        updateTitle();
    }

    private void updateTitle() {
        String title = "OpenCV Pipeline Editor - ";
        if (currentFilePath != null) {
            title += new File(currentFilePath).getName();
        } else {
            title += "(untitled)";
        }
        if (isDirty) {
            title += " *";
        }
        primaryStage.setTitle(title);
    }

    private boolean checkUnsavedChanges() {
        if (!isDirty) {
            return true;
        }
        Alert alert = new Alert(Alert.AlertType.CONFIRMATION);
        alert.setTitle("Unsaved Changes");
        alert.setHeaderText("You have unsaved changes");
        alert.setContentText("Do you want to save before continuing?");

        ButtonType saveButton = new ButtonType("Save");
        ButtonType dontSaveButton = new ButtonType("Don't Save");
        ButtonType cancelButton = new ButtonType("Cancel", ButtonBar.ButtonData.CANCEL_CLOSE);

        alert.getButtonTypes().setAll(saveButton, dontSaveButton, cancelButton);

        var result = alert.showAndWait();
        if (result.isPresent()) {
            if (result.get() == saveButton) {
                saveDiagram();
                return true;
            } else if (result.get() == dontSaveButton) {
                return true;
            }
        }
        return false;
    }

    private void loadRecentFiles() {
        String recentFilesStr = prefs.get(RECENT_FILES_KEY, "");
        recentFiles.clear();
        if (!recentFilesStr.isEmpty()) {
            for (String path : recentFilesStr.split("\n")) {
                if (!path.isEmpty() && new File(path).exists()) {
                    recentFiles.add(path);
                }
            }
        }
    }

    private void saveRecentFiles() {
        prefs.put(RECENT_FILES_KEY, String.join("\n", recentFiles));
    }

    private void addToRecentFiles(String path) {
        recentFiles.remove(path);
        recentFiles.add(0, path);
        while (recentFiles.size() > MAX_RECENT_FILES) {
            recentFiles.remove(recentFiles.size() - 1);
        }
        saveRecentFiles();
        prefs.put(LAST_FILE_KEY, path);
        updateOpenRecentMenu();
    }

    private void updateOpenRecentMenu() {
        openRecentMenu.getItems().clear();
        for (String path : recentFiles) {
            MenuItem item = new MenuItem(new File(path).getName());
            item.setOnAction(e -> {
                if (checkUnsavedChanges()) {
                    loadDiagramFromPath(path);
                }
            });
            openRecentMenu.getItems().add(item);
        }
        if (recentFiles.isEmpty()) {
            MenuItem emptyItem = new MenuItem("(No recent files)");
            emptyItem.setDisable(true);
            openRecentMenu.getItems().add(emptyItem);
        } else {
            openRecentMenu.getItems().add(new SeparatorMenuItem());
            MenuItem clearItem = new MenuItem("Clear Recent Files");
            clearItem.setOnAction(e -> {
                recentFiles.clear();
                saveRecentFiles();
                updateOpenRecentMenu();
            });
            openRecentMenu.getItems().add(clearItem);
        }
    }

    public static void main(String[] args) {
        launch(args);
    }
}
