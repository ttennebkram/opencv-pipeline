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

import com.ttennebkram.pipeline.fx.FXConnection;
import com.ttennebkram.pipeline.fx.FXContainerEditorWindow;
import com.ttennebkram.pipeline.fx.FXHelpBrowser;
import com.ttennebkram.pipeline.fx.FXImageUtils;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXNodeFactory;
import com.ttennebkram.pipeline.fx.FXNodeRegistry;
import com.ttennebkram.pipeline.fx.FXPipelineExecutor;
import com.ttennebkram.pipeline.fx.FXPipelineSerializer;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import com.ttennebkram.pipeline.fx.FXWebcamSource;
import com.ttennebkram.pipeline.fx.NodeRenderer;

import org.opencv.core.Mat;

import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
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
    private static final Color COLOR_SELECTION_BOX = Color.rgb(0, 0, 255);  // Bright blue - matches NodeRenderer selection
    private static final Color COLOR_SELECTION_BOX_FILL = Color.rgb(0, 0, 255, 0.1);  // Bright blue with alpha
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
    private Tooltip canvasTooltip;
    private FXNode tooltipNode = null;

    private double zoomLevel = 1.0;
    private static final int[] ZOOM_LEVELS = {25, 50, 75, 100, 125, 150, 200, 300, 400};

    // Pipeline data model
    private List<FXNode> nodes = new ArrayList<>();
    private List<FXConnection> connections = new ArrayList<>();
    private Set<FXNode> selectedNodes = new HashSet<>();
    private Set<FXConnection> selectedConnections = new HashSet<>();

    // Drag state
    private FXNode dragNode = null;
    private double dragOffsetX, dragOffsetY;
    private boolean isDragging = false;

    // Connection drawing state
    private FXNode connectionSource = null;
    private int connectionOutputIndex = 0;
    private double connectionEndX, connectionEndY;
    private boolean isDrawingConnection = false;

    // Selection box state
    private double selectionBoxStartX, selectionBoxStartY;
    private double selectionBoxEndX, selectionBoxEndY;
    private boolean isSelectionBoxDragging = false;

    // Pipeline execution state
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

    // Webcam sources for live preview (ConcurrentHashMap for thread-safe access from executor thread)
    private Map<Integer, FXWebcamSource> webcamSources = new ConcurrentHashMap<>();

    // Pipeline executor
    private FXPipelineExecutor pipelineExecutor;

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

        // Main content - SplitPane with canvas (plus status bar) and preview
        // Put status bar under canvas only, not under preview
        VBox canvasWithStatus = new VBox();
        canvasWithStatus.getChildren().addAll(createCanvasPane(), createStatusBar());
        VBox.setVgrow(canvasWithStatus.getChildren().get(0), Priority.ALWAYS);

        SplitPane splitPane = new SplitPane();
        splitPane.setOrientation(Orientation.HORIZONTAL);
        splitPane.getItems().addAll(canvasWithStatus, createPreviewPane());
        splitPane.setDividerPositions(0.75);
        root.setCenter(splitPane);

        // Create scene
        Scene scene = new Scene(root, 1400, 800);

        // Add keyboard handler for delete key and other shortcuts
        scene.setOnKeyPressed(e -> {
            if (e.getCode() == KeyCode.DELETE || e.getCode() == KeyCode.BACK_SPACE) {
                deleteSelected();
                e.consume();
            }
        });

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
                Platform.exit();
            }
        });

        primaryStage.show();

        // Initial canvas paint
        Platform.runLater(this::paintCanvas);

        // Load last opened file if it exists
        Platform.runLater(this::loadLastFile);

        System.out.println("JavaFX Pipeline Editor started");
        System.out.println("OpenCV version: " + org.opencv.core.Core.VERSION);
    }

    private void loadLastFile() {
        String lastFile = prefs.get(LAST_FILE_KEY, null);
        if (lastFile != null && new File(lastFile).exists()) {
            loadDiagramFromPath(lastFile);
        }
    }

    private MenuBar createMenuBar() {
        MenuBar menuBar = new MenuBar();

        // Note: On macOS with setUseSystemMenuBar(true), the application menu is created
        // automatically by the system using the app name. We don't create a separate "OpenCV" menu
        // as it would appear as a second menu, not replace the auto-generated app menu.
        // The Hide/Show All items are standard macOS menu items that cannot be removed.

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

        MenuItem restartItem = new MenuItem("Restart");
        restartItem.setOnAction(e -> restartApplication());

        MenuItem quitItem = new MenuItem("Quit");
        quitItem.setAccelerator(new KeyCodeCombination(KeyCode.Q, KeyCombination.SHORTCUT_DOWN));
        quitItem.setOnAction(e -> {
            stopPipeline();
            Platform.exit();
        });

        fileMenu.getItems().addAll(newItem, openItem, openRecentMenu, new SeparatorMenuItem(), saveItem, saveAsItem,
                new SeparatorMenuItem(), quitItem, restartItem);

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
        VBox toolbar = new VBox(8);
        toolbar.setPadding(new Insets(8));
        toolbar.setStyle("-fx-background-color: rgb(160, 200, 160);");
        toolbar.setPrefWidth(200);

        // Search box with clear button inside using StackPane
        searchBox = new TextField();
        searchBox.setPromptText("Search nodes...");
        searchBox.textProperty().addListener((obs, oldVal, newVal) -> filterToolbarButtons());
        // Add padding on right for clear button
        searchBox.setStyle("-fx-padding: 2 20 2 5;");

        Button clearSearchBtn = new Button("\u00D7"); // Unicode multiplication sign
        clearSearchBtn.setStyle("-fx-font-size: 12px; -fx-padding: 0 5 0 5; -fx-background-color: transparent; -fx-cursor: hand;");
        clearSearchBtn.setOnAction(e -> searchBox.clear());
        // Only show when search box has text
        clearSearchBtn.visibleProperty().bind(searchBox.textProperty().isNotEmpty());

        StackPane searchStack = new StackPane();
        searchStack.getChildren().addAll(searchBox, clearSearchBtn);
        StackPane.setAlignment(clearSearchBtn, javafx.geometry.Pos.CENTER_RIGHT);
        StackPane.setMargin(clearSearchBtn, new Insets(0, 2, 0, 0));
        toolbar.getChildren().add(searchStack);

        // Scrollable content for node buttons
        ScrollPane scrollPane = new ScrollPane();
        scrollPane.setFitToWidth(true);
        scrollPane.setVbarPolicy(ScrollPane.ScrollBarPolicy.AS_NEEDED);
        scrollPane.setHbarPolicy(ScrollPane.ScrollBarPolicy.NEVER);
        // Set background color on viewport and scrollpane itself
        scrollPane.setStyle("-fx-background: rgb(160, 200, 160); -fx-background-color: rgb(160, 200, 160); -fx-control-inner-background: rgb(160, 200, 160);");
        VBox.setVgrow(scrollPane, Priority.ALWAYS);

        toolbarContent = new VBox(0);  // No spacing between buttons
        toolbarContent.setPadding(new Insets(4, 8, 4, 8));  // Padding around all buttons
        toolbarContent.setStyle("-fx-background-color: rgb(160, 200, 160);");
        scrollPane.setContent(toolbarContent);

        // Populate node buttons from registry (exclude Container I/O which is only for container editor)
        for (String category : FXNodeRegistry.getCategoriesExcluding("Container I/O")) {
            addToolbarCategory(category);
            for (FXNodeRegistry.NodeType nodeType : FXNodeRegistry.getNodesInCategory(category)) {
                final String typeName = nodeType.name;
                addToolbarButton(nodeType.getButtonName(), () -> addNodeAt(typeName, getNextNodeX(), getNextNodeY()));
            }
        }

        toolbar.getChildren().add(scrollPane);

        // Start/Stop button moved to preview pane

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
        canvasScrollPane.setPannable(false);  // Disable panning - we handle canvas interactions ourselves

        // Mouse event handlers
        pipelineCanvas.setOnMousePressed(this::handleMousePressed);
        pipelineCanvas.setOnMouseDragged(this::handleMouseDragged);
        pipelineCanvas.setOnMouseReleased(this::handleMouseReleased);
        pipelineCanvas.setOnMouseMoved(this::handleMouseMoved);
        pipelineCanvas.setOnMouseClicked(e -> {
            if (e.getClickCount() == 2) {
                handleDoubleClick(e);
            }
        });

        // Initialize tooltip for nodes (installed/uninstalled dynamically based on hover)
        canvasTooltip = new Tooltip();
        canvasTooltip.setShowDelay(javafx.util.Duration.millis(500));

        // Context menu - built dynamically based on clicked node
        pipelineCanvas.setOnContextMenuRequested(e -> {
            double canvasX = e.getX() / zoomLevel;
            double canvasY = e.getY() / zoomLevel;
            FXNode clickedNode = getNodeAt(canvasX, canvasY);

            ContextMenu contextMenu = new ContextMenu();

            if (clickedNode != null) {
                // Select the node if not already selected
                if (!selectedNodes.contains(clickedNode)) {
                    selectedNodes.clear();
                    selectedConnections.clear();
                    selectedNodes.add(clickedNode);
                    paintCanvas();
                }

                // Properties option
                MenuItem propertiesItem = new MenuItem("Properties...");
                propertiesItem.setOnAction(ev -> showNodeProperties(clickedNode));
                contextMenu.getItems().add(propertiesItem);

                // Edit Container option for container nodes
                if (clickedNode.isContainer) {
                    MenuItem editContainerItem = new MenuItem("Edit Container...");
                    editContainerItem.setOnAction(ev -> openContainerEditor(clickedNode));
                    contextMenu.getItems().add(editContainerItem);
                }

                // Enable/Disable toggle
                MenuItem enableItem = new MenuItem(clickedNode.enabled ? "Disable" : "Enable");
                enableItem.setOnAction(ev -> {
                    clickedNode.enabled = !clickedNode.enabled;
                    paintCanvas();
                });
                contextMenu.getItems().add(enableItem);

                contextMenu.getItems().add(new SeparatorMenuItem());

                // Delete option
                MenuItem deleteItem = new MenuItem("Delete");
                deleteItem.setOnAction(ev -> deleteSelected());
                contextMenu.getItems().add(deleteItem);
            } else {
                // Canvas context menu (no node clicked)
                MenuItem pasteItem = new MenuItem("Paste");
                pasteItem.setDisable(true); // Not implemented yet
                contextMenu.getItems().add(pasteItem);
            }

            contextMenu.show(pipelineCanvas, e.getScreenX(), e.getScreenY());
        });

        return canvasScrollPane;
    }

    private VBox createPreviewPane() {
        VBox previewPane = new VBox(5);
        previewPane.setPadding(new Insets(10));
        previewPane.setStyle("-fx-background-color: #f0f0f0;");

        // Save and Cancel buttons (matching ContainerEditorWindow layout)
        HBox buttonPanel = new HBox(5);
        buttonPanel.setMaxWidth(Double.MAX_VALUE);

        Button saveButton = new Button("Save");
        saveButton.setMaxWidth(Double.MAX_VALUE);
        HBox.setHgrow(saveButton, Priority.ALWAYS);
        saveButton.setOnAction(e -> saveDiagram());

        Button cancelButton = new Button("Cancel");
        cancelButton.setMaxWidth(Double.MAX_VALUE);
        HBox.setHgrow(cancelButton, Priority.ALWAYS);
        cancelButton.setOnAction(e -> handleCancel());

        buttonPanel.getChildren().addAll(saveButton, cancelButton);

        // Start/Stop pipeline button at top
        startStopBtn = new Button("Start Pipeline");
        startStopBtn.setStyle("-fx-background-color: rgb(100, 180, 100); -fx-font-weight: bold;");
        startStopBtn.setOnAction(e -> togglePipeline());
        startStopBtn.setMaxWidth(Double.MAX_VALUE);

        // Instructions label
        Label instructionsLabel = new Label("Instructions:");
        instructionsLabel.setStyle("-fx-font-weight: bold; -fx-padding: 5 0 0 0;");

        Label instructions = new Label(NodeRenderer.INSTRUCTIONS_TEXT);
        instructions.setStyle("-fx-font-size: 11px;");
        instructions.setWrapText(true);

        Label previewLabel = new Label("Preview");
        previewLabel.setStyle("-fx-font-weight: bold; -fx-padding: 10 0 0 0;");

        previewImageView = new ImageView();
        previewImageView.setPreserveRatio(true);
        previewImageView.setFitWidth(300);
        previewImageView.setFitHeight(300);

        // Placeholder for when no image
        StackPane imageContainer = new StackPane(previewImageView);
        imageContainer.setStyle("-fx-background-color: #cccccc; -fx-min-height: 200;");
        VBox.setVgrow(imageContainer, Priority.ALWAYS);

        // Zoom controls moved to status bar

        previewPane.getChildren().addAll(buttonPanel, startStopBtn, instructionsLabel, instructions, previewLabel, imageContainer);

        return previewPane;
    }

    private HBox createStatusBar() {
        HBox statusBarBox = new HBox(10);
        statusBarBox.setPadding(new Insets(5, 10, 5, 10));
        statusBarBox.setStyle("-fx-background-color: rgb(160, 160, 160);");
        statusBarBox.setAlignment(javafx.geometry.Pos.CENTER_LEFT);

        // Zoom controls on the left
        Label zoomLabel = new Label("Zoom:");
        zoomCombo = new ComboBox<>();
        for (int level : ZOOM_LEVELS) {
            zoomCombo.getItems().add(level + "%");
        }
        zoomCombo.setValue("100%");
        zoomCombo.setPrefWidth(80);
        zoomCombo.setOnAction(e -> {
            String selected = zoomCombo.getValue();
            if (selected != null) {
                zoomLevel = Integer.parseInt(selected.replace("%", "")) / 100.0;
                paintCanvas();
            }
        });

        // Node count on the left
        nodeCountLabel = new Label("Nodes: 0");

        Region spacer1 = new Region();
        HBox.setHgrow(spacer1, Priority.ALWAYS);

        // Pipeline status in the center - more prominent
        statusBar = new Label("Pipeline stopped");
        statusBar.setTextFill(COLOR_STATUS_STOPPED);
        statusBar.setStyle("-fx-font-weight: bold; -fx-font-size: 12px;");

        Region spacer2 = new Region();
        HBox.setHgrow(spacer2, Priority.ALWAYS);

        // Zoom controls on the right
        statusBarBox.getChildren().addAll(nodeCountLabel, spacer1, statusBar, spacer2, zoomLabel, zoomCombo);

        return statusBarBox;
    }

    // ========================= Canvas Painting =========================

    private void paintCanvas() {
        // Sync stats from processors to FXNodes before rendering
        if (pipelineExecutor != null && pipelineExecutor.isRunning()) {
            pipelineExecutor.syncAllStats();
        }

        GraphicsContext gc = pipelineCanvas.getGraphicsContext2D();

        // Apply zoom transform
        gc.save();
        gc.scale(zoomLevel, zoomLevel);

        // Clear background
        gc.setFill(Color.WHITE);
        gc.fillRect(0, 0, pipelineCanvas.getWidth() / zoomLevel, pipelineCanvas.getHeight() / zoomLevel);

        // Draw grid
        gc.setStroke(COLOR_GRID_LINES);
        gc.setLineWidth(1);
        double gridSize = 20;
        for (double x = 0; x < pipelineCanvas.getWidth() / zoomLevel; x += gridSize) {
            gc.strokeLine(x, 0, x, pipelineCanvas.getHeight() / zoomLevel);
        }
        for (double y = 0; y < pipelineCanvas.getHeight() / zoomLevel; y += gridSize) {
            gc.strokeLine(0, y, pipelineCanvas.getWidth() / zoomLevel, y);
        }

        // Draw connections first (behind nodes)
        for (FXConnection conn : connections) {
            double[] start = conn.getStartPoint();
            double[] end = conn.getEndPoint();
            if (start != null && end != null) {
                NodeRenderer.renderConnection(gc, start[0], start[1], end[0], end[1],
                    conn.selected || selectedConnections.contains(conn),
                    conn.queueSize, conn.totalFrames);
            }
        }

        // Draw connection being drawn
        if (isDrawingConnection && connectionSource != null) {
            double[] start = connectionSource.getOutputPoint(connectionOutputIndex);
            if (start != null) {
                NodeRenderer.renderConnection(gc, start[0], start[1], connectionEndX, connectionEndY, true);
            }
        }

        // Draw nodes
        for (FXNode node : nodes) {
            boolean isSelected = selectedNodes.contains(node);
            int[] outputCounters = new int[] { node.outputCount1, node.outputCount2, node.outputCount3, node.outputCount4 };
            NodeRenderer.renderNode(gc, node.x, node.y, node.width, node.height,
                node.label, isSelected, node.enabled, node.backgroundColor,
                node.hasInput, node.hasDualInput, node.outputCount, node.thumbnail,
                node.isContainer, node.inputCount, node.inputCount2, outputCounters, node.nodeType);

            // Draw stats line (Pri/Work/FPS) below title - always show (including after load)
            // Source nodes (no input) show FPS; processing nodes don't
            if (!node.hasInput) {
                NodeRenderer.drawSourceStatsLine(gc, node.x + 22, node.y + node.height - 8,
                    node.threadPriority, node.workUnitsCompleted, node.effectiveFps);
            } else {
                NodeRenderer.drawStatsLine(gc, node.x + 22, node.y + node.height - 8,
                    node.threadPriority, node.workUnitsCompleted);
            }
        }

        // Draw selection box
        if (isSelectionBoxDragging) {
            gc.setStroke(COLOR_SELECTION_BOX);
            gc.setLineWidth(1);
            gc.setLineDashes(5);
            double x = Math.min(selectionBoxStartX, selectionBoxEndX);
            double y = Math.min(selectionBoxStartY, selectionBoxEndY);
            double w = Math.abs(selectionBoxEndX - selectionBoxStartX);
            double h = Math.abs(selectionBoxEndY - selectionBoxStartY);
            gc.strokeRect(x, y, w, h);
            gc.setFill(COLOR_SELECTION_BOX_FILL);
            gc.fillRect(x, y, w, h);
            gc.setLineDashes(null);
        }

        gc.restore();

        // Update node count
        updateNodeCount();
    }

    // ========================= Mouse Handlers =========================

    private void handleMousePressed(javafx.scene.input.MouseEvent e) {
        double canvasX = e.getX() / zoomLevel;
        double canvasY = e.getY() / zoomLevel;

        // First check if clicking on a node's checkbox (before other node interactions)
        for (FXNode node : nodes) {
            if (node.isOnCheckbox(canvasX, canvasY)) {
                // Toggle enabled state
                node.enabled = !node.enabled;
                markDirty();
                paintCanvas();
                return;
            }
            // Check if clicking on help icon
            if (node.isOnHelpIcon(canvasX, canvasY)) {
                System.out.println("DEBUG: Help icon clicked for node: " + node.nodeType);
                FXHelpBrowser.openForNodeType(primaryStage, node.nodeType);
                return;
            }
        }

        // Check if clicking on a node's output point (to start connection)
        for (FXNode node : nodes) {
            int outputIdx = node.getOutputPointAt(canvasX, canvasY, NodeRenderer.CONNECTION_RADIUS + 3);
            if (outputIdx >= 0) {
                // Start drawing a connection
                connectionSource = node;
                connectionOutputIndex = outputIdx;
                connectionEndX = canvasX;
                connectionEndY = canvasY;
                isDrawingConnection = true;
                paintCanvas();
                return;
            }
        }

        // Check if clicking on a node's input point (to start reverse connection)
        for (FXNode node : nodes) {
            int inputIdx = node.getInputPointAt(canvasX, canvasY, NodeRenderer.CONNECTION_RADIUS + 3);
            if (inputIdx >= 0) {
                // Don't start selection box - just consume the click on the input point
                // (Could implement reverse connection drawing in the future)
                return;
            }
        }

        // Check if clicking on a node
        FXNode clickedNode = getNodeAt(canvasX, canvasY);
        if (clickedNode != null) {
            // Check if shift is held for multi-select
            if (!e.isShiftDown() && !selectedNodes.contains(clickedNode)) {
                // Clear selection and select this node
                selectedNodes.clear();
                selectedConnections.clear();
            }
            selectedNodes.add(clickedNode);

            // Start dragging
            dragNode = clickedNode;
            dragOffsetX = canvasX - clickedNode.x;
            dragOffsetY = canvasY - clickedNode.y;
            isDragging = true;
            paintCanvas();
            return;
        }

        // Check if clicking on a connection
        FXConnection clickedConn = getConnectionAt(canvasX, canvasY);
        if (clickedConn != null) {
            if (!e.isShiftDown()) {
                selectedNodes.clear();
                selectedConnections.clear();
            }
            selectedConnections.add(clickedConn);
            paintCanvas();
            return;
        }

        // Clicked on empty space - start selection box or clear selection
        if (!e.isShiftDown()) {
            selectedNodes.clear();
            selectedConnections.clear();
        }
        selectionBoxStartX = canvasX;
        selectionBoxStartY = canvasY;
        selectionBoxEndX = canvasX;
        selectionBoxEndY = canvasY;
        isSelectionBoxDragging = true;
        paintCanvas();
    }

    private void handleMouseDragged(javafx.scene.input.MouseEvent e) {
        double canvasX = e.getX() / zoomLevel;
        double canvasY = e.getY() / zoomLevel;

        if (isDrawingConnection) {
            connectionEndX = canvasX;
            connectionEndY = canvasY;
            paintCanvas();
        } else if (isDragging && dragNode != null) {
            // Move the dragged node
            double dx = canvasX - dragOffsetX - dragNode.x;
            double dy = canvasY - dragOffsetY - dragNode.y;

            // Move all selected nodes
            for (FXNode node : selectedNodes) {
                node.x += dx;
                node.y += dy;
            }

            dragNode.x = canvasX - dragOffsetX;
            dragNode.y = canvasY - dragOffsetY;
            markDirty();
            paintCanvas();
        } else if (isSelectionBoxDragging) {
            selectionBoxEndX = canvasX;
            selectionBoxEndY = canvasY;
            paintCanvas();
        }
    }

    private void handleMouseReleased(javafx.scene.input.MouseEvent e) {
        double canvasX = e.getX() / zoomLevel;
        double canvasY = e.getY() / zoomLevel;

        if (isDrawingConnection) {
            // Check if we're over a node's input point
            for (FXNode node : nodes) {
                if (node != connectionSource) {
                    int inputIdx = node.getInputPointAt(canvasX, canvasY, NodeRenderer.CONNECTION_RADIUS + 5);
                    if (inputIdx >= 0) {
                        // Create connection
                        FXConnection conn = new FXConnection(connectionSource, connectionOutputIndex, node, inputIdx);
                        connections.add(conn);

                        // Propagate initial frame to the newly connected node
                        propagateInitialFrame(conn);

                        markDirty();
                        break;
                    }
                }
            }
            isDrawingConnection = false;
            connectionSource = null;
        }

        if (isSelectionBoxDragging) {
            // Select all nodes within the selection box
            double x1 = Math.min(selectionBoxStartX, selectionBoxEndX);
            double y1 = Math.min(selectionBoxStartY, selectionBoxEndY);
            double x2 = Math.max(selectionBoxStartX, selectionBoxEndX);
            double y2 = Math.max(selectionBoxStartY, selectionBoxEndY);

            for (FXNode node : nodes) {
                if (node.x >= x1 && node.x + node.width <= x2 &&
                    node.y >= y1 && node.y + node.height <= y2) {
                    selectedNodes.add(node);
                }
            }
            isSelectionBoxDragging = false;
        }

        isDragging = false;
        dragNode = null;
        paintCanvas();
    }

    private void handleDoubleClick(javafx.scene.input.MouseEvent e) {
        double canvasX = e.getX() / zoomLevel;
        double canvasY = e.getY() / zoomLevel;

        FXNode node = getNodeAt(canvasX, canvasY);
        System.out.println("[DEBUG] handleDoubleClick: node = " + (node != null ? node.nodeType : "null"));
        if (node != null) {
            if (node.isContainer) {
                openContainerEditor(node);
            } else {
                showNodeProperties(node);
            }
        }
    }

    private void openContainerEditor(FXNode containerNode) {
        FXContainerEditorWindow editorWindow = new FXContainerEditorWindow(primaryStage, containerNode, this::markDirty);

        // Wire up pipeline control callbacks
        editorWindow.setOnStartPipeline(this::startPipeline);
        editorWindow.setOnStopPipeline(this::stopPipeline);
        editorWindow.setIsPipelineRunning(() -> pipelineRunning);
        editorWindow.setOnRequestGlobalSave(this::saveDiagram);

        // Set base path for resolving relative pipeline file paths
        editorWindow.setBasePath(currentFilePath);

        editorWindow.show();
        editorWindow.updatePipelineButtonState();
    }

    private void handleMouseMoved(javafx.scene.input.MouseEvent e) {
        double canvasX = e.getX() / zoomLevel;
        double canvasY = e.getY() / zoomLevel;

        FXNode hoveredNode = getNodeAt(canvasX, canvasY);

        // Determine what element we're hovering over
        boolean onCheckbox = hoveredNode != null && hoveredNode.isOnCheckbox(canvasX, canvasY);
        boolean onHelpIcon = hoveredNode != null && hoveredNode.isOnHelpIcon(canvasX, canvasY);

        // Update cursor - show hand for checkbox and help icon (help icon always clickable)
        if (onCheckbox || onHelpIcon) {
            pipelineCanvas.setCursor(javafx.scene.Cursor.HAND);
        } else {
            pipelineCanvas.setCursor(javafx.scene.Cursor.DEFAULT);
        }

        // Update tooltip based on what we're hovering over
        String tooltipText = null;
        if (onCheckbox) {
            tooltipText = "Enable / Disable this Node";
        } else if (onHelpIcon) {
            if (FXHelpBrowser.hasHelp(hoveredNode.nodeType)) {
                tooltipText = "Help";
            } else {
                tooltipText = "Help (not available for this node type)";
            }
        }

        // Install or uninstall tooltip based on hover state
        if (tooltipText != null) {
            canvasTooltip.setText(tooltipText);
            Tooltip.install(pipelineCanvas, canvasTooltip);
        } else {
            Tooltip.uninstall(pipelineCanvas, canvasTooltip);
        }
        tooltipNode = hoveredNode;
    }

    private FXNode getNodeAt(double x, double y) {
        // Search in reverse order (top-most first)
        for (int i = nodes.size() - 1; i >= 0; i--) {
            FXNode node = nodes.get(i);
            if (node.contains(x, y)) {
                return node;
            }
        }
        return null;
    }

    private FXConnection getConnectionAt(double x, double y) {
        for (FXConnection conn : connections) {
            if (conn.isNear(x, y, 8)) {
                return conn;
            }
        }
        return null;
    }

    private void showNodeProperties(FXNode node) {
        System.out.println("[DEBUG] showNodeProperties called with nodeType: '" + node.nodeType + "'");
        FXPropertiesDialog dialog = new FXPropertiesDialog(
            primaryStage,
            node.label + " Properties",
            node.nodeType,
            node.label
        );

        // Add method signature/description from registry
        FXNodeRegistry.NodeType typeInfo = FXNodeRegistry.getNodeType(node.nodeType);
        if (typeInfo != null && typeInfo.description != null) {
            dialog.addDescription(typeInfo.description);
        }

        // Use FXNodePropertiesHelper to add comprehensive properties for supported node types
        // This helper covers all nodes with properties that differ from the main branch
        com.ttennebkram.pipeline.fx.FXNodePropertiesHelper.addPropertiesForNode(dialog, node);

        // Add node-type-specific properties
        Spinner<Integer> cameraSpinner = null;
        TextField filePathField = null;
        TextField pipelineFileField = null;
        ComboBox<String> fpsCombo = null;

        if ("WebcamSource".equals(node.nodeType)) {
            cameraSpinner = dialog.addSpinner("Camera Index:", 0, 5, node.cameraIndex);
            dialog.addDescription("Camera 0 is often a virtual camera (e.g., iPhone).\nTry camera 1 for your built-in webcam.");
        } else if ("FileSource".equals(node.nodeType)) {
            // File path with browse button
            filePathField = new TextField(node.filePath);
            filePathField.setPrefWidth(250);

            Button browseBtn = new Button("Browse...");
            final TextField finalFilePathField = filePathField;
            browseBtn.setOnAction(e -> {
                e.consume();  // Prevent event from propagating
                FileChooser fileChooser = new FileChooser();
                fileChooser.setTitle("Select Image or Video File");
                fileChooser.getExtensionFilters().addAll(
                    new FileChooser.ExtensionFilter("Image Files", "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif"),
                    new FileChooser.ExtensionFilter("Video Files", "*.mp4", "*.avi", "*.mov", "*.mkv"),
                    new FileChooser.ExtensionFilter("All Files", "*.*")
                );
                // Start in file's directory if path exists
                if (!node.filePath.isEmpty()) {
                    java.io.File currentFile = new java.io.File(node.filePath);
                    if (currentFile.getParentFile() != null && currentFile.getParentFile().exists()) {
                        fileChooser.setInitialDirectory(currentFile.getParentFile());
                    }
                }
                // Use null owner to avoid modal dialog beep issue on macOS
                java.io.File file = fileChooser.showOpenDialog(null);
                if (file != null) {
                    finalFilePathField.setText(file.getAbsolutePath());
                }
            });

            HBox fileRow = new HBox(5);
            Label fileLabel = new Label("File:");
            fileLabel.setMinWidth(40);
            fileRow.getChildren().addAll(fileLabel, filePathField, browseBtn);
            dialog.addCustomContent(fileRow);

            // FPS combo box
            fpsCombo = new ComboBox<>();
            fpsCombo.getItems().addAll("Automatic", "1", "5", "10", "15", "30");
            // Select current value
            if (node.fps < 0) {
                fpsCombo.setValue("Automatic");
            } else {
                fpsCombo.setValue(String.valueOf((int) node.fps));
            }
            fpsCombo.setEditable(true);  // Allow custom fps values

            HBox fpsRow = new HBox(5);
            Label fpsLabel = new Label("FPS:");
            fpsLabel.setMinWidth(40);
            fpsRow.getChildren().addAll(fpsLabel, fpsCombo);
            dialog.addCustomContent(fpsRow);
            dialog.addDescription("Automatic: 1 FPS for images, video native rate for videos.");
        } else if ("BlankSource".equals(node.nodeType)) {
            // BlankSource properties: width, height, color, fps
            int currentWidth = node.properties.containsKey("imageWidth") ?
                ((Number) node.properties.get("imageWidth")).intValue() : 640;
            int currentHeight = node.properties.containsKey("imageHeight") ?
                ((Number) node.properties.get("imageHeight")).intValue() : 480;
            int currentColorIndex = node.properties.containsKey("colorIndex") ?
                ((Number) node.properties.get("colorIndex")).intValue() : 0;
            int currentFpsIndex = node.properties.containsKey("fpsIndex") ?
                ((Number) node.properties.get("fpsIndex")).intValue() : 2;

            // Width spinner
            Spinner<Integer> widthSpinner = new Spinner<>(1, 4096, currentWidth);
            widthSpinner.setEditable(true);
            widthSpinner.setPrefWidth(100);
            HBox widthRow = new HBox(10);
            Label widthLabel = new Label("Width:");
            widthLabel.setMinWidth(50);
            widthRow.getChildren().addAll(widthLabel, widthSpinner);
            dialog.addCustomContent(widthRow);

            // Height spinner
            Spinner<Integer> heightSpinner = new Spinner<>(1, 4096, currentHeight);
            heightSpinner.setEditable(true);
            heightSpinner.setPrefWidth(100);
            HBox heightRow = new HBox(10);
            Label heightLabel = new Label("Height:");
            heightLabel.setMinWidth(50);
            heightRow.getChildren().addAll(heightLabel, heightSpinner);
            dialog.addCustomContent(heightRow);

            // Color combo box
            String[] colorOptions = {"Black", "White", "Red", "Green", "Blue", "Yellow"};
            ComboBox<String> colorCombo = new ComboBox<>();
            colorCombo.getItems().addAll(colorOptions);
            colorCombo.setValue(colorOptions[Math.min(currentColorIndex, colorOptions.length - 1)]);
            HBox colorRow = new HBox(10);
            Label colorLabel = new Label("Color:");
            colorLabel.setMinWidth(50);
            colorRow.getChildren().addAll(colorLabel, colorCombo);
            dialog.addCustomContent(colorRow);

            // FPS combo box
            String[] fpsOptions = {"1", "15", "30", "60"};
            ComboBox<String> blankFpsCombo = new ComboBox<>();
            blankFpsCombo.getItems().addAll(fpsOptions);
            blankFpsCombo.setValue(fpsOptions[Math.min(currentFpsIndex, fpsOptions.length - 1)]);
            HBox blankFpsRow = new HBox(10);
            Label blankFpsLabel = new Label("FPS:");
            blankFpsLabel.setMinWidth(50);
            blankFpsRow.getChildren().addAll(blankFpsLabel, blankFpsCombo);
            dialog.addCustomContent(blankFpsRow);

            // Store control references for OK handler
            node.properties.put("_widthSpinner", widthSpinner);
            node.properties.put("_heightSpinner", heightSpinner);
            node.properties.put("_colorCombo", colorCombo);
            node.properties.put("_blankFpsCombo", blankFpsCombo);
        } else if ("Container".equals(node.nodeType)) {
            // Pipeline file path with browse button
            pipelineFileField = new TextField(node.pipelineFilePath != null ? node.pipelineFilePath : "");
            pipelineFileField.setPrefWidth(250);

            Button browseBtn = new Button("Browse...");
            final TextField finalPipelineField = pipelineFileField;
            browseBtn.setOnAction(e -> {
                e.consume();  // Prevent event from propagating
                FileChooser fileChooser = new FileChooser();
                fileChooser.setTitle("Select Pipeline File");
                fileChooser.getExtensionFilters().add(
                    new FileChooser.ExtensionFilter("Pipeline Files", "*.json")
                );
                // Start in file's directory if path exists
                if (node.pipelineFilePath != null && !node.pipelineFilePath.isEmpty()) {
                    java.io.File currentFile = new java.io.File(node.pipelineFilePath);
                    if (currentFile.getParentFile() != null && currentFile.getParentFile().exists()) {
                        fileChooser.setInitialDirectory(currentFile.getParentFile());
                    }
                } else if (currentFilePath != null) {
                    // Start in current document's directory
                    java.io.File currentDoc = new java.io.File(currentFilePath);
                    if (currentDoc.getParentFile() != null && currentDoc.getParentFile().exists()) {
                        fileChooser.setInitialDirectory(currentDoc.getParentFile());
                    }
                }
                // Use null owner to avoid modal dialog beep issue on macOS
                java.io.File file = fileChooser.showOpenDialog(null);
                if (file != null) {
                    finalPipelineField.setText(file.getAbsolutePath());
                }
            });

            HBox pipelineRow = new HBox(5);
            Label pipelineLabel = new Label("Pipeline File:");
            pipelineLabel.setMinWidth(80);
            pipelineRow.getChildren().addAll(pipelineLabel, pipelineFileField, browseBtn);
            dialog.addCustomContent(pipelineRow);

            dialog.addDescription("Select an external pipeline file for this sub-diagram.\nLeave empty to edit the sub-diagram inline.");
        } else if ("Gain".equals(node.nodeType)) {
            // Get current gain value from properties (default 1.0)
            double currentGain = 1.0;
            if (node.properties.containsKey("gain")) {
                currentGain = ((Number) node.properties.get("gain")).doubleValue();
            }

            // Create logarithmic gain slider (0.05x to 20x)
            // Slider range: 0-100, mapped logarithmically
            // log10(0.05) = -1.301, log10(20) = 1.301, range = 2.602
            // Formula: gain = 10^((slider - 50) / 50 * 1.301)
            // Or equivalently: slider = (log10(gain) / 1.301) * 50 + 50
            final double LOG_RANGE = Math.log10(20.0);  // ~1.301
            double sliderVal = (Math.log10(currentGain) / LOG_RANGE) * 50 + 50;
            sliderVal = Math.max(0, Math.min(100, sliderVal));  // Clamp to valid range

            Slider gainSlider = new Slider(0, 100, sliderVal);
            gainSlider.setPrefWidth(200);
            gainSlider.setShowTickMarks(true);

            Label gainValueLabel = new Label(formatGainValue(currentGain));
            gainValueLabel.setMinWidth(60);

            gainSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
                double logVal = (newVal.doubleValue() - 50) / 50.0 * LOG_RANGE;
                double g = Math.pow(10, logVal);
                gainValueLabel.setText(formatGainValue(g));
            });

            HBox gainRow = new HBox(10);
            gainRow.getChildren().addAll(new Label("Gain (5% - 20x):"), gainSlider, gainValueLabel);
            dialog.addCustomContent(gainRow);
            dialog.addDescription("Brightness/Gain Adjustment\ncv2.multiply(src, gain)");

            // Store slider reference for OK handler
            node.properties.put("_gainSlider", gainSlider);
            node.properties.put("_gainLogRange", LOG_RANGE);
        } else if ("MedianBlur".equals(node.nodeType)) {
            int currentKsize = node.properties.containsKey("ksize") ?
                ((Number) node.properties.get("ksize")).intValue() : 5;

            Slider ksizeSlider = new Slider(1, 31, currentKsize);
            ksizeSlider.setPrefWidth(200);
            ksizeSlider.setMajorTickUnit(10);
            ksizeSlider.setShowTickMarks(true);

            Label ksizeValueLabel = new Label(String.valueOf(currentKsize));
            ksizeValueLabel.setMinWidth(30);

            ksizeSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
                int v = newVal.intValue();
                if (v % 2 == 0) v++;  // Must be odd
                ksizeValueLabel.setText(String.valueOf(v));
            });

            HBox ksizeRow = new HBox(10);
            ksizeRow.getChildren().addAll(new Label("Kernel Size:"), ksizeSlider, ksizeValueLabel);
            dialog.addCustomContent(ksizeRow);

            node.properties.put("_ksizeSlider", ksizeSlider);
        } else if ("Erode".equals(node.nodeType) || "Dilate".equals(node.nodeType)) {
            int currentKsize = node.properties.containsKey("kernelSize") ?
                ((Number) node.properties.get("kernelSize")).intValue() : 5;
            int currentIter = node.properties.containsKey("iterations") ?
                ((Number) node.properties.get("iterations")).intValue() : 1;

            Slider ksizeSlider = new Slider(1, 21, currentKsize);
            ksizeSlider.setPrefWidth(200);
            ksizeSlider.setMajorTickUnit(5);
            ksizeSlider.setShowTickMarks(true);

            Label ksizeValueLabel = new Label(String.valueOf(currentKsize));
            ksizeValueLabel.setMinWidth(30);

            ksizeSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
                ksizeValueLabel.setText(String.valueOf(newVal.intValue()));
            });

            HBox ksizeRow = new HBox(10);
            ksizeRow.getChildren().addAll(new Label("Kernel Size:"), ksizeSlider, ksizeValueLabel);
            dialog.addCustomContent(ksizeRow);

            Slider iterSlider = new Slider(1, 10, currentIter);
            iterSlider.setPrefWidth(200);
            iterSlider.setMajorTickUnit(3);
            iterSlider.setShowTickMarks(true);

            Label iterValueLabel = new Label(String.valueOf(currentIter));
            iterValueLabel.setMinWidth(30);

            iterSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
                iterValueLabel.setText(String.valueOf(newVal.intValue()));
            });

            HBox iterRow = new HBox(10);
            iterRow.getChildren().addAll(new Label("Iterations:"), iterSlider, iterValueLabel);
            dialog.addCustomContent(iterRow);

            node.properties.put("_kernelSizeSlider", ksizeSlider);
            node.properties.put("_iterationsSlider", iterSlider);
        } else if ("MorphOpen".equals(node.nodeType) || "MorphClose".equals(node.nodeType)) {
            int currentKsize = node.properties.containsKey("kernelSize") ?
                ((Number) node.properties.get("kernelSize")).intValue() : 5;

            Slider ksizeSlider = new Slider(1, 21, currentKsize);
            ksizeSlider.setPrefWidth(200);
            ksizeSlider.setMajorTickUnit(5);
            ksizeSlider.setShowTickMarks(true);

            Label ksizeValueLabel = new Label(String.valueOf(currentKsize));
            ksizeValueLabel.setMinWidth(30);

            ksizeSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
                ksizeValueLabel.setText(String.valueOf(newVal.intValue()));
            });

            HBox ksizeRow = new HBox(10);
            ksizeRow.getChildren().addAll(new Label("Kernel Size:"), ksizeSlider, ksizeValueLabel);
            dialog.addCustomContent(ksizeRow);

            node.properties.put("_kernelSizeSlider", ksizeSlider);
        } else if ("BitPlanesGrayscale".equals(node.nodeType)) {
            // Bit Planes Grayscale - 8 bit planes with enable checkbox and gain slider each
            dialog.addDescription("Bit Planes Grayscale: Select and adjust bit planes\nBit plane decomposition with gain");

            // Load current values from properties
            boolean[] bitEnabled = new boolean[8];
            double[] bitGain = new double[8];
            for (int i = 0; i < 8; i++) {
                bitEnabled[i] = true;
                bitGain[i] = 1.0;
            }
            if (node.properties.containsKey("bitEnabled")) {
                boolean[] arr = (boolean[]) node.properties.get("bitEnabled");
                for (int i = 0; i < Math.min(arr.length, 8); i++) bitEnabled[i] = arr[i];
            }
            if (node.properties.containsKey("bitGain")) {
                double[] arr = (double[]) node.properties.get("bitGain");
                for (int i = 0; i < Math.min(arr.length, 8); i++) bitGain[i] = arr[i];
            }

            // Create arrays to hold controls
            CheckBox[] checkBoxes = new CheckBox[8];
            Slider[] gainSliders = new Slider[8];
            Label[] gainLabels = new Label[8];

            // Header
            HBox headerRow = new HBox(10);
            Label bitHeader = new Label("Bit");
            bitHeader.setMinWidth(30);
            Label onHeader = new Label("On");
            onHeader.setMinWidth(30);
            Label gainHeader = new Label("Gain (0.1x - 10x)");
            gainHeader.setMinWidth(200);
            headerRow.getChildren().addAll(bitHeader, onHeader, gainHeader);
            dialog.addCustomContent(headerRow);

            // 8 rows for bit planes
            for (int i = 0; i < 8; i++) {
                int bitNum = 7 - i;
                final int idx = i;

                HBox row = new HBox(10);
                row.setAlignment(javafx.geometry.Pos.CENTER_LEFT);

                Label bitLabel = new Label(String.valueOf(bitNum));
                bitLabel.setMinWidth(30);

                checkBoxes[i] = new CheckBox();
                checkBoxes[i].setSelected(bitEnabled[i]);

                // Logarithmic gain slider: 0-200 maps to 0.1x-10x (center=100 = 1.0x)
                gainSliders[i] = new Slider(0, 200, Math.log10(bitGain[i]) * 100 + 100);
                gainSliders[i].setPrefWidth(180);

                gainLabels[i] = new Label(String.format("%.2fx", bitGain[i]));
                gainLabels[i].setMinWidth(50);

                gainSliders[i].valueProperty().addListener((obs, oldVal, newVal) -> {
                    double g = Math.pow(10, (newVal.doubleValue() - 100) / 100.0);
                    gainLabels[idx].setText(String.format("%.2fx", g));
                });

                row.getChildren().addAll(bitLabel, checkBoxes[i], gainSliders[i], gainLabels[i]);
                dialog.addCustomContent(row);
            }

            // Store control references
            node.properties.put("_bitCheckBoxes", checkBoxes);
            node.properties.put("_bitGainSliders", gainSliders);

        } else if ("BitPlanesColor".equals(node.nodeType)) {
            // Bit Planes Color - Tabbed interface with Red, Green, Blue channels
            dialog.addDescription("Bit Planes Color: Select and adjust RGB bit planes\nBit plane decomposition with gain (RGB)");

            // Load current values from properties
            boolean[][] bitEnabled = new boolean[3][8];
            double[][] bitGain = new double[3][8];
            String[] channelNames = {"red", "green", "blue"};

            for (int c = 0; c < 3; c++) {
                for (int i = 0; i < 8; i++) {
                    bitEnabled[c][i] = true;
                    bitGain[c][i] = 1.0;
                }
                String enabledKey = channelNames[c] + "BitEnabled";
                String gainKey = channelNames[c] + "BitGain";
                if (node.properties.containsKey(enabledKey)) {
                    boolean[] arr = (boolean[]) node.properties.get(enabledKey);
                    for (int i = 0; i < Math.min(arr.length, 8); i++) bitEnabled[c][i] = arr[i];
                }
                if (node.properties.containsKey(gainKey)) {
                    double[] arr = (double[]) node.properties.get(gainKey);
                    for (int i = 0; i < Math.min(arr.length, 8); i++) bitGain[c][i] = arr[i];
                }
            }

            // Create tabbed pane
            TabPane tabPane = new TabPane();
            tabPane.setTabClosingPolicy(TabPane.TabClosingPolicy.UNAVAILABLE);

            CheckBox[][] checkBoxes = new CheckBox[3][8];
            Slider[][] gainSliders = new Slider[3][8];

            String[] tabNames = {"Red", "Green", "Blue"};
            for (int c = 0; c < 3; c++) {
                Tab tab = new Tab(tabNames[c]);
                VBox tabContent = new VBox(5);
                tabContent.setPadding(new Insets(10));

                // Header
                HBox headerRow = new HBox(10);
                Label bitHeader = new Label("Bit");
                bitHeader.setMinWidth(30);
                Label onHeader = new Label("On");
                onHeader.setMinWidth(30);
                Label gainHeader = new Label("Gain (0.1x - 10x)");
                headerRow.getChildren().addAll(bitHeader, onHeader, gainHeader);
                tabContent.getChildren().add(headerRow);

                // 8 rows for bit planes
                for (int i = 0; i < 8; i++) {
                    int bitNum = 7 - i;
                    final int channel = c;
                    final int idx = i;

                    HBox row = new HBox(10);
                    row.setAlignment(javafx.geometry.Pos.CENTER_LEFT);

                    Label bitLabel = new Label(String.valueOf(bitNum));
                    bitLabel.setMinWidth(30);

                    checkBoxes[c][i] = new CheckBox();
                    checkBoxes[c][i].setSelected(bitEnabled[c][i]);

                    gainSliders[c][i] = new Slider(0, 200, Math.log10(bitGain[c][i]) * 100 + 100);
                    gainSliders[c][i].setPrefWidth(180);

                    Label gainLabel = new Label(String.format("%.2fx", bitGain[c][i]));
                    gainLabel.setMinWidth(50);

                    gainSliders[c][i].valueProperty().addListener((obs, oldVal, newVal) -> {
                        double g = Math.pow(10, (newVal.doubleValue() - 100) / 100.0);
                        gainLabel.setText(String.format("%.2fx", g));
                    });

                    row.getChildren().addAll(bitLabel, checkBoxes[c][i], gainSliders[c][i], gainLabel);
                    tabContent.getChildren().add(row);
                }

                tab.setContent(tabContent);
                tabPane.getTabs().add(tab);
            }

            dialog.addCustomContent(tabPane);

            // Store control references
            node.properties.put("_colorBitCheckBoxes", checkBoxes);
            node.properties.put("_colorBitGainSliders", gainSliders);

        } else if (isFFTNodeType(node.nodeType)) {
            // FFT Low-Pass / High-Pass filter properties
            System.out.println("[DEBUG] Creating FFT properties UI for nodeType: " + node.nodeType);
            int currentRadius = 100;
            int currentSmoothness = 0;
            if (node.properties.containsKey("radius")) {
                currentRadius = ((Number) node.properties.get("radius")).intValue();
            }
            if (node.properties.containsKey("smoothness")) {
                currentSmoothness = ((Number) node.properties.get("smoothness")).intValue();
            }

            // Radius slider (0-200)
            Slider radiusSlider = new Slider(0, 200, currentRadius);
            radiusSlider.setPrefWidth(200);
            radiusSlider.setShowTickMarks(true);
            radiusSlider.setMajorTickUnit(50);

            Label radiusValueLabel = new Label(String.valueOf(currentRadius));
            radiusValueLabel.setMinWidth(40);

            radiusSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
                radiusValueLabel.setText(String.valueOf(newVal.intValue()));
            });

            HBox radiusRow = new HBox(10);
            radiusRow.getChildren().addAll(new Label("Radius:"), radiusSlider, radiusValueLabel);
            dialog.addCustomContent(radiusRow);

            // Smoothness slider (0-100)
            Slider smoothnessSlider = new Slider(0, 100, currentSmoothness);
            smoothnessSlider.setPrefWidth(200);
            smoothnessSlider.setShowTickMarks(true);
            smoothnessSlider.setMajorTickUnit(25);

            Label smoothnessValueLabel = new Label(String.valueOf(currentSmoothness));
            smoothnessValueLabel.setMinWidth(40);

            smoothnessSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
                smoothnessValueLabel.setText(String.valueOf(newVal.intValue()));
            });

            HBox smoothnessRow = new HBox(10);
            smoothnessRow.getChildren().addAll(new Label("Smoothness:"), smoothnessSlider, smoothnessValueLabel);
            dialog.addCustomContent(smoothnessRow);

            // Add a note about performance (filter type/method signature already shown from registry)
            dialog.addDescription("Note: FFT processing is computationally expensive (it's slow!).");

            // Store slider references for OK handler
            node.properties.put("_radiusSlider", radiusSlider);
            node.properties.put("_smoothnessSlider", smoothnessSlider);
        } else if ("AdaptiveThreshold".equals(node.nodeType)) {
            // AdaptiveThreshold properties
            int maxValue = node.properties.containsKey("maxValue") ?
                ((Number) node.properties.get("maxValue")).intValue() : 255;
            int methodIndex = node.properties.containsKey("methodIndex") ?
                ((Number) node.properties.get("methodIndex")).intValue() : 1;
            int typeIndex = node.properties.containsKey("typeIndex") ?
                ((Number) node.properties.get("typeIndex")).intValue() : 0;
            int blockSize = node.properties.containsKey("blockSize") ?
                ((Number) node.properties.get("blockSize")).intValue() : 11;
            int cValue = node.properties.containsKey("cValue") ?
                ((Number) node.properties.get("cValue")).intValue() : 2;

            Slider maxValueSlider = dialog.addSlider("Max Value:", 0, 255, maxValue, "%.0f");

            String[] methods = {"Mean", "Gaussian"};
            ComboBox<String> methodCombo = dialog.addComboBox("Method:", methods, methods[methodIndex]);

            String[] types = {"Binary", "Binary Inv"};
            ComboBox<String> typeCombo = dialog.addComboBox("Type:", types, types[typeIndex]);

            // Block size must be odd and >= 3
            Slider blockSizeSlider = dialog.addSlider("Block Size:", 3, 99, blockSize, "%.0f");
            Slider cValueSlider = dialog.addSlider("C Value:", 0, 50, cValue, "%.0f");

            node.properties.put("_maxValueSlider", maxValueSlider);
            node.properties.put("_methodCombo", methodCombo);
            node.properties.put("_typeCombo", typeCombo);
            node.properties.put("_blockSizeSlider", blockSizeSlider);
            node.properties.put("_cValueSlider", cValueSlider);
        } else if ("AddWeighted".equals(node.nodeType)) {
            // AddWeighted properties (alpha, beta, gamma for blending)
            double alpha = node.properties.containsKey("alpha") ?
                ((Number) node.properties.get("alpha")).doubleValue() : 0.5;
            double beta = node.properties.containsKey("beta") ?
                ((Number) node.properties.get("beta")).doubleValue() : 0.5;
            double gamma = node.properties.containsKey("gamma") ?
                ((Number) node.properties.get("gamma")).doubleValue() : 0.0;

            Slider alphaSlider = dialog.addSlider("Alpha (Input 1 weight):", 0, 100, alpha * 100, "%.0f%%");
            Slider betaSlider = dialog.addSlider("Beta (Input 2 weight):", 0, 100, beta * 100, "%.0f%%");
            Slider gammaSlider = dialog.addSlider("Gamma (brightness):", 0, 255, gamma, "%.0f");

            node.properties.put("_alphaSlider", alphaSlider);
            node.properties.put("_betaSlider", betaSlider);
            node.properties.put("_gammaSlider", gammaSlider);
        } else if ("BilateralFilter".equals(node.nodeType)) {
            // BilateralFilter properties
            int diameter = node.properties.containsKey("diameter") ?
                ((Number) node.properties.get("diameter")).intValue() : 9;
            int sigmaColor = node.properties.containsKey("sigmaColor") ?
                ((Number) node.properties.get("sigmaColor")).intValue() : 75;
            int sigmaSpace = node.properties.containsKey("sigmaSpace") ?
                ((Number) node.properties.get("sigmaSpace")).intValue() : 75;

            Slider diameterSlider = dialog.addSlider("Diameter:", 1, 25, diameter, "%.0f");
            Slider sigmaColorSlider = dialog.addSlider("Sigma Color:", 1, 200, sigmaColor, "%.0f");
            Slider sigmaSpaceSlider = dialog.addSlider("Sigma Space:", 1, 200, sigmaSpace, "%.0f");

            node.properties.put("_diameterSlider", diameterSlider);
            node.properties.put("_sigmaColorSlider", sigmaColorSlider);
            node.properties.put("_sigmaSpaceSlider", sigmaSpaceSlider);
        } else if ("CLAHE".equals(node.nodeType)) {
            // CLAHE properties
            double clipLimit = node.properties.containsKey("clipLimit") ?
                ((Number) node.properties.get("clipLimit")).doubleValue() : 2.0;
            int tileSize = node.properties.containsKey("tileSize") ?
                ((Number) node.properties.get("tileSize")).intValue() : 8;
            int colorModeIndex = node.properties.containsKey("colorModeIndex") ?
                ((Number) node.properties.get("colorModeIndex")).intValue() : 0;

            Slider clipLimitSlider = dialog.addSlider("Clip Limit:", 1.0, 40.0, clipLimit, "%.1f");
            Slider tileSizeSlider = dialog.addSlider("Tile Size:", 2, 32, tileSize, "%.0f");

            String[] colorModes = {"LAB", "HSV", "Grayscale"};
            ComboBox<String> colorModeCombo = dialog.addComboBox("Color Mode:", colorModes, colorModes[colorModeIndex]);

            node.properties.put("_clipLimitSlider", clipLimitSlider);
            node.properties.put("_tileSizeSlider", tileSizeSlider);
            node.properties.put("_colorModeCombo", colorModeCombo);
        } else if ("ColorInRange".equals(node.nodeType)) {
            // ColorInRange properties
            boolean useHSV = node.properties.containsKey("useHSV") ?
                (Boolean) node.properties.get("useHSV") : true;
            int hLow = node.properties.containsKey("hLow") ?
                ((Number) node.properties.get("hLow")).intValue() : 0;
            int hHigh = node.properties.containsKey("hHigh") ?
                ((Number) node.properties.get("hHigh")).intValue() : 179;
            int sLow = node.properties.containsKey("sLow") ?
                ((Number) node.properties.get("sLow")).intValue() : 0;
            int sHigh = node.properties.containsKey("sHigh") ?
                ((Number) node.properties.get("sHigh")).intValue() : 255;
            int vLow = node.properties.containsKey("vLow") ?
                ((Number) node.properties.get("vLow")).intValue() : 0;
            int vHigh = node.properties.containsKey("vHigh") ?
                ((Number) node.properties.get("vHigh")).intValue() : 255;
            int outputMode = node.properties.containsKey("outputMode") ?
                ((Number) node.properties.get("outputMode")).intValue() : 0;

            CheckBox useHSVCheckBox = dialog.addCheckbox("Use HSV (unchecked = BGR)", useHSV);
            Slider hLowSlider = dialog.addSlider("H/B Low:", 0, 255, hLow, "%.0f");
            Slider hHighSlider = dialog.addSlider("H/B High:", 0, 255, hHigh, "%.0f");
            Slider sLowSlider = dialog.addSlider("S/G Low:", 0, 255, sLow, "%.0f");
            Slider sHighSlider = dialog.addSlider("S/G High:", 0, 255, sHigh, "%.0f");
            Slider vLowSlider = dialog.addSlider("V/R Low:", 0, 255, vLow, "%.0f");
            Slider vHighSlider = dialog.addSlider("V/R High:", 0, 255, vHigh, "%.0f");

            String[] outputModes = {"Mask Only", "Keep In-Range", "Keep Out-of-Range"};
            ComboBox<String> outputModeCombo = dialog.addComboBox("Output Mode:", outputModes, outputModes[outputMode]);

            node.properties.put("_useHSVCheckBox", useHSVCheckBox);
            node.properties.put("_hLowSlider", hLowSlider);
            node.properties.put("_hHighSlider", hHighSlider);
            node.properties.put("_sLowSlider", sLowSlider);
            node.properties.put("_sHighSlider", sHighSlider);
            node.properties.put("_vLowSlider", vLowSlider);
            node.properties.put("_vHighSlider", vHighSlider);
            node.properties.put("_outputModeCombo", outputModeCombo);
        } else if ("Crop".equals(node.nodeType)) {
            // Crop properties
            int cropX = node.properties.containsKey("cropX") ?
                ((Number) node.properties.get("cropX")).intValue() : 0;
            int cropY = node.properties.containsKey("cropY") ?
                ((Number) node.properties.get("cropY")).intValue() : 0;
            int cropWidth = node.properties.containsKey("cropWidth") ?
                ((Number) node.properties.get("cropWidth")).intValue() : 100;
            int cropHeight = node.properties.containsKey("cropHeight") ?
                ((Number) node.properties.get("cropHeight")).intValue() : 100;

            Spinner<Integer> xSpinner = dialog.addSpinner("X:", -4096, 4096, cropX);
            Spinner<Integer> ySpinner = dialog.addSpinner("Y:", -4096, 4096, cropY);
            Spinner<Integer> widthSpinner = dialog.addSpinner("Width:", 1, 4096, cropWidth);
            Spinner<Integer> heightSpinner = dialog.addSpinner("Height:", 1, 4096, cropHeight);

            node.properties.put("_cropXSpinner", xSpinner);
            node.properties.put("_cropYSpinner", ySpinner);
            node.properties.put("_cropWidthSpinner", widthSpinner);
            node.properties.put("_cropHeightSpinner", heightSpinner);
        } else if ("Sobel".equals(node.nodeType)) {
            // Sobel properties
            int dx = node.properties.containsKey("dx") ?
                ((Number) node.properties.get("dx")).intValue() : 1;
            int dy = node.properties.containsKey("dy") ?
                ((Number) node.properties.get("dy")).intValue() : 0;
            int kernelSizeIndex = node.properties.containsKey("kernelSizeIndex") ?
                ((Number) node.properties.get("kernelSizeIndex")).intValue() : 1;

            String[] derivOrders = {"0", "1", "2"};
            ToggleGroup dxGroup = dialog.addRadioButtons("dx (X derivative):", derivOrders, dx);
            ToggleGroup dyGroup = dialog.addRadioButtons("dy (Y derivative):", derivOrders, dy);
            String[] kernelSizes = {"1", "3", "5", "7"};
            ComboBox<String> kernelSizeCombo = dialog.addComboBox("Kernel Size:", kernelSizes, kernelSizes[kernelSizeIndex]);
            dialog.addDescription("Note: dx + dy must be >= 1");

            node.properties.put("_dxGroup", dxGroup);
            node.properties.put("_dyGroup", dyGroup);
            node.properties.put("_kernelSizeCombo", kernelSizeCombo);
        } else if ("Scharr".equals(node.nodeType)) {
            // Scharr properties
            int directionIndex = node.properties.containsKey("directionIndex") ?
                ((Number) node.properties.get("directionIndex")).intValue() : 2;
            int scalePercent = node.properties.containsKey("scalePercent") ?
                ((Number) node.properties.get("scalePercent")).intValue() : 100;
            int delta = node.properties.containsKey("delta") ?
                ((Number) node.properties.get("delta")).intValue() : 0;

            String[] directions = {"X", "Y", "Both"};
            ComboBox<String> dirCombo = dialog.addComboBox("Direction:", directions, directions[directionIndex]);
            Slider scaleSlider = dialog.addSlider("Scale (%):", 10, 500, scalePercent, "%.0f%%");
            Slider deltaSlider = dialog.addSlider("Delta:", 0, 255, delta, "%.0f");

            node.properties.put("_directionCombo", dirCombo);
            node.properties.put("_scaleSlider", scaleSlider);
            node.properties.put("_deltaSlider", deltaSlider);
        } else if ("Laplacian".equals(node.nodeType)) {
            // Laplacian properties
            int kernelSizeIndex = node.properties.containsKey("kernelSizeIndex") ?
                ((Number) node.properties.get("kernelSizeIndex")).intValue() : 1;
            int scalePercent = node.properties.containsKey("scalePercent") ?
                ((Number) node.properties.get("scalePercent")).intValue() : 100;
            int delta = node.properties.containsKey("delta") ?
                ((Number) node.properties.get("delta")).intValue() : 0;
            boolean useAbsolute = node.properties.containsKey("useAbsolute") ?
                (Boolean) node.properties.get("useAbsolute") : true;

            String[] ksizes = {"1", "3", "5", "7"};
            ComboBox<String> ksizeCombo = dialog.addComboBox("Kernel Size:", ksizes, ksizes[kernelSizeIndex]);
            Slider scaleSlider = dialog.addSlider("Scale (%):", 10, 500, scalePercent, "%.0f%%");
            Slider deltaSlider = dialog.addSlider("Delta:", 0, 255, delta, "%.0f");
            CheckBox absCheckBox = dialog.addCheckbox("Use Absolute Value", useAbsolute);

            node.properties.put("_ksizeCombo", ksizeCombo);
            node.properties.put("_scaleSlider", scaleSlider);
            node.properties.put("_deltaSlider", deltaSlider);
            node.properties.put("_absCheckBox", absCheckBox);
        } else if ("Rectangle".equals(node.nodeType)) {
            // Rectangle drawing properties
            int x1 = node.properties.containsKey("x1") ?
                ((Number) node.properties.get("x1")).intValue() : 50;
            int y1 = node.properties.containsKey("y1") ?
                ((Number) node.properties.get("y1")).intValue() : 50;
            int x2 = node.properties.containsKey("x2") ?
                ((Number) node.properties.get("x2")).intValue() : 200;
            int y2 = node.properties.containsKey("y2") ?
                ((Number) node.properties.get("y2")).intValue() : 150;
            int colorR = node.properties.containsKey("colorR") ?
                ((Number) node.properties.get("colorR")).intValue() : 0;
            int colorG = node.properties.containsKey("colorG") ?
                ((Number) node.properties.get("colorG")).intValue() : 255;
            int colorB = node.properties.containsKey("colorB") ?
                ((Number) node.properties.get("colorB")).intValue() : 0;
            int thickness = node.properties.containsKey("thickness") ?
                ((Number) node.properties.get("thickness")).intValue() : 2;
            boolean filled = node.properties.containsKey("filled") ?
                (Boolean) node.properties.get("filled") : false;

            Spinner<Integer> x1Spinner = dialog.addSpinner("X1:", -4096, 4096, x1);
            Spinner<Integer> y1Spinner = dialog.addSpinner("Y1:", -4096, 4096, y1);
            Spinner<Integer> x2Spinner = dialog.addSpinner("X2:", -4096, 4096, x2);
            Spinner<Integer> y2Spinner = dialog.addSpinner("Y2:", -4096, 4096, y2);
            Spinner<Integer> rSpinner = dialog.addSpinner("Color R:", 0, 255, colorR);
            Spinner<Integer> gSpinner = dialog.addSpinner("Color G:", 0, 255, colorG);
            Spinner<Integer> bSpinner = dialog.addSpinner("Color B:", 0, 255, colorB);
            Spinner<Integer> thicknessSpinner = dialog.addSpinner("Thickness:", 1, 50, thickness);
            CheckBox filledCheckBox = dialog.addCheckbox("Filled", filled);

            node.properties.put("_x1Spinner", x1Spinner);
            node.properties.put("_y1Spinner", y1Spinner);
            node.properties.put("_x2Spinner", x2Spinner);
            node.properties.put("_y2Spinner", y2Spinner);
            node.properties.put("_colorRSpinner", rSpinner);
            node.properties.put("_colorGSpinner", gSpinner);
            node.properties.put("_colorBSpinner", bSpinner);
            node.properties.put("_thicknessSpinner", thicknessSpinner);
            node.properties.put("_filledCheckBox", filledCheckBox);
        } else if ("Circle".equals(node.nodeType)) {
            // Circle drawing properties
            int centerX = node.properties.containsKey("centerX") ?
                ((Number) node.properties.get("centerX")).intValue() : 100;
            int centerY = node.properties.containsKey("centerY") ?
                ((Number) node.properties.get("centerY")).intValue() : 100;
            int radius = node.properties.containsKey("radius") ?
                ((Number) node.properties.get("radius")).intValue() : 50;
            int colorR = node.properties.containsKey("colorR") ?
                ((Number) node.properties.get("colorR")).intValue() : 0;
            int colorG = node.properties.containsKey("colorG") ?
                ((Number) node.properties.get("colorG")).intValue() : 255;
            int colorB = node.properties.containsKey("colorB") ?
                ((Number) node.properties.get("colorB")).intValue() : 0;
            int thickness = node.properties.containsKey("thickness") ?
                ((Number) node.properties.get("thickness")).intValue() : 2;
            boolean filled = node.properties.containsKey("filled") ?
                (Boolean) node.properties.get("filled") : false;

            Spinner<Integer> cxSpinner = dialog.addSpinner("Center X:", -4096, 4096, centerX);
            Spinner<Integer> cySpinner = dialog.addSpinner("Center Y:", -4096, 4096, centerY);
            Spinner<Integer> radiusSpinner = dialog.addSpinner("Radius:", 1, 2000, radius);
            Spinner<Integer> rSpinner = dialog.addSpinner("Color R:", 0, 255, colorR);
            Spinner<Integer> gSpinner = dialog.addSpinner("Color G:", 0, 255, colorG);
            Spinner<Integer> bSpinner = dialog.addSpinner("Color B:", 0, 255, colorB);
            Spinner<Integer> thicknessSpinner = dialog.addSpinner("Thickness:", 1, 50, thickness);
            CheckBox filledCheckBox = dialog.addCheckbox("Filled", filled);

            node.properties.put("_centerXSpinner", cxSpinner);
            node.properties.put("_centerYSpinner", cySpinner);
            node.properties.put("_radiusSpinner", radiusSpinner);
            node.properties.put("_colorRSpinner", rSpinner);
            node.properties.put("_colorGSpinner", gSpinner);
            node.properties.put("_colorBSpinner", bSpinner);
            node.properties.put("_thicknessSpinner", thicknessSpinner);
            node.properties.put("_filledCheckBox", filledCheckBox);
        } else if ("Grayscale".equals(node.nodeType)) {
            // Grayscale/Color Convert properties
            int conversionIndex = node.properties.containsKey("conversionIndex") ?
                ((Number) node.properties.get("conversionIndex")).intValue() : 0;

            String[] conversions = {"BGR to Gray", "BGR to HSV", "BGR to LAB", "BGR to RGB",
                                   "Gray to BGR", "HSV to BGR", "LAB to BGR", "RGB to BGR"};
            ComboBox<String> conversionCombo = dialog.addComboBox("Conversion:", conversions,
                conversions[Math.min(conversionIndex, conversions.length - 1)]);

            node.properties.put("_conversionCombo", conversionCombo);
        }

        // Add "Queues in Sync" checkbox for dual-input nodes
        // Check both the flag and the node type (in case loaded from old saved file)
        CheckBox syncCheckBox = null;
        boolean isDualInput = node.hasDualInput || isDualInputNodeType(node.nodeType);
        if (isDualInput) {
            syncCheckBox = dialog.addCheckbox("Queues in Sync", node.queuesInSync);
            dialog.addDescription("When enabled, wait for new data on both\ninputs before processing (synchronized mode).");
        }

        // Set OK handler to save values
        final Spinner<Integer> finalCameraSpinner = cameraSpinner;
        final TextField finalFileField = filePathField;
        final TextField finalPipelineField = pipelineFileField;
        final ComboBox<String> finalFpsCombo = fpsCombo;
        final CheckBox finalSyncCheckBox = syncCheckBox;
        dialog.setOnOk(() -> {
            node.label = dialog.getNameValue();

            // Save properties handled by the helper class (covers all nodes with main branch differences)
            com.ttennebkram.pipeline.fx.FXNodePropertiesHelper.savePropertiesForNode(node);

            // Handle webcam-specific properties
            if ("WebcamSource".equals(node.nodeType) && finalCameraSpinner != null) {
                int newCameraIndex = finalCameraSpinner.getValue();
                if (newCameraIndex != node.cameraIndex) {
                    node.cameraIndex = newCameraIndex;
                    // Restart webcam with new camera index
                    stopWebcamForNode(node);
                    startWebcamForNode(node);
                }
            }

            // Handle file source properties
            if ("FileSource".equals(node.nodeType) && finalFileField != null) {
                String newPath = finalFileField.getText().trim();
                if (!newPath.equals(node.filePath)) {
                    node.filePath = newPath;
                    loadFileImageForNode(node);
                }
                // Handle FPS setting
                if (finalFpsCombo != null) {
                    String fpsValue = finalFpsCombo.getValue();
                    if ("Automatic".equals(fpsValue) || fpsValue == null || fpsValue.isEmpty()) {
                        node.fps = -1.0;
                    } else {
                        try {
                            node.fps = Double.parseDouble(fpsValue);
                        } catch (NumberFormatException e) {
                            node.fps = -1.0;  // Default to automatic on parse error
                        }
                    }
                }
            }

            // Handle container properties
            if ("Container".equals(node.nodeType) && finalPipelineField != null) {
                String newPath = finalPipelineField.getText().trim();
                node.pipelineFilePath = newPath;
            }

            // Handle BlankSource properties
            if ("BlankSource".equals(node.nodeType) && node.properties.containsKey("_widthSpinner")) {
                @SuppressWarnings("unchecked")
                Spinner<Integer> widthSpinner = (Spinner<Integer>) node.properties.get("_widthSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> heightSpinner = (Spinner<Integer>) node.properties.get("_heightSpinner");
                @SuppressWarnings("unchecked")
                ComboBox<String> colorCombo = (ComboBox<String>) node.properties.get("_colorCombo");
                @SuppressWarnings("unchecked")
                ComboBox<String> blankFpsCombo = (ComboBox<String>) node.properties.get("_blankFpsCombo");

                node.properties.put("imageWidth", widthSpinner.getValue());
                node.properties.put("imageHeight", heightSpinner.getValue());

                // Get color index from combo box selection
                String[] colorOptions = {"Black", "White", "Red", "Green", "Blue", "Yellow"};
                int colorIndex = 0;
                String selectedColor = colorCombo.getValue();
                for (int i = 0; i < colorOptions.length; i++) {
                    if (colorOptions[i].equals(selectedColor)) {
                        colorIndex = i;
                        break;
                    }
                }
                node.properties.put("colorIndex", colorIndex);

                // Get fps index from combo box selection
                String[] fpsOptions = {"1", "15", "30", "60"};
                int fpsIndex = 2;  // default to 30 fps
                String selectedFps = blankFpsCombo.getValue();
                for (int i = 0; i < fpsOptions.length; i++) {
                    if (fpsOptions[i].equals(selectedFps)) {
                        fpsIndex = i;
                        break;
                    }
                }
                node.properties.put("fpsIndex", fpsIndex);

                // Clean up temp references
                node.properties.remove("_widthSpinner");
                node.properties.remove("_heightSpinner");
                node.properties.remove("_colorCombo");
                node.properties.remove("_blankFpsCombo");
            }

            // Handle Gain properties
            if ("Gain".equals(node.nodeType) && node.properties.containsKey("_gainSlider")) {
                Slider gainSlider = (Slider) node.properties.get("_gainSlider");
                double logRange = (Double) node.properties.getOrDefault("_gainLogRange", Math.log10(20.0));
                double logVal = (gainSlider.getValue() - 50) / 50.0 * logRange;
                double gain = Math.pow(10, logVal);
                node.properties.put("gain", gain);
                node.properties.remove("_gainSlider");  // Clean up temp reference
                node.properties.remove("_gainLogRange");
            }

            // Handle FFT filter properties
            if (isFFTNodeType(node.nodeType) && node.properties.containsKey("_radiusSlider")) {
                Slider radiusSlider = (Slider) node.properties.get("_radiusSlider");
                Slider smoothnessSlider = (Slider) node.properties.get("_smoothnessSlider");
                node.properties.put("radius", (int) radiusSlider.getValue());
                node.properties.put("smoothness", (int) smoothnessSlider.getValue());
                node.properties.remove("_radiusSlider");  // Clean up temp references
                node.properties.remove("_smoothnessSlider");
            }

            // Handle MedianBlur properties
            if ("MedianBlur".equals(node.nodeType) && node.properties.containsKey("_ksizeSlider")) {
                Slider ksizeSlider = (Slider) node.properties.get("_ksizeSlider");
                int ksize = (int) ksizeSlider.getValue();
                if (ksize % 2 == 0) ksize++;  // Must be odd
                node.properties.put("ksize", ksize);
                node.properties.remove("_ksizeSlider");
            }

            // Handle Erode/Dilate properties
            if (("Erode".equals(node.nodeType) || "Dilate".equals(node.nodeType)) && node.properties.containsKey("_kernelSizeSlider")) {
                Slider ksizeSlider = (Slider) node.properties.get("_kernelSizeSlider");
                Slider iterSlider = (Slider) node.properties.get("_iterationsSlider");
                node.properties.put("kernelSize", (int) ksizeSlider.getValue());
                node.properties.put("iterations", (int) iterSlider.getValue());
                node.properties.remove("_kernelSizeSlider");
                node.properties.remove("_iterationsSlider");
            }

            // Handle MorphOpen/MorphClose properties
            if (("MorphOpen".equals(node.nodeType) || "MorphClose".equals(node.nodeType)) && node.properties.containsKey("_kernelSizeSlider")) {
                Slider ksizeSlider = (Slider) node.properties.get("_kernelSizeSlider");
                node.properties.put("kernelSize", (int) ksizeSlider.getValue());
                node.properties.remove("_kernelSizeSlider");
            }

            // Handle BitPlanesGrayscale properties
            if ("BitPlanesGrayscale".equals(node.nodeType) && node.properties.containsKey("_bitCheckBoxes")) {
                CheckBox[] checkBoxes = (CheckBox[]) node.properties.get("_bitCheckBoxes");
                Slider[] gainSliders = (Slider[]) node.properties.get("_bitGainSliders");
                boolean[] bitEnabled = new boolean[8];
                double[] bitGain = new double[8];
                for (int i = 0; i < 8; i++) {
                    bitEnabled[i] = checkBoxes[i].isSelected();
                    bitGain[i] = Math.pow(10, (gainSliders[i].getValue() - 100) / 100.0);
                }
                node.properties.put("bitEnabled", bitEnabled);
                node.properties.put("bitGain", bitGain);
                node.properties.remove("_bitCheckBoxes");
                node.properties.remove("_bitGainSliders");
            }

            // Handle BitPlanesColor properties
            if ("BitPlanesColor".equals(node.nodeType) && node.properties.containsKey("_colorBitCheckBoxes")) {
                CheckBox[][] checkBoxes = (CheckBox[][]) node.properties.get("_colorBitCheckBoxes");
                Slider[][] gainSliders = (Slider[][]) node.properties.get("_colorBitGainSliders");
                String[] channelNames = {"red", "green", "blue"};
                for (int c = 0; c < 3; c++) {
                    boolean[] bitEnabled = new boolean[8];
                    double[] bitGain = new double[8];
                    for (int i = 0; i < 8; i++) {
                        bitEnabled[i] = checkBoxes[c][i].isSelected();
                        bitGain[i] = Math.pow(10, (gainSliders[c][i].getValue() - 100) / 100.0);
                    }
                    node.properties.put(channelNames[c] + "BitEnabled", bitEnabled);
                    node.properties.put(channelNames[c] + "BitGain", bitGain);
                }
                node.properties.remove("_colorBitCheckBoxes");
                node.properties.remove("_colorBitGainSliders");
            }

            // Handle AdaptiveThreshold properties
            if ("AdaptiveThreshold".equals(node.nodeType) && node.properties.containsKey("_maxValueSlider")) {
                Slider maxValueSlider = (Slider) node.properties.get("_maxValueSlider");
                @SuppressWarnings("unchecked")
                ComboBox<String> methodCombo = (ComboBox<String>) node.properties.get("_methodCombo");
                @SuppressWarnings("unchecked")
                ComboBox<String> typeCombo = (ComboBox<String>) node.properties.get("_typeCombo");
                Slider blockSizeSlider = (Slider) node.properties.get("_blockSizeSlider");
                Slider cValueSlider = (Slider) node.properties.get("_cValueSlider");

                node.properties.put("maxValue", (int) maxValueSlider.getValue());
                String[] methods = {"Mean", "Gaussian"};
                int methodIndex = java.util.Arrays.asList(methods).indexOf(methodCombo.getValue());
                node.properties.put("methodIndex", Math.max(0, methodIndex));
                String[] types = {"Binary", "Binary Inv"};
                int typeIndex = java.util.Arrays.asList(types).indexOf(typeCombo.getValue());
                node.properties.put("typeIndex", Math.max(0, typeIndex));
                int blockSize = (int) blockSizeSlider.getValue();
                if (blockSize % 2 == 0) blockSize++;
                if (blockSize < 3) blockSize = 3;
                node.properties.put("blockSize", blockSize);
                node.properties.put("cValue", (int) cValueSlider.getValue());

                node.properties.remove("_maxValueSlider");
                node.properties.remove("_methodCombo");
                node.properties.remove("_typeCombo");
                node.properties.remove("_blockSizeSlider");
                node.properties.remove("_cValueSlider");
            }

            // Handle AddWeighted properties
            if ("AddWeighted".equals(node.nodeType) && node.properties.containsKey("_alphaSlider")) {
                Slider alphaSlider = (Slider) node.properties.get("_alphaSlider");
                Slider betaSlider = (Slider) node.properties.get("_betaSlider");
                Slider gammaSlider = (Slider) node.properties.get("_gammaSlider");
                node.properties.put("alpha", alphaSlider.getValue() / 100.0);
                node.properties.put("beta", betaSlider.getValue() / 100.0);
                node.properties.put("gamma", gammaSlider.getValue());
                node.properties.remove("_alphaSlider");
                node.properties.remove("_betaSlider");
                node.properties.remove("_gammaSlider");
            }

            // Handle BilateralFilter properties
            if ("BilateralFilter".equals(node.nodeType) && node.properties.containsKey("_diameterSlider")) {
                Slider diameterSlider = (Slider) node.properties.get("_diameterSlider");
                Slider sigmaColorSlider = (Slider) node.properties.get("_sigmaColorSlider");
                Slider sigmaSpaceSlider = (Slider) node.properties.get("_sigmaSpaceSlider");
                node.properties.put("diameter", (int) diameterSlider.getValue());
                node.properties.put("sigmaColor", (int) sigmaColorSlider.getValue());
                node.properties.put("sigmaSpace", (int) sigmaSpaceSlider.getValue());
                node.properties.remove("_diameterSlider");
                node.properties.remove("_sigmaColorSlider");
                node.properties.remove("_sigmaSpaceSlider");
            }

            // Handle CLAHE properties
            if ("CLAHE".equals(node.nodeType) && node.properties.containsKey("_clipLimitSlider")) {
                Slider clipLimitSlider = (Slider) node.properties.get("_clipLimitSlider");
                Slider tileSizeSlider = (Slider) node.properties.get("_tileSizeSlider");
                @SuppressWarnings("unchecked")
                ComboBox<String> colorModeCombo = (ComboBox<String>) node.properties.get("_colorModeCombo");
                node.properties.put("clipLimit", clipLimitSlider.getValue());
                node.properties.put("tileSize", (int) tileSizeSlider.getValue());
                String[] colorModes = {"LAB", "HSV", "Grayscale"};
                int colorModeIndex = java.util.Arrays.asList(colorModes).indexOf(colorModeCombo.getValue());
                node.properties.put("colorModeIndex", Math.max(0, colorModeIndex));
                node.properties.remove("_clipLimitSlider");
                node.properties.remove("_tileSizeSlider");
                node.properties.remove("_colorModeCombo");
            }

            // Handle ColorInRange properties
            if ("ColorInRange".equals(node.nodeType) && node.properties.containsKey("_useHSVCheckBox")) {
                CheckBox useHSVCheckBox = (CheckBox) node.properties.get("_useHSVCheckBox");
                Slider hLowSlider = (Slider) node.properties.get("_hLowSlider");
                Slider hHighSlider = (Slider) node.properties.get("_hHighSlider");
                Slider sLowSlider = (Slider) node.properties.get("_sLowSlider");
                Slider sHighSlider = (Slider) node.properties.get("_sHighSlider");
                Slider vLowSlider = (Slider) node.properties.get("_vLowSlider");
                Slider vHighSlider = (Slider) node.properties.get("_vHighSlider");
                @SuppressWarnings("unchecked")
                ComboBox<String> outputModeCombo = (ComboBox<String>) node.properties.get("_outputModeCombo");
                node.properties.put("useHSV", useHSVCheckBox.isSelected());
                node.properties.put("hLow", (int) hLowSlider.getValue());
                node.properties.put("hHigh", (int) hHighSlider.getValue());
                node.properties.put("sLow", (int) sLowSlider.getValue());
                node.properties.put("sHigh", (int) sHighSlider.getValue());
                node.properties.put("vLow", (int) vLowSlider.getValue());
                node.properties.put("vHigh", (int) vHighSlider.getValue());
                String[] outputModes = {"Mask Only", "Keep In-Range", "Keep Out-of-Range"};
                int outputMode = java.util.Arrays.asList(outputModes).indexOf(outputModeCombo.getValue());
                node.properties.put("outputMode", Math.max(0, outputMode));
                node.properties.remove("_useHSVCheckBox");
                node.properties.remove("_hLowSlider");
                node.properties.remove("_hHighSlider");
                node.properties.remove("_sLowSlider");
                node.properties.remove("_sHighSlider");
                node.properties.remove("_vLowSlider");
                node.properties.remove("_vHighSlider");
                node.properties.remove("_outputModeCombo");
            }

            // Handle Crop properties
            if ("Crop".equals(node.nodeType) && node.properties.containsKey("_cropXSpinner")) {
                @SuppressWarnings("unchecked")
                Spinner<Integer> xSpinner = (Spinner<Integer>) node.properties.get("_cropXSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> ySpinner = (Spinner<Integer>) node.properties.get("_cropYSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> widthSpinner = (Spinner<Integer>) node.properties.get("_cropWidthSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> heightSpinner = (Spinner<Integer>) node.properties.get("_cropHeightSpinner");
                node.properties.put("cropX", xSpinner.getValue());
                node.properties.put("cropY", ySpinner.getValue());
                node.properties.put("cropWidth", widthSpinner.getValue());
                node.properties.put("cropHeight", heightSpinner.getValue());
                node.properties.remove("_cropXSpinner");
                node.properties.remove("_cropYSpinner");
                node.properties.remove("_cropWidthSpinner");
                node.properties.remove("_cropHeightSpinner");
            }

            // Handle Sobel properties
            if ("Sobel".equals(node.nodeType) && node.properties.containsKey("_dxGroup")) {
                ToggleGroup dxGroup = (ToggleGroup) node.properties.get("_dxGroup");
                ToggleGroup dyGroup = (ToggleGroup) node.properties.get("_dyGroup");
                @SuppressWarnings("unchecked")
                ComboBox<String> kernelSizeCombo = (ComboBox<String>) node.properties.get("_kernelSizeCombo");
                // Get selected index from radio button groups (stored in userData)
                int dxVal = dxGroup.getSelectedToggle() != null ? (Integer) dxGroup.getSelectedToggle().getUserData() : 1;
                int dyVal = dyGroup.getSelectedToggle() != null ? (Integer) dyGroup.getSelectedToggle().getUserData() : 0;
                node.properties.put("dx", dxVal);
                node.properties.put("dy", dyVal);
                String[] kernelSizes = {"1", "3", "5", "7"};
                int kernelSizeIndex = java.util.Arrays.asList(kernelSizes).indexOf(kernelSizeCombo.getValue());
                node.properties.put("kernelSizeIndex", Math.max(0, kernelSizeIndex));
                node.properties.remove("_dxGroup");
                node.properties.remove("_dyGroup");
                node.properties.remove("_kernelSizeCombo");
            }

            // Handle Scharr properties
            if ("Scharr".equals(node.nodeType) && node.properties.containsKey("_directionCombo")) {
                @SuppressWarnings("unchecked")
                ComboBox<String> dirCombo = (ComboBox<String>) node.properties.get("_directionCombo");
                Slider scaleSlider = (Slider) node.properties.get("_scaleSlider");
                Slider deltaSlider = (Slider) node.properties.get("_deltaSlider");
                String[] directions = {"X", "Y", "Both"};
                int directionIndex = java.util.Arrays.asList(directions).indexOf(dirCombo.getValue());
                node.properties.put("directionIndex", Math.max(0, directionIndex));
                node.properties.put("scalePercent", (int) scaleSlider.getValue());
                node.properties.put("delta", (int) deltaSlider.getValue());
                node.properties.remove("_directionCombo");
                node.properties.remove("_scaleSlider");
                node.properties.remove("_deltaSlider");
            }

            // Handle Laplacian properties
            if ("Laplacian".equals(node.nodeType) && node.properties.containsKey("_ksizeCombo")) {
                @SuppressWarnings("unchecked")
                ComboBox<String> ksizeCombo = (ComboBox<String>) node.properties.get("_ksizeCombo");
                Slider scaleSlider = (Slider) node.properties.get("_scaleSlider");
                Slider deltaSlider = (Slider) node.properties.get("_deltaSlider");
                CheckBox absCheckBox = (CheckBox) node.properties.get("_absCheckBox");
                String[] ksizes = {"1", "3", "5", "7"};
                int ksizeIndex = java.util.Arrays.asList(ksizes).indexOf(ksizeCombo.getValue());
                node.properties.put("kernelSizeIndex", Math.max(0, ksizeIndex));
                node.properties.put("scalePercent", (int) scaleSlider.getValue());
                node.properties.put("delta", (int) deltaSlider.getValue());
                node.properties.put("useAbsolute", absCheckBox.isSelected());
                node.properties.remove("_ksizeCombo");
                node.properties.remove("_scaleSlider");
                node.properties.remove("_deltaSlider");
                node.properties.remove("_absCheckBox");
            }

            // Handle HoughCircles properties
            if ("HoughCircles".equals(node.nodeType) && node.properties.containsKey("_showOrigCheckBox")) {
                CheckBox showOrigCheckBox = (CheckBox) node.properties.get("_showOrigCheckBox");
                Slider minDistSlider = (Slider) node.properties.get("_minDistSlider");
                Slider param1Slider = (Slider) node.properties.get("_param1Slider");
                Slider param2Slider = (Slider) node.properties.get("_param2Slider");
                Slider minRadiusSlider = (Slider) node.properties.get("_minRadiusSlider");
                Slider maxRadiusSlider = (Slider) node.properties.get("_maxRadiusSlider");
                Slider thicknessSlider = (Slider) node.properties.get("_thicknessSlider");
                @SuppressWarnings("unchecked")
                Spinner<Integer> rSpinner = (Spinner<Integer>) node.properties.get("_colorRSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> gSpinner = (Spinner<Integer>) node.properties.get("_colorGSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> bSpinner = (Spinner<Integer>) node.properties.get("_colorBSpinner");
                node.properties.put("showOriginal", showOrigCheckBox.isSelected());
                node.properties.put("minDist", (int) minDistSlider.getValue());
                node.properties.put("param1", (int) param1Slider.getValue());
                node.properties.put("param2", (int) param2Slider.getValue());
                node.properties.put("minRadius", (int) minRadiusSlider.getValue());
                node.properties.put("maxRadius", (int) maxRadiusSlider.getValue());
                node.properties.put("thickness", (int) thicknessSlider.getValue());
                node.properties.put("colorR", rSpinner.getValue());
                node.properties.put("colorG", gSpinner.getValue());
                node.properties.put("colorB", bSpinner.getValue());
                node.properties.remove("_showOrigCheckBox");
                node.properties.remove("_minDistSlider");
                node.properties.remove("_param1Slider");
                node.properties.remove("_param2Slider");
                node.properties.remove("_minRadiusSlider");
                node.properties.remove("_maxRadiusSlider");
                node.properties.remove("_thicknessSlider");
                node.properties.remove("_colorRSpinner");
                node.properties.remove("_colorGSpinner");
                node.properties.remove("_colorBSpinner");
            }

            // Handle HoughLines properties
            if ("HoughLines".equals(node.nodeType) && node.properties.containsKey("_thresholdSlider")) {
                Slider thresholdSlider = (Slider) node.properties.get("_thresholdSlider");
                Slider minLengthSlider = (Slider) node.properties.get("_minLengthSlider");
                Slider maxGapSlider = (Slider) node.properties.get("_maxGapSlider");
                Slider thicknessSlider = (Slider) node.properties.get("_thicknessSlider");
                @SuppressWarnings("unchecked")
                Spinner<Integer> rSpinner = (Spinner<Integer>) node.properties.get("_colorRSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> gSpinner = (Spinner<Integer>) node.properties.get("_colorGSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> bSpinner = (Spinner<Integer>) node.properties.get("_colorBSpinner");
                node.properties.put("threshold", (int) thresholdSlider.getValue());
                node.properties.put("minLineLength", (int) minLengthSlider.getValue());
                node.properties.put("maxLineGap", (int) maxGapSlider.getValue());
                node.properties.put("thickness", (int) thicknessSlider.getValue());
                node.properties.put("colorR", rSpinner.getValue());
                node.properties.put("colorG", gSpinner.getValue());
                node.properties.put("colorB", bSpinner.getValue());
                node.properties.remove("_thresholdSlider");
                node.properties.remove("_minLengthSlider");
                node.properties.remove("_maxGapSlider");
                node.properties.remove("_thicknessSlider");
                node.properties.remove("_colorRSpinner");
                node.properties.remove("_colorGSpinner");
                node.properties.remove("_colorBSpinner");
            }

            // Handle HarrisCorners properties
            if ("HarrisCorners".equals(node.nodeType) && node.properties.containsKey("_showOrigCheckBox")) {
                CheckBox showOrigCheckBox = (CheckBox) node.properties.get("_showOrigCheckBox");
                Slider blockSizeSlider = (Slider) node.properties.get("_blockSizeSlider");
                @SuppressWarnings("unchecked")
                ComboBox<String> ksizeCombo = (ComboBox<String>) node.properties.get("_ksizeCombo");
                Slider kSlider = (Slider) node.properties.get("_kSlider");
                Slider threshSlider = (Slider) node.properties.get("_threshSlider");
                Slider markerSlider = (Slider) node.properties.get("_markerSlider");
                @SuppressWarnings("unchecked")
                Spinner<Integer> rSpinner = (Spinner<Integer>) node.properties.get("_colorRSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> gSpinner = (Spinner<Integer>) node.properties.get("_colorGSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> bSpinner = (Spinner<Integer>) node.properties.get("_colorBSpinner");
                node.properties.put("showOriginal", showOrigCheckBox.isSelected());
                node.properties.put("blockSize", (int) blockSizeSlider.getValue());
                node.properties.put("ksize", Integer.parseInt(ksizeCombo.getValue()));
                node.properties.put("kPercent", (int) kSlider.getValue());
                node.properties.put("thresholdPercent", (int) threshSlider.getValue());
                node.properties.put("markerSize", (int) markerSlider.getValue());
                node.properties.put("colorR", rSpinner.getValue());
                node.properties.put("colorG", gSpinner.getValue());
                node.properties.put("colorB", bSpinner.getValue());
                node.properties.remove("_showOrigCheckBox");
                node.properties.remove("_blockSizeSlider");
                node.properties.remove("_ksizeCombo");
                node.properties.remove("_kSlider");
                node.properties.remove("_threshSlider");
                node.properties.remove("_markerSlider");
                node.properties.remove("_colorRSpinner");
                node.properties.remove("_colorGSpinner");
                node.properties.remove("_colorBSpinner");
            }

            // Handle Contours properties
            if ("Contours".equals(node.nodeType) && node.properties.containsKey("_showOrigCheckBox")) {
                CheckBox showOrigCheckBox = (CheckBox) node.properties.get("_showOrigCheckBox");
                Slider thicknessSlider = (Slider) node.properties.get("_thicknessSlider");
                @SuppressWarnings("unchecked")
                Spinner<Integer> rSpinner = (Spinner<Integer>) node.properties.get("_colorRSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> gSpinner = (Spinner<Integer>) node.properties.get("_colorGSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> bSpinner = (Spinner<Integer>) node.properties.get("_colorBSpinner");
                node.properties.put("showOriginal", showOrigCheckBox.isSelected());
                node.properties.put("thickness", (int) thicknessSlider.getValue());
                node.properties.put("colorR", rSpinner.getValue());
                node.properties.put("colorG", gSpinner.getValue());
                node.properties.put("colorB", bSpinner.getValue());
                node.properties.remove("_showOrigCheckBox");
                node.properties.remove("_thicknessSlider");
                node.properties.remove("_colorRSpinner");
                node.properties.remove("_colorGSpinner");
                node.properties.remove("_colorBSpinner");
            }

            // Handle Rectangle properties
            if ("Rectangle".equals(node.nodeType) && node.properties.containsKey("_x1Spinner")) {
                @SuppressWarnings("unchecked")
                Spinner<Integer> x1Spinner = (Spinner<Integer>) node.properties.get("_x1Spinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> y1Spinner = (Spinner<Integer>) node.properties.get("_y1Spinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> x2Spinner = (Spinner<Integer>) node.properties.get("_x2Spinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> y2Spinner = (Spinner<Integer>) node.properties.get("_y2Spinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> rSpinner = (Spinner<Integer>) node.properties.get("_colorRSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> gSpinner = (Spinner<Integer>) node.properties.get("_colorGSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> bSpinner = (Spinner<Integer>) node.properties.get("_colorBSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> thicknessSpinner = (Spinner<Integer>) node.properties.get("_thicknessSpinner");
                CheckBox filledCheckBox = (CheckBox) node.properties.get("_filledCheckBox");
                node.properties.put("x1", x1Spinner.getValue());
                node.properties.put("y1", y1Spinner.getValue());
                node.properties.put("x2", x2Spinner.getValue());
                node.properties.put("y2", y2Spinner.getValue());
                node.properties.put("colorR", rSpinner.getValue());
                node.properties.put("colorG", gSpinner.getValue());
                node.properties.put("colorB", bSpinner.getValue());
                node.properties.put("thickness", thicknessSpinner.getValue());
                node.properties.put("filled", filledCheckBox.isSelected());
                node.properties.remove("_x1Spinner");
                node.properties.remove("_y1Spinner");
                node.properties.remove("_x2Spinner");
                node.properties.remove("_y2Spinner");
                node.properties.remove("_colorRSpinner");
                node.properties.remove("_colorGSpinner");
                node.properties.remove("_colorBSpinner");
                node.properties.remove("_thicknessSpinner");
                node.properties.remove("_filledCheckBox");
            }

            // Handle Circle properties
            if ("Circle".equals(node.nodeType) && node.properties.containsKey("_centerXSpinner")) {
                @SuppressWarnings("unchecked")
                Spinner<Integer> cxSpinner = (Spinner<Integer>) node.properties.get("_centerXSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> cySpinner = (Spinner<Integer>) node.properties.get("_centerYSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> radiusSpinner = (Spinner<Integer>) node.properties.get("_radiusSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> rSpinner = (Spinner<Integer>) node.properties.get("_colorRSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> gSpinner = (Spinner<Integer>) node.properties.get("_colorGSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> bSpinner = (Spinner<Integer>) node.properties.get("_colorBSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> thicknessSpinner = (Spinner<Integer>) node.properties.get("_thicknessSpinner");
                CheckBox filledCheckBox = (CheckBox) node.properties.get("_filledCheckBox");
                node.properties.put("centerX", cxSpinner.getValue());
                node.properties.put("centerY", cySpinner.getValue());
                node.properties.put("radius", radiusSpinner.getValue());
                node.properties.put("colorR", rSpinner.getValue());
                node.properties.put("colorG", gSpinner.getValue());
                node.properties.put("colorB", bSpinner.getValue());
                node.properties.put("thickness", thicknessSpinner.getValue());
                node.properties.put("filled", filledCheckBox.isSelected());
                node.properties.remove("_centerXSpinner");
                node.properties.remove("_centerYSpinner");
                node.properties.remove("_radiusSpinner");
                node.properties.remove("_colorRSpinner");
                node.properties.remove("_colorGSpinner");
                node.properties.remove("_colorBSpinner");
                node.properties.remove("_thicknessSpinner");
                node.properties.remove("_filledCheckBox");
            }

            // Handle Ellipse properties
            if ("Ellipse".equals(node.nodeType) && node.properties.containsKey("_centerXSpinner")) {
                @SuppressWarnings("unchecked")
                Spinner<Integer> cxSpinner = (Spinner<Integer>) node.properties.get("_centerXSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> cySpinner = (Spinner<Integer>) node.properties.get("_centerYSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> axisXSpinner = (Spinner<Integer>) node.properties.get("_axisXSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> axisYSpinner = (Spinner<Integer>) node.properties.get("_axisYSpinner");
                Slider angleSlider = (Slider) node.properties.get("_angleSlider");
                @SuppressWarnings("unchecked")
                Spinner<Integer> rSpinner = (Spinner<Integer>) node.properties.get("_colorRSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> gSpinner = (Spinner<Integer>) node.properties.get("_colorGSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> bSpinner = (Spinner<Integer>) node.properties.get("_colorBSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> thicknessSpinner = (Spinner<Integer>) node.properties.get("_thicknessSpinner");
                CheckBox filledCheckBox = (CheckBox) node.properties.get("_filledCheckBox");
                node.properties.put("centerX", cxSpinner.getValue());
                node.properties.put("centerY", cySpinner.getValue());
                node.properties.put("axisX", axisXSpinner.getValue());
                node.properties.put("axisY", axisYSpinner.getValue());
                node.properties.put("angle", (int) angleSlider.getValue());
                node.properties.put("colorR", rSpinner.getValue());
                node.properties.put("colorG", gSpinner.getValue());
                node.properties.put("colorB", bSpinner.getValue());
                node.properties.put("thickness", thicknessSpinner.getValue());
                node.properties.put("filled", filledCheckBox.isSelected());
                node.properties.remove("_centerXSpinner");
                node.properties.remove("_centerYSpinner");
                node.properties.remove("_axisXSpinner");
                node.properties.remove("_axisYSpinner");
                node.properties.remove("_angleSlider");
                node.properties.remove("_colorRSpinner");
                node.properties.remove("_colorGSpinner");
                node.properties.remove("_colorBSpinner");
                node.properties.remove("_thicknessSpinner");
                node.properties.remove("_filledCheckBox");
            }

            // Handle Line/Arrow properties
            if (("Line".equals(node.nodeType) || "Arrow".equals(node.nodeType)) && node.properties.containsKey("_x1Spinner")) {
                @SuppressWarnings("unchecked")
                Spinner<Integer> x1Spinner = (Spinner<Integer>) node.properties.get("_x1Spinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> y1Spinner = (Spinner<Integer>) node.properties.get("_y1Spinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> x2Spinner = (Spinner<Integer>) node.properties.get("_x2Spinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> y2Spinner = (Spinner<Integer>) node.properties.get("_y2Spinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> rSpinner = (Spinner<Integer>) node.properties.get("_colorRSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> gSpinner = (Spinner<Integer>) node.properties.get("_colorGSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> bSpinner = (Spinner<Integer>) node.properties.get("_colorBSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> thicknessSpinner = (Spinner<Integer>) node.properties.get("_thicknessSpinner");
                node.properties.put("x1", x1Spinner.getValue());
                node.properties.put("y1", y1Spinner.getValue());
                node.properties.put("x2", x2Spinner.getValue());
                node.properties.put("y2", y2Spinner.getValue());
                node.properties.put("colorR", rSpinner.getValue());
                node.properties.put("colorG", gSpinner.getValue());
                node.properties.put("colorB", bSpinner.getValue());
                node.properties.put("thickness", thicknessSpinner.getValue());
                node.properties.remove("_x1Spinner");
                node.properties.remove("_y1Spinner");
                node.properties.remove("_x2Spinner");
                node.properties.remove("_y2Spinner");
                node.properties.remove("_colorRSpinner");
                node.properties.remove("_colorGSpinner");
                node.properties.remove("_colorBSpinner");
                node.properties.remove("_thicknessSpinner");
            }

            // Handle Text properties
            if ("Text".equals(node.nodeType) && node.properties.containsKey("_textField")) {
                TextField textField = (TextField) node.properties.get("_textField");
                @SuppressWarnings("unchecked")
                Spinner<Integer> xSpinner = (Spinner<Integer>) node.properties.get("_posXSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> ySpinner = (Spinner<Integer>) node.properties.get("_posYSpinner");
                @SuppressWarnings("unchecked")
                ComboBox<String> fontCombo = (ComboBox<String>) node.properties.get("_fontCombo");
                Slider scaleSlider = (Slider) node.properties.get("_fontScaleSlider");
                @SuppressWarnings("unchecked")
                Spinner<Integer> rSpinner = (Spinner<Integer>) node.properties.get("_colorRSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> gSpinner = (Spinner<Integer>) node.properties.get("_colorGSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> bSpinner = (Spinner<Integer>) node.properties.get("_colorBSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> thicknessSpinner = (Spinner<Integer>) node.properties.get("_thicknessSpinner");
                node.properties.put("text", textField.getText());
                node.properties.put("posX", xSpinner.getValue());
                node.properties.put("posY", ySpinner.getValue());
                String[] fonts = {"Simplex", "Plain", "Duplex", "Complex", "Triplex", "Complex Small", "Script Simplex", "Script Complex"};
                int fontIndex = java.util.Arrays.asList(fonts).indexOf(fontCombo.getValue());
                node.properties.put("fontIndex", Math.max(0, fontIndex));
                node.properties.put("fontScale", scaleSlider.getValue());
                node.properties.put("colorR", rSpinner.getValue());
                node.properties.put("colorG", gSpinner.getValue());
                node.properties.put("colorB", bSpinner.getValue());
                node.properties.put("thickness", thicknessSpinner.getValue());
                node.properties.remove("_textField");
                node.properties.remove("_posXSpinner");
                node.properties.remove("_posYSpinner");
                node.properties.remove("_fontCombo");
                node.properties.remove("_fontScaleSlider");
                node.properties.remove("_colorRSpinner");
                node.properties.remove("_colorGSpinner");
                node.properties.remove("_colorBSpinner");
                node.properties.remove("_thicknessSpinner");
            }

            // Handle Grayscale/Color Convert properties
            if ("Grayscale".equals(node.nodeType) && node.properties.containsKey("_conversionCombo")) {
                @SuppressWarnings("unchecked")
                ComboBox<String> conversionCombo = (ComboBox<String>) node.properties.get("_conversionCombo");
                String[] conversions = {"BGR to Gray", "BGR to HSV", "BGR to LAB", "BGR to RGB",
                                       "Gray to BGR", "HSV to BGR", "LAB to BGR", "RGB to BGR"};
                int conversionIndex = java.util.Arrays.asList(conversions).indexOf(conversionCombo.getValue());
                node.properties.put("conversionIndex", Math.max(0, conversionIndex));
                node.properties.remove("_conversionCombo");
            }

            // Handle dual-input "Queues in Sync" property
            if (node.hasDualInput && finalSyncCheckBox != null) {
                node.queuesInSync = finalSyncCheckBox.isSelected();
            }

            markDirty();
            paintCanvas();
        });

        dialog.showAndWaitForResult();
    }

    /**
     * Check if a node type is a dual-input node by type name.
     * This is used as a fallback for older saved files that might not have hasDualInput=true.
     */
    private boolean isDualInputNodeType(String nodeType) {
        return "AddClamp".equals(nodeType) ||
               "SubtractClamp".equals(nodeType) ||
               "AddWeighted".equals(nodeType) ||
               "BitwiseAnd".equals(nodeType) ||
               "BitwiseOr".equals(nodeType) ||
               "BitwiseXor".equals(nodeType);
    }

    /**
     * Check if a node type is an FFT filter node.
     */
    private boolean isFFTNodeType(String nodeType) {
        return "FFTLowPass".equals(nodeType) ||
               "FFTHighPass".equals(nodeType) ||
               "FFTLowPass4".equals(nodeType) ||
               "FFTHighPass4".equals(nodeType);
    }

    /**
     * Format gain value for display: "xx%" if <1.0, "x.xxX" if >=1.0
     */
    private String formatGainValue(double gain) {
        if (gain < 1.0) {
            return String.format("%.0f%%", gain * 100);
        } else {
            return String.format("%.2fx", gain);
        }
    }

    // ========================= Actions =========================

    private void newDiagram() {
        if (checkUnsavedChanges()) {
            // Stop pipeline if running
            if (pipelineRunning) {
                stopPipeline();
            }

            // Stop all webcams
            stopAllWebcams();

            // Clear nodes and connections
            nodes.clear();
            connections.clear();
            selectedNodes.clear();
            selectedConnections.clear();
            currentFilePath = null;
            isDirty = false;

            // Clear preview image
            previewImageView.setImage(null);

            updateTitle();
            paintCanvas();
        }
    }

    private void loadDiagram() {
        if (!checkUnsavedChanges()) {
            return;  // User cancelled
        }
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
        try {
            // Stop pipeline if running
            if (pipelineRunning) {
                stopPipeline();
            }

            // Stop any running webcams before loading
            stopAllWebcams();

            // Clear preview image
            previewImageView.setImage(null);

            // Load the pipeline
            FXPipelineSerializer.PipelineDocument doc = FXPipelineSerializer.load(path);

            // Clear current state
            nodes.clear();
            connections.clear();
            selectedNodes.clear();
            selectedConnections.clear();

            // Add loaded nodes and connections
            nodes.addAll(doc.nodes);
            connections.addAll(doc.connections);

            // Start webcams for WebcamSource nodes
            for (FXNode node : nodes) {
                if ("WebcamSource".equals(node.nodeType)) {
                    startWebcamForNode(node);
                }
            }

            currentFilePath = path;
            isDirty = false;
            addToRecentFiles(path);
            updateTitle();
            paintCanvas();

            System.out.println("Loaded pipeline: " + path + " (" + nodes.size() + " nodes, " + connections.size() + " connections)");
        } catch (Exception e) {
            System.err.println("Failed to load pipeline: " + e.getMessage());
            e.printStackTrace();
            showError("Load Failed", "Failed to load pipeline: " + e.getMessage());
        }
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
        try {
            FXPipelineSerializer.save(path, nodes, connections);
            currentFilePath = path;
            isDirty = false;
            addToRecentFiles(path);
            updateTitle();
            System.out.println("Saved pipeline: " + path + " (" + nodes.size() + " nodes, " + connections.size() + " connections)");
        } catch (Exception e) {
            System.err.println("Failed to save pipeline: " + e.getMessage());
            e.printStackTrace();
            showError("Save Failed", "Failed to save pipeline: " + e.getMessage());
        }
    }

    private void handleCancel() {
        // Reload from file if saved, otherwise confirm discard
        if (currentFilePath != null && !currentFilePath.isEmpty()) {
            if (isDirty) {
                Alert confirm = new Alert(Alert.AlertType.CONFIRMATION);
                confirm.setTitle("Revert Changes");
                confirm.setHeaderText(null);
                confirm.setContentText("Discard all changes and reload from disk?");
                confirm.showAndWait().ifPresent(response -> {
                    if (response == ButtonType.OK) {
                        loadDiagramFromPath(currentFilePath);
                    }
                });
            }
        } else {
            // No file to revert to - offer to clear
            if (isDirty || !nodes.isEmpty()) {
                Alert confirm = new Alert(Alert.AlertType.CONFIRMATION);
                confirm.setTitle("Discard Changes");
                confirm.setHeaderText(null);
                confirm.setContentText("Clear all nodes and start fresh?");
                confirm.showAndWait().ifPresent(response -> {
                    if (response == ButtonType.OK) {
                        nodes.clear();
                        connections.clear();
                        selectedNodes.clear();
                        selectedConnections.clear();
                        isDirty = false;
                        updateTitle();
                        paintCanvas();
                    }
                });
            }
        }
    }

    private void deleteSelected() {
        // Remove selected connections
        connections.removeAll(selectedConnections);

        // Remove connections to/from selected nodes
        for (FXNode node : selectedNodes) {
            connections.removeIf(conn -> conn.source == node || conn.target == node);
        }

        // Remove selected nodes
        nodes.removeAll(selectedNodes);

        selectedNodes.clear();
        selectedConnections.clear();
        markDirty();
        paintCanvas();
    }

    private void selectAll() {
        selectedNodes.clear();
        selectedNodes.addAll(nodes);
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
        updatePipelineStatus();

        // Clear all node and connection stats before starting
        clearPipelineStats();

        // Create and start the pipeline executor
        pipelineExecutor = new FXPipelineExecutor(nodes, connections, webcamSources);
        pipelineExecutor.setBasePath(currentFilePath);  // For resolving relative paths in nested containers
        pipelineExecutor.setOnNodeOutput((node, mat) -> {
            // Update node thumbnail
            node.thumbnail = FXImageUtils.matToImage(mat,
                NodeRenderer.PROCESSING_NODE_THUMB_WIDTH,
                NodeRenderer.PROCESSING_NODE_THUMB_HEIGHT);

            // Update preview if this node is selected
            if (selectedNodes.contains(node) && selectedNodes.size() == 1) {
                previewImageView.setImage(FXImageUtils.matToImage(mat));
            }

            // Release the mat
            mat.release();

            // Repaint canvas
            paintCanvas();
        });
        pipelineExecutor.start();
    }

    private void updatePipelineStatus() {
        // Count threads: 1 for executor + 1 for each active webcam
        int threadCount = 1 + webcamSources.size();
        statusBar.setText("Pipeline running (" + threadCount + " thread" + (threadCount != 1 ? "s" : "") + ")");
        statusBar.setTextFill(COLOR_STATUS_RUNNING);
    }

    private void stopPipeline() {
        pipelineRunning = false;
        startStopBtn.setText("Start Pipeline");
        startStopBtn.setStyle("-fx-background-color: rgb(100, 180, 100);");
        statusBar.setText("Pipeline stopped");
        statusBar.setTextFill(COLOR_STATUS_STOPPED);

        // Stop the pipeline executor
        if (pipelineExecutor != null) {
            pipelineExecutor.stop();
            pipelineExecutor = null;
        }
    }

    /**
     * Clear all pipeline statistics (node counters, connection queue stats).
     * Called when pipeline starts to reset counters from previous runs.
     */
    private void clearPipelineStats() {
        // Clear node counters
        for (FXNode node : nodes) {
            node.inputCount = 0;
            node.outputCount1 = 0;
            node.outputCount2 = 0;
            node.outputCount3 = 0;
            node.outputCount4 = 0;
        }
        // Clear connection queue stats
        for (FXConnection conn : connections) {
            conn.queueSize = 0;
            conn.totalFrames = 0;
        }
    }

    private void restartApplication() {
        // Stop pipeline first
        stopPipeline();

        try {
            // Get the java command from current JVM
            String javaBin = System.getProperty("java.home") + File.separator + "bin" + File.separator + "java";
            String classpath = System.getProperty("java.class.path");

            // Simple restart like the SWT version - just java -cp classpath MainClass
            ProcessBuilder builder = new ProcessBuilder(
                javaBin, "-cp", classpath, PipelineEditorLauncher.class.getName()
            );
            builder.inheritIO();
            builder.start();

            Platform.exit();
            System.exit(0);
        } catch (Exception e) {
            System.err.println("Failed to restart application: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Propagate an initial frame through a newly created connection.
     * This gives immediate visual feedback when connecting nodes.
     */
    private void propagateInitialFrame(FXConnection conn) {
        if (conn.source == null || conn.target == null) return;

        // If the source has a thumbnail, copy it to the target as an initial preview
        if (conn.source.thumbnail != null) {
            // For simplicity, just copy the source thumbnail to the target
            // In a real implementation, you'd process the frame through the target node
            conn.target.thumbnail = conn.source.thumbnail;
        }
    }

    private void addNodeAt(String nodeTypeName, int x, int y) {
        // Use factory to create the FXNode with correct type information
        FXNode node = FXNodeFactory.createFXNode(nodeTypeName, x, y);

        nodes.add(node);
        selectedNodes.clear();
        selectedNodes.add(node);

        // If this is a webcam source, start capturing
        if ("WebcamSource".equals(nodeTypeName)) {
            startWebcamForNode(node);
        }

        markDirty();
        paintCanvas();
    }

    /**
     * Start webcam capture for a webcam source node.
     * Runs camera detection and initialization on a background thread to avoid blocking UI.
     */
    private void startWebcamForNode(FXNode node) {
        // Run camera detection and initialization in background thread
        Thread initThread = new Thread(() -> {
            // Auto-detect highest camera if not set
            int cameraIdx = node.cameraIndex;
            if (cameraIdx < 0) {
                cameraIdx = FXWebcamSource.findHighestCamera();
                final int detectedIdx = cameraIdx;
                Platform.runLater(() -> {
                    node.cameraIndex = detectedIdx;
                    System.out.println("Auto-detected highest camera: " + detectedIdx);
                });
            }

            FXWebcamSource webcam = new FXWebcamSource(cameraIdx);
            webcam.setOnFrame(image -> {
                // Use the image directly as the thumbnail
                node.thumbnail = image;

                // Update preview if this node is selected
                if (selectedNodes.contains(node) && selectedNodes.size() == 1) {
                    previewImageView.setImage(image);
                }

                // Repaint canvas to show updated thumbnail
                paintCanvas();
            });

            if (webcam.open()) {
                // Add to map immediately (ConcurrentHashMap is thread-safe)
                webcamSources.put(node.id, webcam);
                webcam.start();
            } else {
                System.err.println("Failed to open webcam for node " + node.id);
            }
        }, "WebcamInit-" + node.id);
        initThread.setDaemon(true);
        initThread.start();
    }

    /**
     * Stop webcam capture for a node.
     */
    private void stopWebcamForNode(FXNode node) {
        FXWebcamSource webcam = webcamSources.remove(node.id);
        if (webcam != null) {
            webcam.close();
        }
    }

    /**
     * Stop all webcam captures.
     */
    private void stopAllWebcams() {
        for (FXWebcamSource webcam : webcamSources.values()) {
            webcam.close();
        }
        webcamSources.clear();
    }

    /**
     * Load an image file for a FileSource node and set it as the thumbnail.
     * Uses background loading to avoid blocking the UI thread.
     */
    private void loadFileImageForNode(FXNode node) {
        if (node.filePath == null || node.filePath.isEmpty()) {
            node.thumbnail = null;
            return;
        }

        java.io.File file = new java.io.File(node.filePath);
        if (!file.exists()) {
            node.thumbnail = null;
            return;
        }

        // Load image in background to avoid blocking UI
        javafx.scene.image.Image image = new javafx.scene.image.Image(
            file.toURI().toString(),
            true  // backgroundLoading = true
        );

        // Set up progress listener to handle when loading completes
        image.progressProperty().addListener((obs, oldVal, newVal) -> {
            if (newVal.doubleValue() >= 1.0) {
                if (image.isError()) {
                    node.thumbnail = null;
                } else {
                    node.thumbnail = image;
                    // Update preview if this node is selected
                    if (selectedNodes.contains(node) && selectedNodes.size() == 1) {
                        previewImageView.setImage(image);
                    }
                    paintCanvas();
                }
            }
        });

        // Also handle errors
        image.errorProperty().addListener((obs, oldVal, newVal) -> {
            if (newVal) {
                node.thumbnail = null;
            }
        });
    }

    /**
     * Show an error dialog.
     */
    private void showError(String title, String message) {
        Alert alert = new Alert(Alert.AlertType.ERROR);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }

    private int getNextNodeX() {
        // Place node in visible area based on current scroll position
        double scrollX = canvasScrollPane.getHvalue() * (pipelineCanvas.getWidth() - canvasScrollPane.getViewportBounds().getWidth());
        int baseX = (int)(scrollX / zoomLevel) + 50;

        // Find a position that doesn't overlap existing nodes
        int x = baseX;
        boolean overlap;
        do {
            overlap = false;
            for (FXNode node : nodes) {
                if (Math.abs(node.x - x) < 20 && Math.abs(node.y - getNextNodeYInternal(x)) < 20) {
                    x += 30;
                    overlap = true;
                    break;
                }
            }
        } while (overlap && x < baseX + 300);

        return x;
    }

    private int getNextNodeY() {
        return getNextNodeYInternal(getNextNodeX());
    }

    private int getNextNodeYInternal(int targetX) {
        // Place node in visible area based on current scroll position
        double scrollY = canvasScrollPane.getVvalue() * (pipelineCanvas.getHeight() - canvasScrollPane.getViewportBounds().getHeight());
        int baseY = (int)(scrollY / zoomLevel) + 50;

        // Find a Y position that doesn't overlap at this X
        int y = baseY;
        boolean overlap;
        do {
            overlap = false;
            for (FXNode node : nodes) {
                if (Math.abs(node.x - targetX) < node.width + 20 &&
                    y >= node.y - 20 && y <= node.y + node.height + 20) {
                    y = (int)(node.y + node.height + 30);
                    overlap = true;
                }
            }
        } while (overlap && y < baseY + 500);

        return y;
    }

    private void filterToolbarButtons() {
        String filter = searchBox.getText().toLowerCase().trim();
        toolbarContent.getChildren().clear();

        for (String category : FXNodeRegistry.getCategoriesExcluding("Container I/O")) {
            List<FXNodeRegistry.NodeType> matchingNodes = new ArrayList<>();
            for (FXNodeRegistry.NodeType nodeType : FXNodeRegistry.getNodesInCategory(category)) {
                if (filter.isEmpty() ||
                    nodeType.displayName.toLowerCase().contains(filter) ||
                    nodeType.name.toLowerCase().contains(filter) ||
                    category.toLowerCase().contains(filter)) {
                    matchingNodes.add(nodeType);
                }
            }

            if (!matchingNodes.isEmpty()) {
                addToolbarCategory(category);
                for (FXNodeRegistry.NodeType nodeType : matchingNodes) {
                    final String typeName = nodeType.name;
                    addToolbarButton(nodeType.getButtonName(), () -> addNodeAt(typeName, getNextNodeX(), getNextNodeY()));
                }
            }
        }
    }

    private void updateNodeCount() {
        nodeCountLabel.setText("Nodes: " + nodes.size() + " | Connections: " + connections.size());
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
        // Set macOS application name before JavaFX initializes
        System.setProperty("apple.awt.application.name", "OpenCV Pipeline Editor");
        // This is for the dock and menu bar on macOS
        System.setProperty("com.apple.mrj.application.apple.menu.about.name", "OpenCV Pipeline Editor");
        launch(args);
    }
}
