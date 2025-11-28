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
                node.isContainer, node.inputCount, outputCounters, node.nodeType);
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
        FXPropertiesDialog dialog = new FXPropertiesDialog(
            primaryStage,
            node.label + " Properties",
            node.nodeType,
            node.label
        );

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
        }

        // Set OK handler to save values
        final Spinner<Integer> finalCameraSpinner = cameraSpinner;
        final TextField finalFileField = filePathField;
        final TextField finalPipelineField = pipelineFileField;
        final ComboBox<String> finalFpsCombo = fpsCombo;
        dialog.setOnOk(() -> {
            node.label = dialog.getNameValue();

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

            markDirty();
            paintCanvas();
        });

        dialog.showAndWaitForResult();
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

        // Create and start the pipeline executor
        pipelineExecutor = new FXPipelineExecutor(nodes, connections, webcamSources);
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
