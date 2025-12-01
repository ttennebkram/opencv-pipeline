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
import com.ttennebkram.pipeline.util.MatTracker;
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
    private FXConnection yankingConnection = null;  // Connection being yanked (detached from one end)
    private boolean yankingFromTarget = false;       // True if yanking from target end, false if from source end

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

    // Command line options
    private boolean autoStartPipeline = false;
    private String commandLinePipelineFile = null;

    @Override
    public void start(Stage primaryStage) {
        // Parse command line arguments
        java.util.List<String> params = getParameters().getRaw();
        for (String param : params) {
            if ("--start".equals(param)) {
                autoStartPipeline = true;
            } else if (!param.startsWith("--")) {
                // Non-flag argument is the pipeline file path
                commandLinePipelineFile = param;
            }
        }
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

        // Add keyboard handler for delete key, arrow keys, and other shortcuts
        scene.setOnKeyPressed(e -> {
            // Ctrl+M: Toggle Mat tracking
            if (e.getCode() == KeyCode.M && e.isControlDown() && !e.isShiftDown()) {
                MatTracker.setEnabled(!MatTracker.isEnabled());
                e.consume();
                return;
            }
            // Ctrl+Shift+M: Dump Mat leaks
            if (e.getCode() == KeyCode.M && e.isControlDown() && e.isShiftDown()) {
                MatTracker.printSummary();
                MatTracker.dumpLeaksByLocation();
                e.consume();
                return;
            }
            if (e.getCode() == KeyCode.DELETE || e.getCode() == KeyCode.BACK_SPACE) {
                deleteSelected();
                e.consume();
            } else if (!selectedNodes.isEmpty()) {
                // Arrow keys move selected nodes by 1 pixel at 100% zoom
                double delta = 1.0 / zoomLevel;  // 1 pixel in canvas coordinates
                boolean moved = false;
                switch (e.getCode()) {
                    case UP:
                        for (FXNode node : selectedNodes) {
                            node.y -= delta;
                        }
                        moved = true;
                        break;
                    case DOWN:
                        for (FXNode node : selectedNodes) {
                            node.y += delta;
                        }
                        moved = true;
                        break;
                    case LEFT:
                        for (FXNode node : selectedNodes) {
                            node.x -= delta;
                        }
                        moved = true;
                        break;
                    case RIGHT:
                        for (FXNode node : selectedNodes) {
                            node.x += delta;
                        }
                        moved = true;
                        break;
                    default:
                        break;
                }
                if (moved) {
                    markDirty();
                    paintCanvas();
                    e.consume();  // Prevent ScrollPane from scrolling
                }
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
        if (autoStartPipeline) {
            System.out.println("Auto-start pipeline requested via --start command line or exec:exec@start in pom.xml target");
        }
    }

    private void loadLastFile() {
        // Command line file takes precedence over last file
        String fileToLoad = null;
        if (commandLinePipelineFile != null) {
            File cmdFile = new File(commandLinePipelineFile);
            if (cmdFile.exists()) {
                fileToLoad = cmdFile.getAbsolutePath();
            } else {
                System.err.println("Pipeline file not found: " + commandLinePipelineFile);
            }
        }

        // Fall back to last file if no command line file
        if (fileToLoad == null) {
            String lastFile = prefs.get(LAST_FILE_KEY, null);
            if (lastFile != null && new File(lastFile).exists()) {
                fileToLoad = lastFile;
            }
        }

        if (fileToLoad != null) {
            loadDiagramFromPath(fileToLoad);
            // Auto-start pipeline if --start was passed
            if (autoStartPipeline && !pipelineRunning) {
                startPipeline();
            }
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

        // Populate node buttons from registry (all categories including Container I/O)
        for (String category : FXNodeRegistry.getCategories()) {
            addToolbarCategory(category);
            for (FXNodeRegistry.NodeType nodeType : FXNodeRegistry.getNodesInCategory(category)) {
                final String typeName = nodeType.name;
                addToolbarButton(nodeType.getButtonName(), () -> addNodeAt(typeName, getNextNodeX(), getNextNodeY()));
            }
            // Add Connector/Queue button to Utility category
            if (category.equals("Utility")) {
                addToolbarButton("Connector/Queue", () -> addConnectorQueue(getNextNodeX(), getNextNodeY()));
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
                    // Show cached preview image if available
                    if (clickedNode.previewImage != null) {
                        previewImageView.setImage(clickedNode.previewImage);
                    }
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
        startStopBtn.setStyle("-fx-base: #90EE90;");  // Light green when not running
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

        // Placeholder for when no image - use StackPane that expands to fill available space
        StackPane imageContainer = new StackPane(previewImageView);
        imageContainer.setStyle("-fx-background-color: #cccccc; -fx-min-height: 200;");
        VBox.setVgrow(imageContainer, Priority.ALWAYS);

        // Bind image view size to container, with padding
        previewImageView.fitWidthProperty().bind(imageContainer.widthProperty().subtract(10));
        previewImageView.fitHeightProperty().bind(imageContainer.heightProperty().subtract(10));

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
                    conn.queueSize, conn.totalFrames,
                    conn.source != null, conn.target != null);
            }
        }

        // Draw connection being drawn (only for NEW connections, not when yanking existing ones)
        // Yanked connections are already in the connections list and render via their free endpoints
        if (isDrawingConnection && connectionSource != null && yankingConnection == null) {
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
                node.isContainer, node.inputCount, node.inputCount2, outputCounters, node.nodeType,
                node.isBoundaryNode);

            // Draw stats line (Pri/Work/FPS) below title - always show (including after load)
            // Source nodes (no input) show FPS; IsNested nodes show Is-Nested status; processing nodes show just Pri/Work
            if (!node.hasInput) {
                NodeRenderer.drawSourceStatsLine(gc, node.x + 22, node.y + node.height - 8,
                    node.threadPriority, node.workUnitsCompleted, node.effectiveFps);
            } else if ("IsNestedInput".equals(node.nodeType) || "IsNotNestedOutput".equals(node.nodeType)) {
                NodeRenderer.drawIsNestedStatsLine(gc, node.x + 22, node.y + node.height - 8,
                    node.threadPriority, node.workUnitsCompleted, node.isEmbedded);
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
                FXHelpBrowser.openForNodeType(primaryStage, node.nodeType);
                return;
            }
        }

        // Check for clicks on free endpoints of dangling connections
        double tolerance = NodeRenderer.CONNECTION_RADIUS + 9;
        for (FXConnection conn : connections) {
            // Check free source endpoint (source-dangling connection)
            if (conn.source == null) {
                double dist = Math.sqrt(Math.pow(canvasX - conn.freeSourceX, 2) + Math.pow(canvasY - conn.freeSourceY, 2));
                if (dist < tolerance) {
                    // Grab the free source end
                    yankingConnection = conn;
                    yankingFromTarget = false;
                    isDrawingConnection = true;
                    paintCanvas();
                    return;
                }
            }
            // Check free target endpoint (target-dangling connection)
            if (conn.target == null) {
                double dist = Math.sqrt(Math.pow(canvasX - conn.freeTargetX, 2) + Math.pow(canvasY - conn.freeTargetY, 2));
                if (dist < tolerance) {
                    // Grab the free target end
                    yankingConnection = conn;
                    yankingFromTarget = true;
                    isDrawingConnection = true;
                    paintCanvas();
                    return;
                }
            }
        }

        // Check if clicking on a node's output point (to start or yank connection)
        for (FXNode node : nodes) {
            int outputIdx = node.getOutputPointAt(canvasX, canvasY, NodeRenderer.CONNECTION_RADIUS + 3);
            if (outputIdx >= 0) {
                // Check if there's an existing connection from this output - yank its SOURCE end
                FXConnection existingConn = findConnectionFromOutput(node, outputIdx);
                if (existingConn != null) {
                    // Yank the SOURCE end (the end at the output point we clicked on)
                    double[] sourcePt = node.getOutputPoint(outputIdx);
                    if (sourcePt != null) {
                        existingConn.freeSourceX = sourcePt[0];
                        existingConn.freeSourceY = sourcePt[1];
                    } else {
                        existingConn.freeSourceX = canvasX;
                        existingConn.freeSourceY = canvasY;
                    }
                    existingConn.source = null;  // Detach source end
                    yankingConnection = existingConn;
                    yankingFromTarget = false;  // Yanking source end
                    isDrawingConnection = true;  // Enables drag tracking
                    markDirty();
                    paintCanvas();
                    return;
                }
                // No existing connection - start drawing a new connection
                connectionSource = node;
                connectionOutputIndex = outputIdx;
                connectionEndX = canvasX;
                connectionEndY = canvasY;
                isDrawingConnection = true;
                yankingConnection = null;
                paintCanvas();
                return;
            }
        }

        // Check if clicking on a node's input point (to yank or start reverse connection)
        for (FXNode node : nodes) {
            int inputIdx = node.getInputPointAt(canvasX, canvasY, NodeRenderer.CONNECTION_RADIUS + 3);
            if (inputIdx >= 0) {
                // Check if there's an existing connection to this input - yank its TARGET end
                FXConnection existingConn = findConnectionToInput(node, inputIdx);
                if (existingConn != null) {
                    // Yank the TARGET end (the end at the input point we clicked on)
                    double[] targetPt = node.getInputPoint(inputIdx);
                    if (targetPt != null) {
                        existingConn.freeTargetX = targetPt[0];
                        existingConn.freeTargetY = targetPt[1];
                    } else {
                        existingConn.freeTargetX = canvasX;
                        existingConn.freeTargetY = canvasY;
                    }
                    existingConn.target = null;  // Detach target end
                    yankingConnection = existingConn;
                    yankingFromTarget = true;  // Yanking target end
                    isDrawingConnection = true;  // Enables drag tracking
                    markDirty();
                    paintCanvas();
                    return;
                }
                // No existing connection - just consume the click on the input point
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

            // Show cached preview image if available (from previous run or loaded from disk)
            // This gives immediate feedback before pipeline runs
            if (selectedNodes.size() == 1 && clickedNode.previewImage != null) {
                previewImageView.setImage(clickedNode.previewImage);
            }

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
            if (yankingConnection != null) {
                // Dragging an existing connection - update its free endpoint
                if (yankingFromTarget) {
                    yankingConnection.freeTargetX = canvasX;
                    yankingConnection.freeTargetY = canvasY;
                } else {
                    yankingConnection.freeSourceX = canvasX;
                    yankingConnection.freeSourceY = canvasY;
                }
            } else {
                // Drawing a new connection
                connectionEndX = canvasX;
                connectionEndY = canvasY;
            }
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
            boolean connected = false;

            if (yankingConnection != null) {
                // We're reconnecting a yanked connection (preserving queue data)
                if (yankingFromTarget) {
                    // Yanking from target end - try to reconnect target to a new input
                    for (FXNode node : nodes) {
                        if (node != yankingConnection.source) {
                            int inputIdx = node.getInputPointAt(canvasX, canvasY, NodeRenderer.CONNECTION_RADIUS + 5);
                            if (inputIdx >= 0) {
                                // Check if there's already a connection to this input - reject drop if occupied
                                FXConnection existingConn = findConnectionToInput(node, inputIdx);
                                if (existingConn != null && existingConn != yankingConnection) {
                                    // Input occupied - leave yanked connection dangling
                                    break;
                                }
                                // Reconnect the yanked connection
                                yankingConnection.reconnectTarget(node, inputIdx);
                                propagateInitialFrame(yankingConnection);
                                connected = true;
                                markDirty();
                                break;
                            }
                        }
                    }
                    if (!connected) {
                        // Leave connection with dangling target - push away from nearby connection points
                        double[] pushed = pushAwayFromConnectionPoints(canvasX, canvasY, 30);
                        yankingConnection.freeTargetX = pushed[0];
                        yankingConnection.freeTargetY = pushed[1];
                        markDirty();
                    }
                } else {
                    // Yanking from source end - try to reconnect source to a new output
                    for (FXNode node : nodes) {
                        if (node != yankingConnection.target) {
                            int outputIdx = node.getOutputPointAt(canvasX, canvasY, NodeRenderer.CONNECTION_RADIUS + 5);
                            if (outputIdx >= 0) {
                                // Check if there's already a connection from this output - reject drop if occupied
                                FXConnection existingConn = findConnectionFromOutput(node, outputIdx);
                                if (existingConn != null && existingConn != yankingConnection) {
                                    // Output occupied - leave yanked connection dangling
                                    break;
                                }
                                // Reconnect the yanked connection
                                yankingConnection.reconnectSource(node, outputIdx);
                                connected = true;
                                markDirty();
                                break;
                            }
                        }
                    }
                    if (!connected) {
                        // Leave connection with dangling source - push away from nearby connection points
                        double[] pushed = pushAwayFromConnectionPoints(canvasX, canvasY, 30);
                        yankingConnection.freeSourceX = pushed[0];
                        yankingConnection.freeSourceY = pushed[1];
                        markDirty();
                    }
                }
                yankingConnection = null;
            } else {
                // Creating a new connection (not yanking)
                for (FXNode node : nodes) {
                    if (node != connectionSource) {
                        int inputIdx = node.getInputPointAt(canvasX, canvasY, NodeRenderer.CONNECTION_RADIUS + 5);
                        if (inputIdx >= 0) {
                            // Check if there's already a connection to this input - reject the drop
                            FXConnection existingConn = findConnectionToInput(node, inputIdx);
                            if (existingConn != null) {
                                // Input already occupied - create dangling connection instead
                                break;
                            }

                            // Create connection
                            FXConnection conn = new FXConnection(connectionSource, connectionOutputIndex, node, inputIdx);
                            connections.add(conn);

                            // Propagate initial frame to the newly connected node
                            propagateInitialFrame(conn);

                            connected = true;
                            markDirty();
                            break;
                        }
                    }
                }
                // New connections that don't connect anywhere are kept as dangling
                if (!connected && connectionSource != null) {
                    // Push away from nearby connection points
                    double[] pushed = pushAwayFromConnectionPoints(canvasX, canvasY, 30);
                    FXConnection danglingConn = FXConnection.createFromSource(connectionSource, connectionOutputIndex, pushed[0], pushed[1]);
                    connections.add(danglingConn);
                    markDirty();
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
        editorWindow.setOnRefreshPipeline(() -> {
            if (pipelineExecutor != null && pipelineExecutor.isRunning()) {
                pipelineExecutor.triggerRefresh();
            }
        });
        editorWindow.setIsPipelineRunning(() -> pipelineRunning);
        editorWindow.setGetThreadCount(() -> pipelineExecutor != null ? pipelineExecutor.getThreadCount() + 1 : 1); // +1 for JavaFX thread
        editorWindow.setOnRequestGlobalSave(this::saveDiagram);
        editorWindow.setOnQuit(Platform::exit);
        editorWindow.setOnRestart(this::restartApplication);

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

        // Check for connection point hover on ALL nodes (connection points are at edges)
        FXNode connectionPointNode = null;
        boolean isInputPoint = false;
        int connectionPointIndex = -1;
        double tolerance = 10.0;  // Hit tolerance for connection points

        for (FXNode node : nodes) {
            // Check input points
            int inputIdx = node.getInputPointAt(canvasX, canvasY, tolerance);
            if (inputIdx >= 0) {
                connectionPointNode = node;
                isInputPoint = true;
                connectionPointIndex = inputIdx;
                break;
            }
            // Check output points
            int outputIdx = node.getOutputPointAt(canvasX, canvasY, tolerance);
            if (outputIdx >= 0) {
                connectionPointNode = node;
                isInputPoint = false;
                connectionPointIndex = outputIdx;
                break;
            }
        }

        // Update cursor - show hand for checkbox, help icon, and connection points
        if (onCheckbox || onHelpIcon || connectionPointNode != null) {
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
        } else if (connectionPointNode != null) {
            // Show connection point tooltip
            tooltipText = getConnectionPointTooltip(connectionPointNode, isInputPoint, connectionPointIndex);
            // Add occupancy info for input points
            if (isInputPoint) {
                FXConnection existingConn = findConnectionToInput(connectionPointNode, connectionPointIndex);
                if (existingConn != null) {
                    tooltipText += "\n(Connected - yank existing wire to replace)";
                }
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

    /**
     * Find a connection from a specific output point.
     * Returns the first connection found from that node's output index,
     * or a source-dangling connection whose free end is near the output point.
     */
    private FXConnection findConnectionFromOutput(FXNode node, int outputIndex) {
        double[] outputPt = node.getOutputPoint(outputIndex);
        for (FXConnection conn : connections) {
            // Check if source is connected to this output
            if (conn.source == node && conn.sourceOutputIndex == outputIndex) {
                return conn;
            }
            // Also check if this is a source-dangling connection whose free end is at this output point
            if (conn.source == null && outputPt != null) {
                double dist = Math.sqrt(Math.pow(conn.freeSourceX - outputPt[0], 2) + Math.pow(conn.freeSourceY - outputPt[1], 2));
                if (dist < 15) {  // Within 15 pixels of the output point
                    return conn;
                }
            }
        }
        return null;
    }

    /**
     * Find a connection to a specific input point.
     * Returns the first connection found to that node's input index.
     */
    private FXConnection findConnectionToInput(FXNode node, int inputIndex) {
        double[] inputPt = node.getInputPoint(inputIndex);
        for (FXConnection conn : connections) {
            // Check if target is connected to this input
            if (conn.target == node && conn.targetInputIndex == inputIndex) {
                return conn;
            }
            // Also check if this is a target-dangling connection whose free end is at this input point
            if (conn.target == null && inputPt != null) {
                double dist = Math.sqrt(Math.pow(conn.freeTargetX - inputPt[0], 2) + Math.pow(conn.freeTargetY - inputPt[1], 2));
                if (dist < 15) {  // Within 15 pixels of the input point
                    return conn;
                }
            }
        }
        return null;
    }

    /**
     * Push a dangling endpoint away from nearby connection points so it's visually
     * clear that the connection is not connected.
     * @param x The current X position
     * @param y The current Y position
     * @param minDistance The minimum distance to maintain from connection points
     * @return A 2-element array [newX, newY] with the pushed-away position
     */
    private double[] pushAwayFromConnectionPoints(double x, double y, double minDistance) {
        double newX = x;
        double newY = y;

        for (FXNode node : nodes) {
            // Check against input points
            int inputCount = node.hasInput ? (node.hasDualInput ? 2 : 1) : 0;
            for (int i = 0; i < inputCount; i++) {
                double[] inputPt = node.getInputPoint(i);
                if (inputPt != null) {
                    double dx = x - inputPt[0];
                    double dy = y - inputPt[1];
                    double dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < minDistance && dist > 0) {
                        // Push away
                        double scale = minDistance / dist;
                        newX = inputPt[0] + dx * scale;
                        newY = inputPt[1] + dy * scale;
                    } else if (dist == 0) {
                        // Directly on the point - push in a default direction
                        newX = x + minDistance;
                    }
                }
            }

            // Check against output points
            for (int i = 0; i < node.outputCount; i++) {
                double[] outputPt = node.getOutputPoint(i);
                if (outputPt != null) {
                    double dx = x - outputPt[0];
                    double dy = y - outputPt[1];
                    double dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < minDistance && dist > 0) {
                        // Push away
                        double scale = minDistance / dist;
                        newX = outputPt[0] + dx * scale;
                        newY = outputPt[1] + dy * scale;
                    } else if (dist == 0) {
                        // Directly on the point - push in a default direction
                        newX = x + minDistance;
                    }
                }
            }
        }

        return new double[] { newX, newY };
    }

    /**
     * Get tooltip text for a connection point on a node.
     * @param node The node
     * @param isInput true for input point, false for output point
     * @param index The input/output index (0 for primary, 1 for secondary input on dual-input nodes)
     * @return Tooltip text describing the connection point
     */
    private String getConnectionPointTooltip(FXNode node, boolean isInput, int index) {
        String nodeType = node.nodeType;

        if (isInput) {
            // Input tooltips
            if (node.hasDualInput) {
                if (index == 0) {
                    // First input for dual-input nodes
                    switch (nodeType) {
                        case "AddClamp": return "Input 1: First image (Mat)";
                        case "SubtractClamp": return "Input 1: Base image (Mat)";
                        case "AddWeighted": return "Input 1: First image (alpha weighted)";
                        case "BitwiseAnd": return "Input 1: First image (Mat)";
                        case "BitwiseOr": return "Input 1: First image (Mat)";
                        case "BitwiseXor": return "Input 1: First image (Mat)";
                        case "MatchTemplate": return "Input 1: Source image to search in";
                        default: return "Input 1 (Mat)";
                    }
                } else {
                    // Second input for dual-input nodes
                    switch (nodeType) {
                        case "AddClamp": return "Input 2: Second image (Mat)";
                        case "SubtractClamp": return "Input 2: Image to subtract (Mat)";
                        case "AddWeighted": return "Input 2: Second image (beta weighted)";
                        case "BitwiseAnd": return "Input 2: Second image (Mat)";
                        case "BitwiseOr": return "Input 2: Second image (Mat)";
                        case "BitwiseXor": return "Input 2: Second image (Mat)";
                        case "MatchTemplate": return "Input 2: Template image to find";
                        default: return "Input 2 (Mat)";
                    }
                }
            } else {
                // Single input nodes
                return "Input (Mat)";
            }
        } else {
            // Output tooltips
            if (node.outputCount == 4) {
                // Multi-output nodes (FFT4)
                switch (nodeType) {
                    case "FFTHighPass4":
                        switch (index) {
                            case 0: return "Output 1: Filtered image (high frequencies)";
                            case 1: return "Output 2: Difference (blocked low frequencies)";
                            case 2: return "Output 3: FFT spectrum visualization";
                            case 3: return "Output 4: Filter curve visualization";
                            default: return "Output " + (index + 1) + " (Mat)";
                        }
                    case "FFTLowPass4":
                        switch (index) {
                            case 0: return "Output 1: Filtered image (low frequencies)";
                            case 1: return "Output 2: Difference (blocked high frequencies)";
                            case 2: return "Output 3: FFT spectrum visualization";
                            case 3: return "Output 4: Filter curve visualization";
                            default: return "Output " + (index + 1) + " (Mat)";
                        }
                    default:
                        return "Output " + (index + 1) + " (Mat)";
                }
            } else if (node.outputCount == 2) {
                // Clone node has 2 outputs
                if ("Clone".equals(nodeType)) {
                    return "Output " + (index + 1) + ": Clone of input (Mat)";
                }
                return "Output " + (index + 1) + " (Mat)";
            } else {
                // Single output nodes - provide descriptive tooltips for key node types
                switch (nodeType) {
                    // Sources
                    case "WebcamSource": return "Output: Live webcam frame (Mat)";
                    case "FileSource": return "Output: Image/video frame (Mat)";
                    case "BlankSource": return "Output: Solid color image (Mat)";

                    // Basic processing
                    case "Grayscale": return "Output: Color-converted image (Mat)";
                    case "Invert": return "Output: Inverted image (Mat)";
                    case "Threshold": return "Output: Binary threshold image (Mat)";
                    case "AdaptiveThreshold": return "Output: Adaptive threshold image (Mat)";
                    case "Gain": return "Output: Brightness-adjusted image (Mat)";
                    case "CLAHE": return "Output: Contrast-enhanced image (Mat)";
                    case "BitPlanesGrayscale":
                    case "BitPlanesColor": return "Output: Bit plane visualization (Mat)";

                    // Blur
                    case "GaussianBlur": return "Output: Gaussian blurred image (Mat)";
                    case "MedianBlur": return "Output: Median blurred image (Mat)";
                    case "BilateralFilter": return "Output: Edge-preserving blurred image (Mat)";
                    case "BoxBlur": return "Output: Box blurred image (Mat)";
                    case "MeanShift": return "Output: Mean-shift filtered image (Mat)";

                    // Edge detection
                    case "CannyEdge": return "Output: Edge map (binary Mat)";
                    case "Sobel": return "Output: Sobel derivatives (Mat)";
                    case "Laplacian": return "Output: Laplacian edges (Mat)";
                    case "Scharr": return "Output: Scharr derivatives (Mat)";

                    // Filter
                    case "ColorInRange": return "Output: Color range mask (binary Mat)";
                    case "BitwiseNot": return "Output: Bitwise NOT image (Mat)";
                    case "Filter2D": return "Output: Convolved image (Mat)";
                    case "FFTLowPass": return "Output: Low-pass filtered image (Mat)";
                    case "FFTHighPass": return "Output: High-pass filtered image (Mat)";

                    // Morphology
                    case "Erode": return "Output: Eroded image (Mat)";
                    case "Dilate": return "Output: Dilated image (Mat)";
                    case "MorphOpen": return "Output: Morphologically opened image (Mat)";
                    case "MorphClose": return "Output: Morphologically closed image (Mat)";
                    case "MorphologyEx": return "Output: Morphology result (Mat)";

                    // Transform
                    case "WarpAffine": return "Output: Affine transformed image (Mat)";
                    case "Crop": return "Output: Cropped region (Mat)";

                    // Detection
                    case "BlobDetector": return "Output: Image with detected blobs (Mat)";
                    case "ConnectedComponents": return "Output: Labeled components image (Mat)";
                    case "HoughCircles": return "Output: Image with detected circles (Mat)";
                    case "HoughLines": return "Output: Image with detected lines (Mat)";
                    case "HarrisCorners": return "Output: Corner response image (Mat)";
                    case "ShiTomasi": return "Output: Image with corners marked (Mat)";
                    case "Contours": return "Output: Image with contours drawn (Mat)";
                    case "SIFTFeatures": return "Output: Image with SIFT features (Mat)";
                    case "ORBFeatures": return "Output: Image with ORB features (Mat)";
                    case "MatchTemplate": return "Output: Image with match location (Mat)";

                    // Visualization
                    case "Histogram": return "Output: Histogram visualization (Mat)";

                    // Utility
                    case "Monitor": return "Output: Passthrough image (Mat)";
                    case "Container": return "Output: Sub-pipeline result (Mat)";
                    case "ContainerInput": return "Output: Container input data (Mat)";
                    case "ContainerOutput": return "Input: Data to send to parent (Mat)";

                    // Content/Drawing
                    case "Rectangle":
                    case "Circle":
                    case "Ellipse":
                    case "Line":
                    case "Arrow":
                    case "Text": return "Output: Image with drawing (Mat)";

                    default: return "Output (Mat)";
                }
            }
        }
    }

    private void showNodeProperties(FXNode node) {
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

        // Sync node fields to properties for FileSource before dialog opens
        if ("FileSource".equals(node.nodeType)) {
            node.properties.put("imagePath", node.filePath != null ? node.filePath : "");
        }

        // Use modular processor from registry if available
        if (com.ttennebkram.pipeline.fx.processors.FXProcessorRegistry.hasProcessor(node.nodeType)) {
            com.ttennebkram.pipeline.fx.processors.FXProcessor processor =
                com.ttennebkram.pipeline.fx.processors.FXProcessorRegistry.createProcessor(node);
            if (processor != null && processor.hasProperties()) {
                processor.buildPropertiesDialog(dialog);
                Runnable originalOnOk = dialog.getOnOk();
                dialog.setOnOk(() -> {
                    if (originalOnOk != null) originalOnOk.run();
                    processor.syncToFXNode(node);
                });
            }
        }

        // Add node-type-specific properties
        Spinner<Integer> cameraSpinner = null;
        TextField pipelineFileField = null;

        if ("WebcamSource".equals(node.nodeType)) {
            cameraSpinner = dialog.addSpinner("Camera Index:", 0, 5, node.cameraIndex);
            dialog.addDescription("Camera 0 is often a virtual camera (e.g., iPhone).\nTry camera 1 for your built-in webcam.");
        // FileSource properties are handled by FXNodePropertiesHelper

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
        // IMPORTANT: Get any existing onOk callback set by FXNodePropertiesHelper (for modular processors)
        // and chain it with our own handler
        final Spinner<Integer> finalCameraSpinner = cameraSpinner;
        final TextField finalPipelineField = pipelineFileField;
        final CheckBox finalSyncCheckBox = syncCheckBox;
        final Runnable existingOnOk = dialog.getOnOk();
        dialog.setOnOk(() -> {
            // First run any callback set by FXNodePropertiesHelper (for modular processors)
            if (existingOnOk != null) {
                existingOnOk.run();
            }

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

            // Handle file source properties (after FXNodePropertiesHelper saved to node.properties)
            if ("FileSource".equals(node.nodeType)) {
                Object imagePathObj = node.properties.get("imagePath");
                String newPath = imagePathObj != null ? imagePathObj.toString().trim() : "";
                if (!newPath.equals(node.filePath)) {
                    node.filePath = newPath;
                    loadFileImageForNode(node);
                    // Invalidate executor cache so new image is picked up during pipeline run
                    if (pipelineExecutor != null && pipelineExecutor.isRunning()) {
                        pipelineExecutor.invalidateFileSourceCache(node.id);
                    }
                }
                // Handle FPS mode (fpsMode is an index: 0=JustOnce, 1=Auto, 2=1fps, 3=5fps, etc.)
                Object fpsModeObj = node.properties.get("fpsMode");
                int fpsMode = fpsModeObj instanceof Number ? ((Number) fpsModeObj).intValue() : 1;
                double[] fpsValues = {0, -1.0, 1, 5, 10, 15, 24, 30, 60};
                node.fps = fpsMode >= 0 && fpsMode < fpsValues.length ? fpsValues[fpsMode] : -1.0;
            }

            // Handle container properties
            if ("Container".equals(node.nodeType) && finalPipelineField != null) {
                String newPath = finalPipelineField.getText().trim();
                String oldPath = node.pipelineFilePath;
                node.pipelineFilePath = newPath;

                // If path changed and new path is not empty, load the pipeline file into inner nodes
                if (!newPath.isEmpty() && !newPath.equals(oldPath)) {
                    try {
                        FXPipelineSerializer.PipelineDocument doc = FXPipelineSerializer.load(newPath);
                        node.innerNodes.clear();
                        node.innerNodes.addAll(doc.nodes);
                        node.innerConnections.clear();
                        node.innerConnections.addAll(doc.connections);
                        // Reassign IDs to avoid collisions with outer nodes
                        FXPipelineSerializer.reassignInnerNodeIds(node.innerNodes);
                        System.out.println("Loaded pipeline into container: " + newPath +
                                           " (" + doc.nodes.size() + " nodes)");
                    } catch (Exception e) {
                        System.err.println("Failed to load pipeline file: " + newPath + " - " + e.getMessage());
                        // Show error dialog
                        javafx.scene.control.Alert alert = new javafx.scene.control.Alert(
                            javafx.scene.control.Alert.AlertType.ERROR);
                        alert.setTitle("Load Error");
                        alert.setHeaderText("Failed to load pipeline file");
                        alert.setContentText(e.getMessage());
                        alert.showAndWait();
                    }
                }
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

            // Trigger pipeline refresh so parameter changes show immediately
            // (especially important for static image sources with low/no FPS)
            if (pipelineExecutor != null && pipelineExecutor.isRunning()) {
                pipelineExecutor.triggerRefresh();
            }
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
            checkAndReportTooManyFiles(e);
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
        // Show saving indicator in status bar
        statusBar.setText("Saving...");
        statusBar.setTextFill(javafx.scene.paint.Color.ORANGE);

        // Create snapshot of data needed for save (avoid concurrent modification)
        final List<FXNode> nodesCopy = new ArrayList<>(nodes);
        final List<FXConnection> connectionsCopy = new ArrayList<>(connections);
        final int nodeCount = nodes.size();
        final int connCount = connections.size();

        // Run save on background thread
        Thread saveThread = new Thread(() -> {
            try {
                FXPipelineSerializer.save(path, nodesCopy, connectionsCopy);

                // Update UI on JavaFX thread
                Platform.runLater(() -> {
                    currentFilePath = path;
                    isDirty = false;
                    addToRecentFiles(path);
                    updateTitle();
                    statusBar.setText("Saved: " + new java.io.File(path).getName());
                    statusBar.setTextFill(COLOR_STATUS_STOPPED);
                    System.out.println("Saved pipeline: " + path + " (" + nodeCount + " nodes, " + connCount + " connections)");
                });
            } catch (Exception e) {
                System.err.println("Failed to save pipeline: " + e.getMessage());
                e.printStackTrace();
                checkAndReportTooManyFiles(e);

                // Show error on JavaFX thread
                Platform.runLater(() -> {
                    statusBar.setText("Save failed");
                    statusBar.setTextFill(javafx.scene.paint.Color.RED);
                    showError("Save Failed", "Failed to save pipeline: " + e.getMessage());
                });
            }
        }, "Pipeline-Save-Thread");
        saveThread.setDaemon(true);
        saveThread.start();
    }

    /**
     * Check if an exception is a "too many open files" error and report Mat tracking info.
     */
    private void checkAndReportTooManyFiles(Throwable t) {
        while (t != null) {
            String msg = t.getMessage();
            if (msg != null && (msg.contains("Too many open files") ||
                               msg.contains("too many open files") ||
                               msg.contains("EMFILE") ||
                               msg.contains("ENFILE"))) {
                System.err.println("\n!!! TOO MANY OPEN FILES ERROR DETECTED !!!");
                System.err.println("Dumping Mat tracking information...\n");
                com.ttennebkram.pipeline.util.MatTracker.printSummary(System.err);
                com.ttennebkram.pipeline.util.MatTracker.dumpLeaksByLocation(System.err);
                return;
            }
            t = t.getCause();
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
        // Remove only explicitly selected connections (user chose to delete them)
        connections.removeAll(selectedConnections);

        // Detach (not remove) connections to/from selected nodes
        // This preserves connections and any queued data they contain
        for (FXNode node : selectedNodes) {
            for (FXConnection conn : connections) {
                if (conn.source == node) {
                    conn.detachSource();
                }
                if (conn.target == node) {
                    conn.detachTarget();
                }
            }
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
        startStopBtn.setStyle("-fx-base: #F08080;");  // Light coral when running (bright enough for 3D effect)

        // Clear all node and connection stats before starting
        clearPipelineStats();

        // Create and start the pipeline executor
        pipelineExecutor = new FXPipelineExecutor(nodes, connections, webcamSources);
        pipelineExecutor.setBasePath(currentFilePath);  // For resolving relative paths in nested containers
        pipelineExecutor.setOnNodeOutput((node, mat) -> {
            try {
                // Update node thumbnail (small, for node box display)
                javafx.scene.image.Image newThumb = FXImageUtils.matToImage(mat,
                    NodeRenderer.PROCESSING_NODE_THUMB_WIDTH,
                    NodeRenderer.PROCESSING_NODE_THUMB_HEIGHT);
                node.thumbnail = newThumb;

                // Update preview image (full resolution, cached for later display)
                javafx.scene.image.Image fullRes = FXImageUtils.matToImage(mat);
                node.previewImage = fullRes;

                // Update preview pane if this node is selected
                if (selectedNodes.contains(node) && selectedNodes.size() == 1) {
                    previewImageView.setImage(fullRes);
                }

                // Repaint canvas
                paintCanvas();
            } finally {
                // Always release the mat
                mat.release();
            }
        });
        pipelineExecutor.start();

        // Update status after executor has started and processors are created
        updatePipelineStatus();
    }

    private void updatePipelineStatus() {
        // Count threads: processor threads + 1 for JavaFX thread + 1 for each active webcam
        int threadCount = (pipelineExecutor != null ? pipelineExecutor.getThreadCount() : 0) + 1 + webcamSources.size();
        statusBar.setText("Pipeline running (" + threadCount + " thread" + (threadCount != 1 ? "s" : "") + ")");
        statusBar.setTextFill(COLOR_STATUS_RUNNING);
    }

    private void stopPipeline() {
        pipelineRunning = false;
        startStopBtn.setText("Start Pipeline");
        startStopBtn.setStyle("-fx-base: #90EE90;");  // Light green when not running
        statusBar.setText("Stopping pipeline...");
        statusBar.setTextFill(COLOR_STATUS_STOPPED);

        // Stop the pipeline executor in a background thread to avoid blocking the UI
        // The stop() method contains blocking join() calls that can cause the beach ball
        final FXPipelineExecutor executorToStop = pipelineExecutor;
        pipelineExecutor = null;

        if (executorToStop != null) {
            new Thread(() -> {
                executorToStop.stop();
                // Update status on the JavaFX thread when done
                Platform.runLater(() -> {
                    statusBar.setText("Pipeline stopped");
                });
            }, "PipelineStopThread").start();
        } else {
            statusBar.setText("Pipeline stopped");
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
        // Check for duplicate boundary nodes - only one of each type allowed
        if ("ContainerInput".equals(nodeTypeName) || "ContainerOutput".equals(nodeTypeName)) {
            for (FXNode existing : nodes) {
                if (nodeTypeName.equals(existing.nodeType)) {
                    String displayName = "ContainerInput".equals(nodeTypeName) ? "Input" : "Output";
                    javafx.scene.control.Alert alert = new javafx.scene.control.Alert(
                        javafx.scene.control.Alert.AlertType.ERROR);
                    alert.setTitle("Cannot Add Node");
                    alert.setHeaderText("Duplicate Boundary Node");
                    alert.setContentText("Only one " + displayName + " boundary node is allowed per diagram.");
                    alert.showAndWait();
                    return;
                }
            }
        }

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
     * Add a standalone Connector/Queue (a dangling connection with both ends free).
     * This creates a connection object that can be grabbed and connected to nodes later.
     */
    private void addConnectorQueue(int x, int y) {
        // Create a fully dangling connection (both source and target are null)
        FXConnection conn = FXConnection.createDangling(x, y, x + 100, y);
        connections.add(conn);

        // Select the new connection
        selectedNodes.clear();
        selectedConnections.clear();
        selectedConnections.add(conn);

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
                    System.out.println("[Preview] Webcam frame, updating preview for: " + node.label);
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

        // Store reference immediately to prevent GC before load completes
        node.thumbnail = image;
        paintCanvas();  // Trigger initial repaint (image will update as it loads)

        // Set up progress listener to handle when loading completes
        image.progressProperty().addListener((obs, oldVal, newVal) -> {
            if (newVal.doubleValue() >= 1.0) {
                if (image.isError()) {
                    node.thumbnail = null;
                } else {
                    // Update preview if this node is selected
                    if (selectedNodes.contains(node) && selectedNodes.size() == 1) {
                        System.out.println("[Preview] FileSource progress complete, updating preview for: " + node.label);
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
                paintCanvas();
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

        boolean hasMatches = false;

        for (String category : FXNodeRegistry.getCategories()) {
            List<FXNodeRegistry.NodeType> matchingNodes = new ArrayList<>();
            for (FXNodeRegistry.NodeType nodeType : FXNodeRegistry.getNodesInCategory(category)) {
                if (filter.isEmpty() ||
                    nodeType.displayName.toLowerCase().contains(filter) ||
                    nodeType.name.toLowerCase().contains(filter) ||
                    category.toLowerCase().contains(filter)) {
                    matchingNodes.add(nodeType);
                }
            }

            // Check if Connector/Queue matches for Utility category
            boolean connectorMatches = category.equals("Utility") && (filter.isEmpty() ||
                "connector".contains(filter) || "queue".contains(filter));

            if (!matchingNodes.isEmpty() || connectorMatches) {
                hasMatches = true;
                addToolbarCategory(category);
                for (FXNodeRegistry.NodeType nodeType : matchingNodes) {
                    final String typeName = nodeType.name;
                    addToolbarButton(nodeType.getButtonName(), () -> addNodeAt(typeName, getNextNodeX(), getNextNodeY()));
                }
                // Add Connector/Queue button to Utility category
                if (connectorMatches) {
                    addToolbarButton("Connector/Queue", () -> addConnectorQueue(getNextNodeX(), getNextNodeY()));
                }
            }
        }

        // Show "No matches found" message when search yields no results
        if (!hasMatches && !filter.isEmpty()) {
            Label noMatchLabel = new Label("No matches found\n\nTry fewer letters or\nclear your search");
            noMatchLabel.setFont(javafx.scene.text.Font.font("System", javafx.scene.text.FontPosture.ITALIC, 12));
            noMatchLabel.setStyle("-fx-text-alignment: center; -fx-padding: 20 10 10 10;");
            noMatchLabel.setWrapText(true);
            noMatchLabel.setMaxWidth(Double.MAX_VALUE);
            noMatchLabel.setAlignment(javafx.geometry.Pos.CENTER);
            toolbarContent.getChildren().add(noMatchLabel);
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
