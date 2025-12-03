package com.ttennebkram.pipeline.fx;

import javafx.geometry.Insets;
import javafx.geometry.Orientation;
import javafx.geometry.Pos;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.*;
import javafx.scene.image.ImageView;
import javafx.scene.input.KeyCode;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.stage.Stage;
import javafx.stage.Window;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Supplier;

/**
 * Unified pipeline editor component for editing node graphs.
 * Used by both the main application window and container sub-pipeline editors.
 */
public class FXPipelineEditor {

    // =========================== COLOR CONSTANTS ===========================
    private static final Color COLOR_TOOLBAR_BG = Color.rgb(160, 200, 160);
    private static final Color COLOR_GRID_LINES = Color.rgb(230, 230, 230);
    private static final Color COLOR_SELECTION_BOX = Color.rgb(0, 0, 255);
    private static final Color COLOR_SELECTION_BOX_FILL = Color.rgb(0, 0, 255, 0.1);
    private static final Color COLOR_STATUS_STOPPED = Color.rgb(180, 0, 0);
    private static final Color COLOR_STATUS_RUNNING = Color.rgb(0, 128, 0);
    // ========================================================================

    private final boolean isRootDiagram;
    private final Window ownerWindow;

    // UI Components
    private BorderPane rootPane;
    private Canvas canvas;
    private ScrollPane canvasScrollPane;
    private ImageView previewImageView;
    private Label statusLabel;
    private TextField searchBox;
    private VBox toolbarContent;
    private Button startStopBtn;
    private ComboBox<String> zoomCombo;
    private Tooltip canvasTooltip;
    private FXNode tooltipNode = null;

    // Zoom settings
    private double zoomLevel = 1.0;
    private static final int[] ZOOM_LEVELS = {25, 50, 75, 100, 125, 150, 200, 300, 400};

    // Pipeline data model
    private List<FXNode> nodes;
    private List<FXConnection> connections;
    private Set<FXNode> selectedNodes = new HashSet<>();
    private Set<FXConnection> selectedConnections = new HashSet<>();

    // Drag state
    private FXNode dragNode = null;
    private double dragOffsetX, dragOffsetY;
    private boolean isDragging = false;

    // Connection drawing state
    private FXNode connectionSource = null;
    private FXNode connectionTarget = null;
    private int connectionOutputIndex = 0;
    private int connectionInputIndex = 0;
    private double connectionEndX, connectionEndY;
    private boolean isDrawingConnection = false;
    private boolean isDrawingReverseConnection = false;
    private FXConnection yankingConnection = null;
    private boolean yankingFromTarget = false;

    // Selection box state
    private double selectionBoxStartX, selectionBoxStartY;
    private double selectionBoxEndX, selectionBoxEndY;
    private boolean isSelectionBoxDragging = false;

    // Base path for resolving relative file paths
    private String basePath;

    // Open container editor windows (so we can close them all on File New/Open)
    private final Set<FXContainerEditorWindow> openContainerWindows = new HashSet<>();

    // ========================= CALLBACKS =========================
    // These callbacks allow the parent to control pipeline-specific behavior

    /** Called when the editor content is modified */
    private Runnable onModified;

    /** Called to start the pipeline */
    private Runnable onStartPipeline;

    /** Called to stop the pipeline */
    private Runnable onStopPipeline;

    /** Called to refresh/trigger the pipeline */
    private Runnable onRefreshPipeline;

    /** Returns whether the pipeline is currently running */
    private Supplier<Boolean> isPipelineRunning;

    /** Returns the current thread count */
    private Supplier<Integer> getThreadCount;

    /** Called to request global save (parent document) */
    private Runnable onRequestGlobalSave;

    /** Called to quit the application */
    private Runnable onQuit;

    /** Called to restart the application */
    private Runnable onRestart;

    /**
     * Create a new pipeline editor.
     *
     * @param isRootDiagram True for top-level pipelines, false for sub-pipelines (container contents).
     *                      Non-root diagrams auto-create ContainerInput/ContainerOutput boundary nodes.
     * @param ownerWindow The parent window (Stage for main, or parent Stage for container)
     * @param nodes The list of nodes to edit (will be modified in place)
     * @param connections The list of connections to edit (will be modified in place)
     */
    public FXPipelineEditor(boolean isRootDiagram, Window ownerWindow, List<FXNode> nodes, List<FXConnection> connections) {
        this.isRootDiagram = isRootDiagram;
        this.ownerWindow = ownerWindow;
        this.nodes = nodes;
        this.connections = connections;

        buildUI();
    }

    // ========================= PUBLIC API =========================

    /**
     * Get the root pane containing the entire editor UI.
     * Embed this in your window/scene.
     */
    public BorderPane getRootPane() {
        return rootPane;
    }

    /**
     * Get the canvas for direct access (e.g., for installing event handlers).
     */
    public Canvas getCanvas() {
        return canvas;
    }

    /**
     * Get the preview ImageView for updating from external sources.
     */
    public ImageView getPreviewImageView() {
        return previewImageView;
    }

    /**
     * Set the base path for resolving relative file paths.
     */
    public void setBasePath(String basePath) {
        this.basePath = basePath;
    }

    /**
     * Get the base path.
     */
    public String getBasePath() {
        return basePath;
    }

    /**
     * Get the currently selected nodes.
     */
    public Set<FXNode> getSelectedNodes() {
        return selectedNodes;
    }

    /**
     * Get the currently selected connections.
     */
    public Set<FXConnection> getSelectedConnections() {
        return selectedConnections;
    }

    /**
     * Handle arrow key press to move selected nodes.
     * Returns true if the key was handled (nodes were moved).
     */
    public boolean handleArrowKey(KeyCode keyCode) {
        if (selectedNodes.isEmpty()) {
            return false;
        }
        double delta = 1.0 / zoomLevel;
        boolean moved = false;
        switch (keyCode) {
            case UP:
                for (FXNode node : selectedNodes) { node.y -= delta; }
                moved = true;
                break;
            case DOWN:
                for (FXNode node : selectedNodes) { node.y += delta; }
                moved = true;
                break;
            case LEFT:
                for (FXNode node : selectedNodes) { node.x -= delta; }
                moved = true;
                break;
            case RIGHT:
                for (FXNode node : selectedNodes) { node.x += delta; }
                moved = true;
                break;
            default:
                break;
        }
        if (moved) {
            notifyModified();
            paintCanvas();
        }
        return moved;
    }

    /**
     * Close all open container editor windows.
     * Called when creating a new diagram or loading a different file.
     */
    public void closeAllContainerWindows() {
        // Make a copy to avoid concurrent modification
        Set<FXContainerEditorWindow> windowsCopy = new HashSet<>(openContainerWindows);
        for (FXContainerEditorWindow window : windowsCopy) {
            window.close();
        }
        openContainerWindows.clear();
    }

    /**
     * Repaint the canvas.
     */
    public void paintCanvas() {
        GraphicsContext gc = canvas.getGraphicsContext2D();

        gc.save();
        gc.scale(zoomLevel, zoomLevel);

        // Clear background
        gc.setFill(Color.WHITE);
        gc.fillRect(0, 0, canvas.getWidth() / zoomLevel, canvas.getHeight() / zoomLevel);

        // Draw grid
        gc.setStroke(COLOR_GRID_LINES);
        gc.setLineWidth(1);
        double gridSize = 20;
        for (double x = 0; x < canvas.getWidth() / zoomLevel; x += gridSize) {
            gc.strokeLine(x, 0, x, canvas.getHeight() / zoomLevel);
        }
        for (double y = 0; y < canvas.getHeight() / zoomLevel; y += gridSize) {
            gc.strokeLine(0, y, canvas.getWidth() / zoomLevel, y);
        }

        // Draw connections (including dangling connections with free endpoints)
        for (FXConnection conn : connections) {
            boolean isSelected = selectedConnections.contains(conn);
            double[] start = conn.getStartPoint();
            double[] end = conn.getEndPoint();
            if (start != null && end != null) {
                NodeRenderer.renderConnection(gc, start[0], start[1], end[0], end[1], isSelected,
                    conn.queueSize, conn.totalFrames,
                    conn.source != null, conn.target != null);
            }
        }

        // Draw in-progress connection (forward: from output)
        if (isDrawingConnection && connectionSource != null && yankingConnection == null) {
            double[] startPt = connectionSource.getOutputPoint(connectionOutputIndex);
            if (startPt != null) {
                NodeRenderer.renderConnection(gc, startPt[0], startPt[1], connectionEndX, connectionEndY, true);
            }
        }

        // Draw in-progress reverse connection (from input)
        if (isDrawingReverseConnection && connectionTarget != null) {
            double[] endPt = connectionTarget.getInputPoint(connectionInputIndex);
            if (endPt != null) {
                NodeRenderer.renderConnection(gc, connectionEndX, connectionEndY, endPt[0], endPt[1], true);
            }
        }

        // Draw nodes
        for (FXNode node : nodes) {
            boolean isSelected = selectedNodes.contains(node);
            int[] outputCounters = new int[] { node.outputCount1, node.outputCount2, node.outputCount3, node.outputCount4 };
            NodeRenderer.renderNode(gc, node.x, node.y, node.width, node.height,
                node.label, isSelected, node.enabled, node.backgroundColor,
                node.hasInput, node.hasDualInput, node.outputCount,
                node.thumbnail, node.isContainer,
                node.inputCount, node.inputCount2, outputCounters, node.nodeType, node.isBoundaryNode);

            // Draw stats line
            boolean isSourceNode = !node.hasInput && !node.isBoundaryNode;
            if (isSourceNode) {
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
    }

    /**
     * Update the pipeline start/stop button state based on running status.
     */
    public void updatePipelineButtonState() {
        if (isPipelineRunning != null && isPipelineRunning.get()) {
            startStopBtn.setText("Stop Pipeline");
            startStopBtn.setStyle("-fx-base: #F08080;");
            if (getThreadCount != null) {
                int threads = getThreadCount.get();
                statusLabel.setText("Pipeline running (" + threads + " thread" + (threads != 1 ? "s" : "") + ")");
            } else {
                statusLabel.setText("Pipeline running");
            }
            statusLabel.setTextFill(COLOR_STATUS_RUNNING);
        } else {
            startStopBtn.setText("Start Pipeline");
            startStopBtn.setStyle("-fx-base: #90EE90;");
            statusLabel.setText("Pipeline stopped");
            statusLabel.setTextFill(COLOR_STATUS_STOPPED);
        }
    }

    /**
     * Update the status label text.
     */
    public void setStatus(String text) {
        statusLabel.setText(text);
    }

    /**
     * Update the status label with node count.
     */
    public void updateStatus() {
        statusLabel.setText("Nodes: " + nodes.size());
    }

    // ========================= CALLBACK SETTERS =========================

    public void setOnModified(Runnable onModified) {
        this.onModified = onModified;
    }

    public void setOnStartPipeline(Runnable onStartPipeline) {
        this.onStartPipeline = onStartPipeline;
    }

    public void setOnStopPipeline(Runnable onStopPipeline) {
        this.onStopPipeline = onStopPipeline;
    }

    public void setOnRefreshPipeline(Runnable onRefreshPipeline) {
        this.onRefreshPipeline = onRefreshPipeline;
    }

    public void setIsPipelineRunning(Supplier<Boolean> isPipelineRunning) {
        this.isPipelineRunning = isPipelineRunning;
    }

    public void setGetThreadCount(Supplier<Integer> getThreadCount) {
        this.getThreadCount = getThreadCount;
    }

    public void setOnRequestGlobalSave(Runnable onRequestGlobalSave) {
        this.onRequestGlobalSave = onRequestGlobalSave;
    }

    public void setOnQuit(Runnable onQuit) {
        this.onQuit = onQuit;
    }

    public void setOnRestart(Runnable onRestart) {
        this.onRestart = onRestart;
    }

    // ========================= UI BUILDING =========================

    private void buildUI() {
        rootPane = new BorderPane();

        // Toolbar on left
        rootPane.setLeft(createToolbar());

        // Main content - SplitPane with canvas and preview
        VBox canvasWithStatus = new VBox();
        canvasWithStatus.getChildren().addAll(createCanvasPane(), createStatusBar());
        VBox.setVgrow(canvasWithStatus.getChildren().get(0), Priority.ALWAYS);

        SplitPane splitPane = new SplitPane();
        splitPane.setOrientation(Orientation.HORIZONTAL);
        splitPane.getItems().addAll(canvasWithStatus, createPreviewPane());
        splitPane.setDividerPositions(0.75);
        rootPane.setCenter(splitPane);

        // For sub-pipelines (non-root), ensure boundary nodes exist
        if (!isRootDiagram) {
            ensureBoundaryNodes();
        }
    }

    private VBox createToolbar() {
        VBox toolbar = new VBox(8);
        toolbar.setPadding(new Insets(8));
        toolbar.setStyle("-fx-background-color: rgb(160, 200, 160);");
        toolbar.setPrefWidth(200);

        // Search box with clear button
        searchBox = new TextField();
        searchBox.setPromptText("Search nodes...");
        searchBox.textProperty().addListener((obs, oldVal, newVal) -> filterToolbarButtons());
        searchBox.setStyle("-fx-padding: 2 20 2 5;");

        Button clearSearchBtn = new Button("\u00D7");
        clearSearchBtn.setStyle("-fx-font-size: 12px; -fx-padding: 0 5 0 5; -fx-background-color: transparent; -fx-cursor: hand;");
        clearSearchBtn.setOnAction(e -> searchBox.clear());
        clearSearchBtn.visibleProperty().bind(searchBox.textProperty().isNotEmpty());

        StackPane searchStack = new StackPane();
        searchStack.getChildren().addAll(searchBox, clearSearchBtn);
        StackPane.setAlignment(clearSearchBtn, Pos.CENTER_RIGHT);
        StackPane.setMargin(clearSearchBtn, new Insets(0, 2, 0, 0));
        toolbar.getChildren().add(searchStack);

        // Scrollable content for node buttons
        ScrollPane scrollPane = new ScrollPane();
        scrollPane.setFitToWidth(true);
        scrollPane.setVbarPolicy(ScrollPane.ScrollBarPolicy.AS_NEEDED);
        scrollPane.setHbarPolicy(ScrollPane.ScrollBarPolicy.NEVER);
        scrollPane.setStyle("-fx-background: rgb(160, 200, 160); -fx-background-color: rgb(160, 200, 160); -fx-control-inner-background: rgb(160, 200, 160);");
        VBox.setVgrow(scrollPane, Priority.ALWAYS);

        toolbarContent = new VBox(0);
        toolbarContent.setPadding(new Insets(4, 8, 4, 8));
        toolbarContent.setStyle("-fx-background-color: rgb(160, 200, 160);");
        scrollPane.setContent(toolbarContent);

        // Populate node buttons from registry (all categories visible)
        for (String category : FXNodeRegistry.getCategories()) {
            addToolbarCategory(category);
            for (FXNodeRegistry.NodeType nodeType : FXNodeRegistry.getNodesInCategory(category)) {
                final String typeName = nodeType.name;
                addToolbarButton(nodeType.getButtonName(), () -> addNode(typeName));
            }
            // Add Connector/Queue button to Utility category
            if (category.equals("Utility")) {
                addToolbarButton("Connector/Queue", this::addConnectorQueue);
            }
        }

        toolbar.getChildren().add(scrollPane);
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
        canvas = new Canvas(2000, 2000);

        Pane canvasContainer = new Pane(canvas);
        canvasContainer.setStyle("-fx-background-color: white;");

        canvasScrollPane = new ScrollPane(canvasContainer);
        canvasScrollPane.setPannable(false);

        // Mouse event handlers
        canvas.setOnMousePressed(this::handleMousePressed);
        canvas.setOnMouseDragged(this::handleMouseDragged);
        canvas.setOnMouseReleased(this::handleMouseReleased);
        canvas.setOnMouseMoved(this::handleMouseMoved);
        canvas.setOnMouseClicked(e -> {
            if (e.getClickCount() == 2) {
                handleDoubleClick(e);
            }
        });

        // Initialize tooltip
        canvasTooltip = new Tooltip();
        canvasTooltip.setShowDelay(javafx.util.Duration.millis(500));

        // Keyboard handler
        canvas.setFocusTraversable(true);
        canvas.setOnKeyPressed(e -> {
            if (e.getCode() == KeyCode.DELETE || e.getCode() == KeyCode.BACK_SPACE) {
                deleteSelected();
                e.consume();
            } else if (!selectedNodes.isEmpty()) {
                double delta = 1.0 / zoomLevel;
                boolean moved = false;
                switch (e.getCode()) {
                    case UP:
                        for (FXNode node : selectedNodes) { node.y -= delta; }
                        moved = true;
                        break;
                    case DOWN:
                        for (FXNode node : selectedNodes) { node.y += delta; }
                        moved = true;
                        break;
                    case LEFT:
                        for (FXNode node : selectedNodes) { node.x -= delta; }
                        moved = true;
                        break;
                    case RIGHT:
                        for (FXNode node : selectedNodes) { node.x += delta; }
                        moved = true;
                        break;
                    default:
                        break;
                }
                if (moved) {
                    notifyModified();
                    paintCanvas();
                    e.consume();
                }
            }
        });

        return canvasScrollPane;
    }

    private VBox createPreviewPane() {
        VBox previewPane = new VBox(5);
        previewPane.setPadding(new Insets(10));
        previewPane.setStyle("-fx-background-color: #f0f0f0;");
        previewPane.setMinWidth(100);  // Allow shrinking to a small width

        // Save and Cancel buttons
        HBox buttonPanel = new HBox(5);
        buttonPanel.setMaxWidth(Double.MAX_VALUE);

        Button saveButton = new Button("Save");
        saveButton.setMaxWidth(Double.MAX_VALUE);
        HBox.setHgrow(saveButton, Priority.ALWAYS);
        saveButton.setOnAction(e -> handleSave());

        Button cancelButton = new Button("Cancel");
        cancelButton.setMaxWidth(Double.MAX_VALUE);
        HBox.setHgrow(cancelButton, Priority.ALWAYS);
        cancelButton.setOnAction(e -> handleCancel());

        buttonPanel.getChildren().addAll(saveButton, cancelButton);

        // Start/Stop pipeline button
        startStopBtn = new Button("Start Pipeline");
        startStopBtn.setStyle("-fx-base: #90EE90;");
        startStopBtn.setOnAction(e -> togglePipeline());
        startStopBtn.setMaxWidth(Double.MAX_VALUE);

        // Instructions
        Label instructionsLabel = new Label("Instructions:");
        instructionsLabel.setStyle("-fx-font-weight: bold; -fx-padding: 5 0 0 0;");

        Label instructions = new Label(NodeRenderer.INSTRUCTIONS_TEXT);
        instructions.setStyle("-fx-font-size: 11px;");
        instructions.setWrapText(true);
        instructions.setMinWidth(0);  // Allow label to shrink

        Label previewLabel = new Label("Preview");
        previewLabel.setStyle("-fx-font-weight: bold; -fx-padding: 10 0 0 0;");

        previewImageView = new ImageView();
        previewImageView.setPreserveRatio(true);

        StackPane imageContainer = new StackPane(previewImageView);
        imageContainer.setStyle("-fx-background-color: #cccccc; -fx-min-height: 200;");
        VBox.setVgrow(imageContainer, Priority.ALWAYS);

        previewImageView.fitWidthProperty().bind(imageContainer.widthProperty().subtract(10));
        previewImageView.fitHeightProperty().bind(imageContainer.heightProperty().subtract(10));

        previewPane.getChildren().addAll(buttonPanel, startStopBtn, instructionsLabel, instructions, previewLabel, imageContainer);

        return previewPane;
    }

    private HBox createStatusBar() {
        HBox statusBarBox = new HBox(10);
        statusBarBox.setPadding(new Insets(5, 10, 5, 10));
        statusBarBox.setStyle("-fx-background-color: rgb(160, 160, 160);");
        statusBarBox.setAlignment(Pos.CENTER_LEFT);

        // Zoom controls
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

        // Node count
        Label nodeCountLabel = new Label("Nodes: 0");

        Region spacer1 = new Region();
        HBox.setHgrow(spacer1, Priority.ALWAYS);

        // Status label
        statusLabel = new Label("Ready");
        statusLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 12px;");

        Region spacer2 = new Region();
        HBox.setHgrow(spacer2, Priority.ALWAYS);

        statusBarBox.getChildren().addAll(nodeCountLabel, spacer1, statusLabel, spacer2, zoomLabel, zoomCombo);

        return statusBarBox;
    }

    // ========================= BOUNDARY NODES (Container Mode) =========================

    /**
     * Ensure the container has ContainerInput and ContainerOutput boundary nodes.
     * Only called in CONTAINER mode.
     */
    private void ensureBoundaryNodes() {
        boolean hasInput = false;
        boolean hasOutput = false;

        for (FXNode node : nodes) {
            if ("ContainerInput".equals(node.nodeType)) hasInput = true;
            if ("ContainerOutput".equals(node.nodeType)) hasOutput = true;
        }

        if (!hasInput) {
            FXNode inputNode = FXNodeFactory.createFXNode("ContainerInput", 50, 50);
            nodes.add(0, inputNode);
        }
        if (!hasOutput) {
            FXNode outputNode = FXNodeFactory.createFXNode("ContainerOutput", 500, 50);
            nodes.add(outputNode);
        }
    }

    // ========================= MOUSE HANDLERS =========================

    private void handleMousePressed(javafx.scene.input.MouseEvent e) {
        double canvasX = e.getX() / zoomLevel;
        double canvasY = e.getY() / zoomLevel;

        // Check checkbox and help icon clicks
        for (FXNode node : nodes) {
            if (node.isOnCheckbox(canvasX, canvasY)) {
                node.enabled = !node.enabled;
                notifyModified();
                paintCanvas();
                return;
            }
            if (node.isOnHelpIcon(canvasX, canvasY)) {
                Stage stage = (ownerWindow instanceof Stage) ? (Stage) ownerWindow : null;
                FXHelpBrowser.openForNodeType(stage, node.nodeType);
                return;
            }
        }

        // Check for clicks on free endpoints of dangling connections
        double tolerance = NodeRenderer.CONNECTION_RADIUS + 9;
        for (FXConnection conn : connections) {
            if (conn.source == null) {
                double dist = Math.sqrt(Math.pow(canvasX - conn.freeSourceX, 2) + Math.pow(canvasY - conn.freeSourceY, 2));
                if (dist < tolerance) {
                    yankingConnection = conn;
                    yankingFromTarget = false;
                    isDrawingConnection = true;
                    paintCanvas();
                    return;
                }
            }
            if (conn.target == null) {
                double dist = Math.sqrt(Math.pow(canvasX - conn.freeTargetX, 2) + Math.pow(canvasY - conn.freeTargetY, 2));
                if (dist < tolerance) {
                    yankingConnection = conn;
                    yankingFromTarget = true;
                    isDrawingConnection = true;
                    paintCanvas();
                    return;
                }
            }
        }

        // Check if clicking on a node's output point
        for (FXNode node : nodes) {
            int outputIdx = node.getOutputPointAt(canvasX, canvasY, NodeRenderer.CONNECTION_RADIUS + 3);
            if (outputIdx >= 0) {
                FXConnection existingConn = findConnectionFromOutput(node, outputIdx);
                if (existingConn != null) {
                    double[] sourcePt = node.getOutputPoint(outputIdx);
                    if (sourcePt != null) {
                        existingConn.freeSourceX = sourcePt[0];
                        existingConn.freeSourceY = sourcePt[1];
                    } else {
                        existingConn.freeSourceX = canvasX;
                        existingConn.freeSourceY = canvasY;
                    }
                    existingConn.source = null;
                    yankingConnection = existingConn;
                    yankingFromTarget = false;
                    isDrawingConnection = true;
                    notifyModified();
                    paintCanvas();
                    return;
                }
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

        // Check if clicking on a node's input point
        for (FXNode node : nodes) {
            int inputIdx = node.getInputPointAt(canvasX, canvasY, NodeRenderer.CONNECTION_RADIUS + 3);
            if (inputIdx >= 0) {
                FXConnection existingConn = findConnectionToInput(node, inputIdx);
                if (existingConn != null) {
                    double[] targetPt = node.getInputPoint(inputIdx);
                    if (targetPt != null) {
                        existingConn.freeTargetX = targetPt[0];
                        existingConn.freeTargetY = targetPt[1];
                    } else {
                        existingConn.freeTargetX = canvasX;
                        existingConn.freeTargetY = canvasY;
                    }
                    existingConn.target = null;
                    yankingConnection = existingConn;
                    yankingFromTarget = true;
                    isDrawingConnection = true;
                    notifyModified();
                    paintCanvas();
                    return;
                }
                return;
            }
        }

        // Check if clicking on a node
        FXNode clickedNode = getNodeAt(canvasX, canvasY);
        if (clickedNode != null) {
            if (!e.isShiftDown() && !selectedNodes.contains(clickedNode)) {
                selectedNodes.clear();
                selectedConnections.clear();
            }
            selectedNodes.add(clickedNode);

            // Update preview
            if (selectedNodes.size() == 1 && clickedNode.previewImage != null) {
                previewImageView.setImage(clickedNode.previewImage);
            }

            dragNode = clickedNode;
            dragOffsetX = canvasX - clickedNode.x;
            dragOffsetY = canvasY - clickedNode.y;
            isDragging = true;
            paintCanvas();
            return;
        }

        // Check if clicking on a connection
        FXConnection clickedConn = getConnectionAt(canvasX, canvasY, 8);
        if (clickedConn != null) {
            if (!e.isShiftDown()) {
                selectedNodes.clear();
                selectedConnections.clear();
            }
            selectedConnections.add(clickedConn);
            paintCanvas();
            return;
        }

        // Clicked on empty space - start selection box
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
                if (yankingFromTarget) {
                    yankingConnection.freeTargetX = canvasX;
                    yankingConnection.freeTargetY = canvasY;
                } else {
                    yankingConnection.freeSourceX = canvasX;
                    yankingConnection.freeSourceY = canvasY;
                }
            } else {
                connectionEndX = canvasX;
                connectionEndY = canvasY;
            }
            paintCanvas();
        } else if (isDragging && dragNode != null) {
            double dx = canvasX - dragOffsetX - dragNode.x;
            double dy = canvasY - dragOffsetY - dragNode.y;

            for (FXNode node : selectedNodes) {
                node.x += dx;
                node.y += dy;
            }

            dragNode.x = canvasX - dragOffsetX;
            dragNode.y = canvasY - dragOffsetY;
            notifyModified();
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
            double tolerance = NodeRenderer.CONNECTION_RADIUS + 9;

            if (yankingConnection != null) {
                if (yankingFromTarget) {
                    for (FXNode targetNode : nodes) {
                        if (targetNode != yankingConnection.source) {
                            int inputIdx = targetNode.getInputPointAt(canvasX, canvasY, tolerance);
                            if (inputIdx >= 0) {
                                FXConnection existingConn = findConnectionToInput(targetNode, inputIdx);
                                if (existingConn != null && existingConn != yankingConnection) {
                                    break;
                                }
                                yankingConnection.reconnectTarget(targetNode, inputIdx);
                                connected = true;
                                notifyModified();
                                break;
                            }
                        }
                    }
                    if (!connected) {
                        double[] pushed = pushAwayFromConnectionPoints(canvasX, canvasY, 30);
                        yankingConnection.freeTargetX = pushed[0];
                        yankingConnection.freeTargetY = pushed[1];
                        notifyModified();
                    }
                } else {
                    for (FXNode sourceNode : nodes) {
                        if (sourceNode != yankingConnection.target) {
                            int outputIdx = sourceNode.getOutputPointAt(canvasX, canvasY, tolerance);
                            if (outputIdx >= 0) {
                                FXConnection existingConn = findConnectionFromOutput(sourceNode, outputIdx);
                                if (existingConn != null && existingConn != yankingConnection) {
                                    break;
                                }
                                yankingConnection.reconnectSource(sourceNode, outputIdx);
                                connected = true;
                                notifyModified();
                                break;
                            }
                        }
                    }
                    if (!connected) {
                        double[] pushed = pushAwayFromConnectionPoints(canvasX, canvasY, 30);
                        yankingConnection.freeSourceX = pushed[0];
                        yankingConnection.freeSourceY = pushed[1];
                        notifyModified();
                    }
                }
                yankingConnection = null;
            } else if (connectionSource != null) {
                for (FXNode targetNode : nodes) {
                    if (targetNode != connectionSource) {
                        int inputIdx = targetNode.getInputPointAt(canvasX, canvasY, tolerance);
                        if (inputIdx >= 0) {
                            FXConnection existingConn = findConnectionToInput(targetNode, inputIdx);
                            if (existingConn != null) {
                                break;
                            }
                            FXConnection conn = new FXConnection(connectionSource, connectionOutputIndex, targetNode, inputIdx);
                            connections.add(conn);
                            notifyModified();
                            connected = true;
                            break;
                        }
                    }
                }
                if (!connected && connectionSource != null) {
                    double[] pushed = pushAwayFromConnectionPoints(canvasX, canvasY, 30);
                    FXConnection danglingConn = FXConnection.createFromSource(connectionSource, connectionOutputIndex, pushed[0], pushed[1]);
                    connections.add(danglingConn);
                    notifyModified();
                }
            }

            isDrawingConnection = false;
            connectionSource = null;
            paintCanvas();
        }

        if (isDrawingReverseConnection && connectionTarget != null) {
            double tolerance = NodeRenderer.CONNECTION_RADIUS + 9;
            boolean connected = false;
            for (FXNode sourceNode : nodes) {
                if (sourceNode != connectionTarget) {
                    int outputIdx = sourceNode.getOutputPointAt(canvasX, canvasY, tolerance);
                    if (outputIdx >= 0) {
                        FXConnection conn = new FXConnection(sourceNode, outputIdx, connectionTarget, connectionInputIndex);
                        connections.add(conn);
                        notifyModified();
                        connected = true;
                        break;
                    }
                }
            }
            isDrawingReverseConnection = false;
            connectionTarget = null;
            paintCanvas();
        }

        if (isSelectionBoxDragging) {
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

            for (FXConnection conn : connections) {
                if (connectionIntersectsRect(conn, x1, y1, x2, y2)) {
                    selectedConnections.add(conn);
                }
            }
            isSelectionBoxDragging = false;
        }

        if (isDragging) {
            notifyModified();
        }
        dragNode = null;
        isDragging = false;
        paintCanvas();
    }

    private void handleMouseMoved(javafx.scene.input.MouseEvent e) {
        double canvasX = e.getX() / zoomLevel;
        double canvasY = e.getY() / zoomLevel;

        FXNode hoveredNode = getNodeAt(canvasX, canvasY);
        boolean onCheckbox = hoveredNode != null && hoveredNode.isOnCheckbox(canvasX, canvasY);
        boolean onHelpIcon = hoveredNode != null && hoveredNode.isOnHelpIcon(canvasX, canvasY);

        // Check for connection point hover
        FXNode connectionPointNode = null;
        boolean isInputPoint = false;
        int connectionPointIndex = -1;
        double tolerance = 10.0;

        for (FXNode node : nodes) {
            int inputIdx = node.getInputPointAt(canvasX, canvasY, tolerance);
            if (inputIdx >= 0) {
                connectionPointNode = node;
                isInputPoint = true;
                connectionPointIndex = inputIdx;
                break;
            }
            int outputIdx = node.getOutputPointAt(canvasX, canvasY, tolerance);
            if (outputIdx >= 0) {
                connectionPointNode = node;
                isInputPoint = false;
                connectionPointIndex = outputIdx;
                break;
            }
        }

        // Update cursor
        if (onCheckbox || onHelpIcon || connectionPointNode != null) {
            canvas.setCursor(javafx.scene.Cursor.HAND);
        } else {
            canvas.setCursor(javafx.scene.Cursor.DEFAULT);
        }

        // Update tooltip
        String tooltipText = null;
        if (onCheckbox) {
            tooltipText = "Enable / Disable this Node";
        } else if (onHelpIcon) {
            tooltipText = FXHelpBrowser.hasHelp(hoveredNode.nodeType) ? "Help" : "Help (not available for this node type)";
        } else if (connectionPointNode != null) {
            tooltipText = isInputPoint ? "Input " + (connectionPointIndex + 1) : "Output " + (connectionPointIndex + 1);
            if (isInputPoint) {
                FXConnection existingConn = findConnectionToInput(connectionPointNode, connectionPointIndex);
                if (existingConn != null) {
                    tooltipText += "\n(Connected - yank existing wire to replace)";
                }
            }
        }

        if (tooltipText != null) {
            canvasTooltip.setText(tooltipText);
            Tooltip.install(canvas, canvasTooltip);
        } else {
            Tooltip.uninstall(canvas, canvasTooltip);
        }
        tooltipNode = hoveredNode;
    }

    private void handleDoubleClick(javafx.scene.input.MouseEvent e) {
        double canvasX = e.getX() / zoomLevel;
        double canvasY = e.getY() / zoomLevel;

        FXNode node = getNodeAt(canvasX, canvasY);
        if (node != null) {
            if (node.isContainer) {
                openNestedContainer(node);
            } else {
                showNodeProperties(node);
            }
        }
    }

    // ========================= NODE OPERATIONS =========================

    private void addNode(String nodeType) {
        // Check for duplicate boundary nodes
        if ("ContainerInput".equals(nodeType) || "ContainerOutput".equals(nodeType)) {
            for (FXNode existing : nodes) {
                if (nodeType.equals(existing.nodeType)) {
                    String displayName = "ContainerInput".equals(nodeType) ? "Input" : "Output";
                    Alert alert = new Alert(Alert.AlertType.ERROR);
                    alert.setTitle("Cannot Add Node");
                    alert.setHeaderText("Duplicate Boundary Node");
                    alert.setContentText("Only one " + displayName + " boundary node is allowed per container.");
                    alert.showAndWait();
                    return;
                }
            }
        }

        double x = 50 + (nodes.size() % 5) * 30;
        double y = 50 + (nodes.size() / 5) * 30;

        FXNode node = FXNodeFactory.createFXNode(nodeType, (int) x, (int) y);
        nodes.add(node);
        selectedNodes.clear();
        selectedNodes.add(node);
        paintCanvas();
        updateStatus();
        notifyModified();
    }

    private void addConnectorQueue() {
        double x = 100;
        double y = 100 + nodes.size() * 30;

        FXConnection conn = FXConnection.createDangling(x, y, x + 100, y);
        connections.add(conn);

        selectedNodes.clear();
        selectedConnections.clear();
        selectedConnections.add(conn);

        paintCanvas();
        notifyModified();
    }

    /**
     * Delete all selected nodes and connections.
     */
    public void deleteSelected() {
        connections.removeAll(selectedConnections);

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

        nodes.removeAll(selectedNodes);

        selectedNodes.clear();
        selectedConnections.clear();
        paintCanvas();
        updateStatus();
        notifyModified();
    }

    public void selectAll() {
        selectedNodes.clear();
        selectedConnections.clear();
        selectedNodes.addAll(nodes);
        selectedConnections.addAll(connections);
        paintCanvas();
        updateStatus();
    }

    // ========================= HELPER METHODS =========================

    private FXNode getNodeAt(double x, double y) {
        for (int i = nodes.size() - 1; i >= 0; i--) {
            FXNode node = nodes.get(i);
            if (node.contains(x, y)) {
                return node;
            }
        }
        return null;
    }

    private FXConnection getConnectionAt(double px, double py, double tolerance) {
        for (FXConnection conn : connections) {
            double[] startPt = conn.getStartPoint();
            double[] endPt = conn.getEndPoint();
            if (startPt != null && endPt != null) {
                if (isPointNearBezier(px, py, startPt[0], startPt[1], endPt[0], endPt[1], tolerance)) {
                    return conn;
                }
            }
        }
        return null;
    }

    private FXConnection findConnectionFromOutput(FXNode node, int outputIndex) {
        double[] outputPt = node.getOutputPoint(outputIndex);
        for (FXConnection conn : connections) {
            if (conn.source == node && conn.sourceOutputIndex == outputIndex) {
                return conn;
            }
            if (conn.source == null && outputPt != null) {
                double dist = Math.sqrt(Math.pow(conn.freeSourceX - outputPt[0], 2) + Math.pow(conn.freeSourceY - outputPt[1], 2));
                if (dist < 15) {
                    return conn;
                }
            }
        }
        return null;
    }

    private FXConnection findConnectionToInput(FXNode node, int inputIndex) {
        double[] inputPt = node.getInputPoint(inputIndex);
        for (FXConnection conn : connections) {
            if (conn.target == node && conn.targetInputIndex == inputIndex) {
                return conn;
            }
            if (conn.target == null && inputPt != null) {
                double dist = Math.sqrt(Math.pow(conn.freeTargetX - inputPt[0], 2) + Math.pow(conn.freeTargetY - inputPt[1], 2));
                if (dist < 15) {
                    return conn;
                }
            }
        }
        return null;
    }

    private boolean isPointNearBezier(double px, double py, double startX, double startY, double endX, double endY, double tolerance) {
        double ctrlOffset = Math.abs(endX - startX) / 2;
        if (ctrlOffset < 30) ctrlOffset = 30;
        double ctrl1X = startX + ctrlOffset;
        double ctrl1Y = startY;
        double ctrl2X = endX - ctrlOffset;
        double ctrl2Y = endY;

        int samples = 20;
        for (int i = 0; i <= samples; i++) {
            double t = (double) i / samples;
            double oneMinusT = 1 - t;
            double bx = oneMinusT * oneMinusT * oneMinusT * startX
                      + 3 * oneMinusT * oneMinusT * t * ctrl1X
                      + 3 * oneMinusT * t * t * ctrl2X
                      + t * t * t * endX;
            double by = oneMinusT * oneMinusT * oneMinusT * startY
                      + 3 * oneMinusT * oneMinusT * t * ctrl1Y
                      + 3 * oneMinusT * t * t * ctrl2Y
                      + t * t * t * endY;

            double dist = Math.sqrt((px - bx) * (px - bx) + (py - by) * (py - by));
            if (dist <= tolerance) {
                return true;
            }
        }
        return false;
    }

    private boolean connectionIntersectsRect(FXConnection conn, double x1, double y1, double x2, double y2) {
        if (conn.source == null || conn.target == null) return false;
        double[] startPt = conn.source.getOutputPoint(conn.sourceOutputIndex);
        double[] endPt = conn.target.getInputPoint(conn.targetInputIndex);
        if (startPt == null || endPt == null) return false;

        double startX = startPt[0], startY = startPt[1];
        double endX = endPt[0], endY = endPt[1];

        double ctrlOffset = Math.abs(endX - startX) / 2;
        if (ctrlOffset < 30) ctrlOffset = 30;
        double ctrl1X = startX + ctrlOffset;
        double ctrl1Y = startY;
        double ctrl2X = endX - ctrlOffset;
        double ctrl2Y = endY;

        int samples = 20;
        for (int i = 0; i <= samples; i++) {
            double t = (double) i / samples;
            double oneMinusT = 1 - t;
            double bx = oneMinusT * oneMinusT * oneMinusT * startX
                      + 3 * oneMinusT * oneMinusT * t * ctrl1X
                      + 3 * oneMinusT * t * t * ctrl2X
                      + t * t * t * endX;
            double by = oneMinusT * oneMinusT * oneMinusT * startY
                      + 3 * oneMinusT * oneMinusT * t * ctrl1Y
                      + 3 * oneMinusT * t * t * ctrl2Y
                      + t * t * t * endY;

            if (bx >= x1 && bx <= x2 && by >= y1 && by <= y2) {
                return true;
            }
        }
        return false;
    }

    private double[] pushAwayFromConnectionPoints(double x, double y, double minDistance) {
        double newX = x;
        double newY = y;

        for (FXNode node : nodes) {
            int inputCount = node.hasInput ? (node.hasDualInput ? 2 : 1) : 0;
            for (int i = 0; i < inputCount; i++) {
                double[] inputPt = node.getInputPoint(i);
                if (inputPt != null) {
                    double dx = x - inputPt[0];
                    double dy = y - inputPt[1];
                    double dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < minDistance && dist > 0) {
                        double scale = minDistance / dist;
                        newX = inputPt[0] + dx * scale;
                        newY = inputPt[1] + dy * scale;
                    } else if (dist == 0) {
                        newX = x + minDistance;
                    }
                }
            }

            for (int i = 0; i < node.outputCount; i++) {
                double[] outputPt = node.getOutputPoint(i);
                if (outputPt != null) {
                    double dx = x - outputPt[0];
                    double dy = y - outputPt[1];
                    double dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < minDistance && dist > 0) {
                        double scale = minDistance / dist;
                        newX = outputPt[0] + dx * scale;
                        newY = outputPt[1] + dy * scale;
                    } else if (dist == 0) {
                        newX = x + minDistance;
                    }
                }
            }
        }

        return new double[] { newX, newY };
    }

    private void filterToolbarButtons() {
        String filter = searchBox.getText().toLowerCase().trim();

        toolbarContent.getChildren().clear();

        boolean hasMatches = false;

        for (String category : FXNodeRegistry.getCategories()) {
            boolean categoryAdded = false;
            for (FXNodeRegistry.NodeType nodeType : FXNodeRegistry.getNodesInCategory(category)) {
                if (filter.isEmpty() || nodeType.displayName.toLowerCase().contains(filter)
                        || nodeType.getButtonName().toLowerCase().contains(filter)) {
                    if (!categoryAdded) {
                        addToolbarCategory(category);
                        categoryAdded = true;
                        hasMatches = true;
                    }
                    final String typeName = nodeType.name;
                    addToolbarButton(nodeType.getButtonName(), () -> addNode(typeName));
                }
            }
            if (category.equals("Utility")) {
                if (filter.isEmpty() || "connector".contains(filter) || "queue".contains(filter)) {
                    if (!categoryAdded) {
                        addToolbarCategory(category);
                        hasMatches = true;
                    }
                    addToolbarButton("Connector/Queue", this::addConnectorQueue);
                }
            }
        }

        if (!hasMatches && !filter.isEmpty()) {
            Label noMatchLabel = new Label("No matches found\n\nTry fewer letters or\nclear your search");
            noMatchLabel.setFont(javafx.scene.text.Font.font("System", javafx.scene.text.FontPosture.ITALIC, 12));
            noMatchLabel.setStyle("-fx-text-alignment: center; -fx-padding: 20 10 10 10;");
            noMatchLabel.setWrapText(true);
            noMatchLabel.setMaxWidth(Double.MAX_VALUE);
            noMatchLabel.setAlignment(Pos.CENTER);
            toolbarContent.getChildren().add(noMatchLabel);
        }
    }

    private void notifyModified() {
        if (onModified != null) {
            onModified.run();
        }
    }

    // ========================= PROPERTIES DIALOG =========================

    private void showNodeProperties(FXNode node) {
        Stage stage = (ownerWindow instanceof Stage) ? (Stage) ownerWindow : null;
        FXPropertiesDialog dialog = new FXPropertiesDialog(
            stage,
            node.label + " Properties",
            node.nodeType,
            node.label
        );

        FXNodeRegistry.NodeType typeInfo = FXNodeRegistry.getNodeType(node.nodeType);
        if (typeInfo != null && typeInfo.description != null) {
            dialog.addDescription(typeInfo.description);
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

        // Add pipeline file path for container nodes
        TextField pipelineFileField = null;
        if (node.isContainer) {
            final TextField fileField = new TextField(node.pipelineFilePath != null ? node.pipelineFilePath : "");
            fileField.setPrefWidth(250);
            pipelineFileField = fileField;

            Button browseButton = new Button("Browse...");
            browseButton.setOnAction(e -> {
                javafx.stage.FileChooser fileChooser = new javafx.stage.FileChooser();
                fileChooser.setTitle("Select Pipeline File");
                fileChooser.getExtensionFilters().add(
                    new javafx.stage.FileChooser.ExtensionFilter("Pipeline Files", "*.json"));
                if (node.pipelineFilePath != null && !node.pipelineFilePath.isEmpty()) {
                    java.io.File currentFile = new java.io.File(node.pipelineFilePath);
                    if (currentFile.getParentFile() != null && currentFile.getParentFile().exists()) {
                        fileChooser.setInitialDirectory(currentFile.getParentFile());
                    }
                }
                java.io.File selectedFile = fileChooser.showOpenDialog(stage);
                if (selectedFile != null) {
                    fileField.setText(selectedFile.getAbsolutePath());
                }
            });

            HBox fileRow = new HBox(5);
            fileRow.getChildren().addAll(new Label("Pipeline File:"), fileField, browseButton);
            dialog.addCustomContent(fileRow);
            dialog.addDescription("External pipeline JSON file for this container.\nDouble-click the container to open its subdiagram.");
        }

        // Add "Queues in Sync" checkbox for dual-input nodes
        CheckBox syncCheckBox = null;
        boolean isDualInput = node.hasDualInput || isDualInputNodeType(node.nodeType);
        if (isDualInput) {
            syncCheckBox = dialog.addCheckbox("Queues in Sync", node.queuesInSync);
            dialog.addDescription("When enabled, wait for new data on both\ninputs before processing (synchronized mode).");
        }

        final CheckBox finalSyncCheckBox = syncCheckBox;
        final TextField finalPipelineFileField = pipelineFileField;
        final String oldFilePath = node.filePath;  // Track original path to detect changes
        final Runnable processorOnOk = dialog.getOnOk();  // Preserve processor's onOk callback
        dialog.setOnOk(() -> {
            // Call processor's onOk first (this syncs processor properties to node)
            if (processorOnOk != null) {
                processorOnOk.run();
            }

            node.label = dialog.getNameValue();

            if (node.isContainer && finalPipelineFileField != null) {
                String newPath = finalPipelineFileField.getText().trim();
                node.pipelineFilePath = newPath;
            }

            if (isDualInput && finalSyncCheckBox != null) {
                node.queuesInSync = finalSyncCheckBox.isSelected();
            }

            // For FileSource nodes, load the image thumbnail if path changed
            if ("FileSource".equals(node.nodeType)) {
                String newPath = node.filePath != null ? node.filePath : "";
                String oldPath = oldFilePath != null ? oldFilePath : "";
                if (!newPath.equals(oldPath)) {
                    loadFileImageForNode(node);
                }
            }

            paintCanvas();
            notifyModified();

            if (onRefreshPipeline != null && isPipelineRunning != null && isPipelineRunning.get()) {
                onRefreshPipeline.run();
            }
        });

        dialog.showAndWaitForResult();
    }

    private boolean isDualInputNodeType(String nodeType) {
        return "AddClamp".equals(nodeType) ||
               "SubtractClamp".equals(nodeType) ||
               "AddWeighted".equals(nodeType) ||
               "BitwiseAnd".equals(nodeType) ||
               "BitwiseOr".equals(nodeType) ||
               "BitwiseXor".equals(nodeType);
    }

    /**
     * Load an image file for a FileSource node and set its thumbnail.
     * Uses JavaFX Image loading with background loading support.
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
                        previewImageView.setImage(image);
                    }
                    paintCanvas();
                }
            }
        });

        // Also handle errors
        image.errorProperty().addListener((obs, oldVal, newVal) -> {
            if (newVal) {
                System.err.println("Error loading image: " + node.filePath);
                node.thumbnail = null;
                paintCanvas();
            }
        });
    }

    // ========================= NESTED CONTAINER =========================

    private String resolvePipelinePath(String pipelinePath) {
        if (pipelinePath == null || pipelinePath.isEmpty()) {
            return pipelinePath;
        }
        java.io.File file = new java.io.File(pipelinePath);
        if (file.isAbsolute()) {
            return pipelinePath;
        }
        if (basePath != null && !basePath.isEmpty()) {
            java.io.File baseDir = new java.io.File(basePath);
            if (baseDir.isFile()) {
                baseDir = baseDir.getParentFile();
            }
            if (baseDir != null) {
                java.io.File resolved = new java.io.File(baseDir, pipelinePath);
                return resolved.getAbsolutePath();
            }
        }
        return pipelinePath;
    }

    private void openNestedContainer(FXNode containerNode) {
        boolean hasExistingNodes = containerNode.innerNodes != null && !containerNode.innerNodes.isEmpty();
        if (!hasExistingNodes && containerNode.pipelineFilePath != null && !containerNode.pipelineFilePath.isEmpty()) {
            try {
                String resolvedPath = resolvePipelinePath(containerNode.pipelineFilePath);
                java.io.File pipelineFile = new java.io.File(resolvedPath);
                if (pipelineFile.exists()) {
                    FXPipelineSerializer.PipelineDocument doc = FXPipelineSerializer.load(resolvedPath);
                    if (containerNode.innerNodes == null) {
                        containerNode.innerNodes = new ArrayList<>();
                    }
                    if (containerNode.innerConnections == null) {
                        containerNode.innerConnections = new ArrayList<>();
                    }
                    containerNode.innerNodes.addAll(doc.nodes);
                    containerNode.innerConnections.addAll(doc.connections);
                } else {
                    Alert alert = new Alert(Alert.AlertType.WARNING);
                    alert.setTitle("File Not Found");
                    alert.setHeaderText("Pipeline file not found");
                    alert.setContentText("The file '" + containerNode.pipelineFilePath + "' does not exist.\n" +
                        "(Resolved to: " + resolvedPath + ")\n" +
                        "Opening with current inner nodes.");
                    alert.showAndWait();
                }
            } catch (Exception ex) {
                Alert alert = new Alert(Alert.AlertType.ERROR);
                alert.setTitle("Error Loading Pipeline");
                alert.setHeaderText("Failed to load nested pipeline");
                alert.setContentText("Error: " + ex.getMessage());
                alert.showAndWait();
                return;
            }
        }

        Stage stage = (ownerWindow instanceof Stage) ? (Stage) ownerWindow : null;
        FXContainerEditorWindow nestedEditor = new FXContainerEditorWindow(stage, containerNode, this::notifyModified);
        nestedEditor.setBasePath(basePath);

        nestedEditor.setOnStartPipeline(() -> {
            if (onStartPipeline != null) onStartPipeline.run();
            javafx.application.Platform.runLater(this::updatePipelineButtonState);
        });
        nestedEditor.setOnStopPipeline(() -> {
            if (onStopPipeline != null) onStopPipeline.run();
            javafx.application.Platform.runLater(this::updatePipelineButtonState);
        });
        nestedEditor.setOnRefreshPipeline(onRefreshPipeline);
        nestedEditor.setIsPipelineRunning(isPipelineRunning);
        nestedEditor.setGetThreadCount(getThreadCount);
        nestedEditor.setOnRequestGlobalSave(onRequestGlobalSave);
        nestedEditor.setOnQuit(onQuit);
        nestedEditor.setOnRestart(onRestart);

        // Track this window so we can close it on File New/Open
        openContainerWindows.add(nestedEditor);
        nestedEditor.setOnHidden(() -> openContainerWindows.remove(nestedEditor));

        nestedEditor.show();
        nestedEditor.updatePipelineButtonState();
    }

    // ========================= ACTIONS =========================

    private void togglePipeline() {
        if (isPipelineRunning != null && isPipelineRunning.get()) {
            if (onStopPipeline != null) onStopPipeline.run();
        } else {
            if (onStartPipeline != null) onStartPipeline.run();
        }
        updatePipelineButtonState();
    }

    private void handleSave() {
        if (onRequestGlobalSave != null) {
            onRequestGlobalSave.run();
        }
    }

    private void handleCancel() {
        // For sub-pipelines (non-root), close the window
        if (!isRootDiagram && ownerWindow instanceof Stage) {
            ((Stage) ownerWindow).close();
        }
    }

    /**
     * Update the preview pane when node selection changes.
     */
    public void updatePreviewForSelection() {
        if (selectedNodes.size() == 1) {
            FXNode node = selectedNodes.iterator().next();
            if (node.previewImage != null) {
                previewImageView.setImage(node.previewImage);
            } else if (node.thumbnail != null) {
                previewImageView.setImage(node.thumbnail);
            } else {
                previewImageView.setImage(null);
            }
        } else {
            previewImageView.setImage(null);
        }
    }
}
