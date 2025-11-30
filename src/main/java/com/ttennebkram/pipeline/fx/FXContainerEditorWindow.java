package com.ttennebkram.pipeline.fx;

import javafx.geometry.Insets;
import javafx.geometry.Orientation;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.*;
import javafx.scene.image.ImageView;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.stage.Modality;
import javafx.stage.Stage;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * JavaFX window for editing the internal pipeline of a Container node.
 * This is a non-modal window that allows adding/connecting nodes inside a container.
 */
public class FXContainerEditorWindow {

    // Color constants
    private static final Color COLOR_TOOLBAR_BG = Color.rgb(160, 200, 160);
    private static final Color COLOR_GRID_LINES = Color.rgb(230, 230, 230);
    private static final Color COLOR_SELECTION_BOX = Color.rgb(0, 0, 255);  // Bright blue - matches NodeRenderer selection
    private static final Color COLOR_SELECTION_BOX_FILL = Color.rgb(0, 0, 255, 0.1);  // Bright blue with alpha
    private static final Color COLOR_START_BUTTON = Color.rgb(100, 180, 100);
    private static final Color COLOR_STOP_BUTTON = Color.rgb(200, 100, 100);

    private Stage stage;
    private Canvas canvas;
    private ScrollPane canvasScrollPane;
    private ImageView previewImageView;
    private VBox toolbarContent;
    private TextField searchBox;
    private Label statusLabel;
    private Label pipelineStatusLabel;
    private Button startStopBtn;

    // Pipeline control callbacks
    private Runnable onStartPipeline;
    private Runnable onStopPipeline;
    private java.util.function.Supplier<Boolean> isPipelineRunning;

    // Callback to trigger global save (parent pipeline)
    private Runnable onRequestGlobalSave;

    private FXNode containerNode;
    private Stage parentStage;
    private Runnable onModified;

    // Base path for resolving relative pipeline file paths (directory of the parent document)
    private String basePath = null;

    // Reference to container's internal nodes and connections
    private List<FXNode> nodes;
    private List<FXConnection> connections;

    // Selection state
    private Set<FXNode> selectedNodes = new HashSet<>();
    private Set<FXConnection> selectedConnections = new HashSet<>();

    // Drag state
    private FXNode dragNode = null;
    private double dragOffsetX, dragOffsetY;
    private boolean isDragging = false;

    // Connection drawing state (forward: drag from output to input)
    private FXNode connectionSource = null;
    private int connectionOutputIndex = 0;
    private double connectionEndX, connectionEndY;
    private boolean isDrawingConnection = false;

    // Reverse connection drawing state (drag from input to output)
    private FXNode connectionTarget = null;
    private int connectionInputIndex = 0;
    private boolean isDrawingReverseConnection = false;

    // Selection box state
    private double selectionBoxStartX, selectionBoxStartY;
    private double selectionBoxEndX, selectionBoxEndY;
    private boolean isSelectionBoxDragging = false;

    // Zoom
    private double zoomLevel = 1.0;
    private static final int[] ZOOM_LEVELS = {25, 50, 75, 100, 125, 150, 200};

    // Repaint timer for updating thumbnails while pipeline is running
    private javafx.animation.AnimationTimer repaintTimer;

    public FXContainerEditorWindow(Stage parentStage, FXNode containerNode, Runnable onModified) {
        this.parentStage = parentStage;
        this.containerNode = containerNode;
        this.onModified = onModified;
        this.nodes = containerNode.innerNodes;
        this.connections = containerNode.innerConnections;

        // Ensure boundary nodes exist
        ensureBoundaryNodes();

        createWindow();
    }

    /**
     * Ensure the container has Input and Output boundary nodes.
     * These are automatically added if not already present.
     * Also ensures existing boundary nodes have correct flags set (for backwards compatibility).
     */
    private void ensureBoundaryNodes() {
        FXNode inputNode = null;
        FXNode outputNode = null;

        for (FXNode node : nodes) {
            if ("ContainerInput".equals(node.nodeType)) {
                inputNode = node;
                // Ensure boundary node properties are set (backwards compatibility)
                node.isBoundaryNode = true;
                node.hasInput = false;
                node.outputCount = 1;
                node.backgroundColor = NodeRenderer.COLOR_CONTAINER_NODE;
                // Fix height if it was wrong
                node.height = NodeRenderer.NODE_HEIGHT;
            } else if ("ContainerOutput".equals(node.nodeType)) {
                outputNode = node;
                // Ensure boundary node properties are set (backwards compatibility)
                node.isBoundaryNode = true;
                node.outputCount = 0;
                node.backgroundColor = NodeRenderer.COLOR_CONTAINER_NODE;
                // Fix height if it was wrong
                node.height = NodeRenderer.NODE_HEIGHT;
            }
        }

        if (inputNode == null) {
            // ContainerInput is a source node (no input, has output)
            inputNode = new FXNode("Input", "ContainerInput", 50, 50);
            inputNode.hasInput = false;  // Source node - no input
            inputNode.outputCount = 1;
            inputNode.isBoundaryNode = true;
            inputNode.backgroundColor = NodeRenderer.COLOR_CONTAINER_NODE;
            nodes.add(inputNode);
        }

        if (outputNode == null) {
            // ContainerOutput has an input but no output
            outputNode = new FXNode("Output", "ContainerOutput", 400, 50);
            outputNode.outputCount = 0;  // No outputs - this is a sink node
            outputNode.isBoundaryNode = true;
            outputNode.backgroundColor = NodeRenderer.COLOR_CONTAINER_NODE;
            nodes.add(outputNode);
        }
    }

    private void createWindow() {
        stage = new Stage();
        stage.setTitle("Container: " + containerNode.label);
        // Don't set owner - allows windows to move independently
        // Non-modal so users can switch between windows
        stage.initModality(Modality.NONE);

        BorderPane root = new BorderPane();

        // Left - toolbar with node palette
        root.setLeft(createToolbar());

        // Center - canvas (plus status bar) and preview
        // Put status bar under canvas only, not under preview
        VBox canvasWithStatus = new VBox();
        canvasWithStatus.getChildren().addAll(createCanvasPane(), createStatusBar());
        VBox.setVgrow(canvasWithStatus.getChildren().get(0), javafx.scene.layout.Priority.ALWAYS);

        SplitPane splitPane = new SplitPane();
        splitPane.setOrientation(Orientation.HORIZONTAL);
        splitPane.getItems().addAll(canvasWithStatus, createPreviewPane());
        splitPane.setDividerPositions(0.75);
        root.setCenter(splitPane);

        // Use parent window size with offset
        double windowWidth = parentStage.getWidth();
        double windowHeight = parentStage.getHeight();
        Scene scene = new Scene(root, windowWidth, windowHeight);

        // Keyboard handler
        scene.setOnKeyPressed(e -> {
            if (e.getCode() == javafx.scene.input.KeyCode.DELETE ||
                e.getCode() == javafx.scene.input.KeyCode.BACK_SPACE) {
                deleteSelected();
                e.consume();
            }
        });

        stage.setScene(scene);

        // Position with 40,40 offset from parent
        stage.setX(parentStage.getX() + 40);
        stage.setY(parentStage.getY() + 40);

        // Create repaint timer for updating thumbnails while pipeline is running
        // Use AnimationTimer to repaint at ~10 fps when pipeline is running
        repaintTimer = new javafx.animation.AnimationTimer() {
            private long lastUpdate = 0;
            private static final long REPAINT_INTERVAL_NS = 100_000_000L; // 100ms = 10fps

            @Override
            public void handle(long now) {
                if (now - lastUpdate >= REPAINT_INTERVAL_NS) {
                    if (isPipelineRunning != null && isPipelineRunning.get()) {
                        paintCanvas();
                    }
                    lastUpdate = now;
                }
            }
        };
        repaintTimer.start();

        // Stop repaint timer when window is closed
        stage.setOnCloseRequest(e -> {
            if (repaintTimer != null) {
                repaintTimer.stop();
            }
        });

        paintCanvas();
    }

    public void show() {
        stage.show();
        stage.toFront();
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
        searchBox.setStyle("-fx-padding: 2 22 2 5;");

        Button clearSearchBtn = new Button("\u00D7"); // Unicode multiplication sign
        clearSearchBtn.setStyle("-fx-font-size: 12px; -fx-padding: 0 5 0 5; -fx-background-color: transparent; -fx-cursor: hand;");
        clearSearchBtn.setOnAction(e -> searchBox.clear());
        // Only show when search box has text
        clearSearchBtn.visibleProperty().bind(searchBox.textProperty().isNotEmpty());

        StackPane searchStack = new StackPane();
        searchStack.getChildren().addAll(searchBox, clearSearchBtn);
        StackPane.setAlignment(clearSearchBtn, javafx.geometry.Pos.CENTER_RIGHT);
        StackPane.setMargin(clearSearchBtn, new Insets(0, 3, 0, 0));
        toolbar.getChildren().add(searchStack);

        // Scrollable node buttons
        ScrollPane scrollPane = new ScrollPane();
        scrollPane.setFitToWidth(true);
        scrollPane.setStyle("-fx-background: rgb(160, 200, 160); -fx-background-color: rgb(160, 200, 160);");
        VBox.setVgrow(scrollPane, Priority.ALWAYS);

        toolbarContent = new VBox(0);
        toolbarContent.setPadding(new Insets(4));
        toolbarContent.setStyle("-fx-background-color: rgb(160, 200, 160);");
        scrollPane.setContent(toolbarContent);

        // Populate with all node types
        // Container I/O nodes are NOT shown in toolbar - they are auto-created via ensureBoundaryNodes()
        for (String category : FXNodeRegistry.getCategoriesExcluding("Container I/O")) {
            boolean hasNodes = false;
            for (FXNodeRegistry.NodeType nodeType : FXNodeRegistry.getNodesInCategory(category)) {
                if (!hasNodes) {
                    addToolbarCategory(category);
                    hasNodes = true;
                }
                final String typeName = nodeType.name;
                addToolbarButton(nodeType.getButtonName(), () -> addNode(typeName));
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

    private void addToolbarButton(String text, Runnable action) {
        Button btn = new Button(text);
        btn.setMaxWidth(Double.MAX_VALUE);
        btn.setOnAction(e -> action.run());
        toolbarContent.getChildren().add(btn);
    }

    private ScrollPane createCanvasPane() {
        canvas = new Canvas(2000, 2000);
        setupCanvasEvents();

        Pane canvasContainer = new Pane(canvas);
        canvasContainer.setStyle("-fx-background-color: white;");

        canvasScrollPane = new ScrollPane(canvasContainer);
        canvasScrollPane.setPannable(false);  // Disable panning - we handle canvas interactions ourselves

        return canvasScrollPane;
    }

    private VBox createPreviewPane() {
        VBox previewPane = new VBox(5);
        previewPane.setPadding(new Insets(10));
        previewPane.setStyle("-fx-background-color: #f0f0f0;");

        // Save and Cancel buttons at top
        HBox buttonPanel = new HBox(5);
        buttonPanel.setMaxWidth(Double.MAX_VALUE);

        Button saveButton = new Button("Save");
        saveButton.setMaxWidth(Double.MAX_VALUE);
        HBox.setHgrow(saveButton, Priority.ALWAYS);
        saveButton.setOnAction(e -> {
            if (onRequestGlobalSave != null) {
                onRequestGlobalSave.run();
            }
        });

        Button cancelButton = new Button("Cancel");
        cancelButton.setMaxWidth(Double.MAX_VALUE);
        HBox.setHgrow(cancelButton, Priority.ALWAYS);
        cancelButton.setOnAction(e -> stage.close());

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
        previewImageView.setFitWidth(200);
        previewImageView.setFitHeight(200);

        StackPane imageContainer = new StackPane(previewImageView);
        imageContainer.setStyle("-fx-background-color: #cccccc; -fx-min-height: 150;");
        VBox.setVgrow(imageContainer, Priority.ALWAYS);

        previewPane.getChildren().addAll(buttonPanel, startStopBtn, instructionsLabel, instructions, previewLabel, imageContainer);

        return previewPane;
    }

    private void togglePipeline() {
        if (onStartPipeline != null && onStopPipeline != null && isPipelineRunning != null) {
            if (isPipelineRunning.get()) {
                onStopPipeline.run();
            } else {
                onStartPipeline.run();
            }
            // Update button state after toggling - use Platform.runLater to ensure state is updated
            javafx.application.Platform.runLater(this::updatePipelineButtonState);
        }
    }

    public void updatePipelineButtonState() {
        if (startStopBtn != null && isPipelineRunning != null) {
            boolean running = isPipelineRunning.get();
            if (running) {
                startStopBtn.setText("Stop Pipeline");
                startStopBtn.setStyle("-fx-background-color: rgb(200, 100, 100); -fx-font-weight: bold;");
            } else {
                startStopBtn.setText("Start Pipeline");
                startStopBtn.setStyle("-fx-background-color: rgb(100, 180, 100); -fx-font-weight: bold;");
            }
            // Update status bar label
            if (pipelineStatusLabel != null) {
                int threadCount = Thread.activeCount();
                if (running) {
                    pipelineStatusLabel.setText("Pipeline running | Threads: " + threadCount);
                    pipelineStatusLabel.setTextFill(javafx.scene.paint.Color.rgb(0, 128, 0));
                } else {
                    pipelineStatusLabel.setText("Pipeline stopped");
                    pipelineStatusLabel.setTextFill(javafx.scene.paint.Color.rgb(180, 0, 0));
                }
            }
        }
    }

    public void setOnStartPipeline(Runnable callback) {
        this.onStartPipeline = callback;
    }

    public void setOnStopPipeline(Runnable callback) {
        this.onStopPipeline = callback;
    }

    public void setIsPipelineRunning(java.util.function.Supplier<Boolean> supplier) {
        this.isPipelineRunning = supplier;
    }

    public void setOnRequestGlobalSave(Runnable callback) {
        this.onRequestGlobalSave = callback;
    }

    /**
     * Set the base path for resolving relative pipeline file paths.
     * This should be the directory containing the parent pipeline document.
     */
    public void setBasePath(String basePath) {
        this.basePath = basePath;
    }

    private HBox createStatusBar() {
        HBox statusBar = new HBox(10);
        statusBar.setPadding(new Insets(5, 10, 5, 10));
        statusBar.setStyle("-fx-background-color: rgb(160, 160, 160);");
        statusBar.setAlignment(Pos.CENTER_LEFT);

        // Node count (left side)
        statusLabel = new Label("Nodes: " + nodes.size());

        Region spacer1 = new Region();
        HBox.setHgrow(spacer1, Priority.ALWAYS);

        // Pipeline status (center)
        pipelineStatusLabel = new Label("Pipeline stopped");
        pipelineStatusLabel.setTextFill(javafx.scene.paint.Color.rgb(180, 0, 0));
        pipelineStatusLabel.setStyle("-fx-font-weight: bold;");

        Region spacer2 = new Region();
        HBox.setHgrow(spacer2, Priority.ALWAYS);

        // Zoom controls (right side)
        Label zoomLabel = new Label("Zoom:");
        ComboBox<String> zoomCombo = new ComboBox<>();
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

        statusBar.getChildren().addAll(statusLabel, spacer1, pipelineStatusLabel, spacer2, zoomLabel, zoomCombo);

        return statusBar;
    }

    private void setupCanvasEvents() {
        canvas.setOnMousePressed(this::handleMousePressed);
        canvas.setOnMouseDragged(this::handleMouseDragged);
        canvas.setOnMouseReleased(this::handleMouseReleased);
        canvas.setOnMouseClicked(e -> {
            if (e.getClickCount() == 2) {
                handleDoubleClick(e);
            }
        });

        // Right-click context menu
        ContextMenu contextMenu = new ContextMenu();
        canvas.setOnContextMenuRequested(e -> {
            contextMenu.getItems().clear();
            double canvasX = e.getX() / zoomLevel;
            double canvasY = e.getY() / zoomLevel;

            FXNode node = getNodeAt(canvasX, canvasY);
            if (node != null) {
                MenuItem propertiesItem = new MenuItem("Properties...");
                propertiesItem.setOnAction(ae -> showNodeProperties(node));
                contextMenu.getItems().add(propertiesItem);

                if (node.isContainer) {
                    MenuItem openContainerItem = new MenuItem("Open Subdiagram");
                    openContainerItem.setOnAction(ae -> openNestedContainer(node));
                    contextMenu.getItems().add(openContainerItem);
                }

                contextMenu.getItems().add(new SeparatorMenuItem());

                MenuItem deleteItem = new MenuItem("Delete");
                deleteItem.setOnAction(ae -> {
                    selectedNodes.clear();
                    selectedNodes.add(node);
                    deleteSelected();
                });
                contextMenu.getItems().add(deleteItem);

                contextMenu.show(canvas, e.getScreenX(), e.getScreenY());
            }
        });
    }

    private void handleMousePressed(javafx.scene.input.MouseEvent e) {
        double canvasX = e.getX() / zoomLevel;
        double canvasY = e.getY() / zoomLevel;

        // First, check ALL nodes for connection point clicks (connection points are on edges,
        // so the click might not be "inside" the node bounds)
        double tolerance = NodeRenderer.CONNECTION_RADIUS + 9;
        for (FXNode node : nodes) {
            int outputIdx = node.getOutputPointAt(canvasX, canvasY, tolerance);
            if (outputIdx >= 0) {
                connectionSource = node;
                connectionOutputIndex = outputIdx;
                connectionEndX = canvasX;
                connectionEndY = canvasY;
                isDrawingConnection = true;
                return;
            }

            int inputIdx = node.getInputPointAt(canvasX, canvasY, tolerance);
            if (inputIdx >= 0) {
                // For input clicks, we start a "reverse" connection - user will drag to an output
                connectionTarget = node;
                connectionInputIndex = inputIdx;
                connectionEndX = canvasX;
                connectionEndY = canvasY;
                isDrawingReverseConnection = true;
                return;
            }
        }

        FXNode clickedNode = getNodeAt(canvasX, canvasY);

        if (clickedNode != null) {
            // Check if clicking on checkbox (not for boundary nodes - they can't be disabled)
            if (!clickedNode.isBoundaryNode && clickedNode.isOnCheckbox(canvasX, canvasY)) {
                clickedNode.enabled = !clickedNode.enabled;
                paintCanvas();
                notifyModified();
                return;
            }

            // Start drag
            if (!selectedNodes.contains(clickedNode)) {
                selectedNodes.clear();
                selectedConnections.clear();
                selectedNodes.add(clickedNode);
            }
            dragNode = clickedNode;
            dragOffsetX = canvasX - clickedNode.x;
            dragOffsetY = canvasY - clickedNode.y;
            isDragging = false;
            paintCanvas();
        } else {
            // Check if clicking on a connection
            FXConnection clickedConn = getConnectionAt(canvasX, canvasY, 8);
            if (clickedConn != null) {
                if (!e.isShiftDown()) {
                    selectedNodes.clear();
                    selectedConnections.clear();
                }
                if (selectedConnections.contains(clickedConn)) {
                    selectedConnections.remove(clickedConn);
                } else {
                    selectedConnections.add(clickedConn);
                }
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
    }

    private void handleMouseDragged(javafx.scene.input.MouseEvent e) {
        double canvasX = e.getX() / zoomLevel;
        double canvasY = e.getY() / zoomLevel;

        if (isDrawingConnection || isDrawingReverseConnection) {
            connectionEndX = canvasX;
            connectionEndY = canvasY;
            paintCanvas();
        } else if (dragNode != null) {
            isDragging = true;
            for (FXNode node : selectedNodes) {
                if (node == dragNode) {
                    node.x = canvasX - dragOffsetX;
                    node.y = canvasY - dragOffsetY;
                }
            }
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

        if (isDrawingConnection && connectionSource != null) {
            // Check ALL nodes for input point hit (connection points are on edges)
            double tolerance = NodeRenderer.CONNECTION_RADIUS + 9;
            boolean connected = false;
            for (FXNode targetNode : nodes) {
                if (targetNode != connectionSource) {
                    int inputIdx = targetNode.getInputPointAt(canvasX, canvasY, tolerance);
                    if (inputIdx >= 0) {
                        // Create connection
                        FXConnection conn = new FXConnection(connectionSource, connectionOutputIndex, targetNode, inputIdx);
                        connections.add(conn);
                        notifyModified();
                        connected = true;
                        break;
                    }
                }
            }
            isDrawingConnection = false;
            connectionSource = null;
            paintCanvas();
        }

        if (isDrawingReverseConnection && connectionTarget != null) {
            // Check ALL nodes for output point hit (connection points are on edges)
            double tolerance = NodeRenderer.CONNECTION_RADIUS + 9;
            boolean connected = false;
            for (FXNode sourceNode : nodes) {
                if (sourceNode != connectionTarget) {
                    int outputIdx = sourceNode.getOutputPointAt(canvasX, canvasY, tolerance);
                    if (outputIdx >= 0) {
                        // Create connection (note: reversed from target to source)
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
            // Select all nodes and connections within the selection box
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

            // Also select connections that pass through the selection box
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

    private void handleDoubleClick(javafx.scene.input.MouseEvent e) {
        double canvasX = e.getX() / zoomLevel;
        double canvasY = e.getY() / zoomLevel;

        FXNode node = getNodeAt(canvasX, canvasY);
        if (node != null) {
            // If it's a container node, open its subdiagram instead of showing properties
            if (node.isContainer) {
                openNestedContainer(node);
            } else {
                showNodeProperties(node);
            }
        }
    }

    /**
     * Resolve a pipeline file path, handling both absolute and relative paths.
     * Relative paths are resolved against the basePath (parent document directory).
     */
    private String resolvePipelinePath(String pipelinePath) {
        if (pipelinePath == null || pipelinePath.isEmpty()) {
            return pipelinePath;
        }
        java.io.File file = new java.io.File(pipelinePath);
        // If it's already absolute and exists, use it as-is
        if (file.isAbsolute()) {
            return pipelinePath;
        }
        // Relative path - resolve against basePath
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
        // No basePath available, return as-is
        return pipelinePath;
    }

    /**
     * Open a nested container's subdiagram in a new editor window.
     */
    private void openNestedContainer(FXNode containerNode) {
        // If there's a pipeline file path set AND the container has no inner nodes yet,
        // load from that file. If the container already has inner nodes (e.g., set up by
        // the executor for a running pipeline), we use those to preserve node identity
        // so that thumbnails and stats update correctly.
        boolean hasExistingNodes = containerNode.innerNodes != null && !containerNode.innerNodes.isEmpty();
        if (!hasExistingNodes && containerNode.pipelineFilePath != null && !containerNode.pipelineFilePath.isEmpty()) {
            try {
                // Resolve the path (handle relative paths)
                String resolvedPath = resolvePipelinePath(containerNode.pipelineFilePath);
                java.io.File pipelineFile = new java.io.File(resolvedPath);
                if (pipelineFile.exists()) {
                    FXPipelineSerializer.PipelineDocument doc = FXPipelineSerializer.load(resolvedPath);
                    // Initialize the container's inner nodes and connections from the file
                    if (containerNode.innerNodes == null) {
                        containerNode.innerNodes = new java.util.ArrayList<>();
                    }
                    if (containerNode.innerConnections == null) {
                        containerNode.innerConnections = new java.util.ArrayList<>();
                    }
                    containerNode.innerNodes.addAll(doc.nodes);
                    containerNode.innerConnections.addAll(doc.connections);
                } else {
                    // File doesn't exist - show warning and continue with existing inner nodes
                    javafx.scene.control.Alert alert = new javafx.scene.control.Alert(
                        javafx.scene.control.Alert.AlertType.WARNING);
                    alert.setTitle("File Not Found");
                    alert.setHeaderText("Pipeline file not found");
                    alert.setContentText("The file '" + containerNode.pipelineFilePath + "' does not exist.\n" +
                        "(Resolved to: " + resolvedPath + ")\n" +
                        "Opening with current inner nodes.");
                    alert.showAndWait();
                }
            } catch (Exception ex) {
                javafx.scene.control.Alert alert = new javafx.scene.control.Alert(
                    javafx.scene.control.Alert.AlertType.ERROR);
                alert.setTitle("Error Loading Pipeline");
                alert.setHeaderText("Failed to load nested pipeline");
                alert.setContentText("Error: " + ex.getMessage());
                alert.showAndWait();
                return;
            }
        }

        // Open a new container editor window for the nested container
        FXContainerEditorWindow nestedEditor = new FXContainerEditorWindow(stage, containerNode, () -> {
            // When nested container is modified, mark this container as modified too
            notifyModified();
        });
        // Pass along the basePath for nested containers
        nestedEditor.setBasePath(basePath);

        // Copy pipeline control callbacks to nested editor
        // Wrap to also update parent window's button state
        nestedEditor.setOnStartPipeline(() -> {
            if (onStartPipeline != null) onStartPipeline.run();
            javafx.application.Platform.runLater(this::updatePipelineButtonState);
        });
        nestedEditor.setOnStopPipeline(() -> {
            if (onStopPipeline != null) onStopPipeline.run();
            javafx.application.Platform.runLater(this::updatePipelineButtonState);
        });
        nestedEditor.setIsPipelineRunning(isPipelineRunning);
        nestedEditor.setOnRequestGlobalSave(onRequestGlobalSave);

        nestedEditor.show();
        nestedEditor.updatePipelineButtonState();
    }

    private void showNodeProperties(FXNode node) {
        FXPropertiesDialog dialog = new FXPropertiesDialog(
            stage,
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
        FXNodePropertiesHelper.addPropertiesForNode(dialog, node);

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
                // Set initial directory based on current path
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
        } else if ("Gain".equals(node.nodeType)) {
            // Get current gain value from properties (default 1.0)
            double currentGain = 1.0;
            if (node.properties.containsKey("gain")) {
                currentGain = ((Number) node.properties.get("gain")).doubleValue();
            }

            // Create logarithmic gain slider (0.05x to 20x)
            final double LOG_RANGE = Math.log10(20.0);  // ~1.301
            double sliderVal = (Math.log10(currentGain) / LOG_RANGE) * 50 + 50;
            sliderVal = Math.max(0, Math.min(100, sliderVal));

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
                if (v % 2 == 0) v++;
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
            dialog.addDescription("Bit Planes Grayscale: Select and adjust bit planes");

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

            CheckBox[] checkBoxes = new CheckBox[8];
            Slider[] gainSliders = new Slider[8];

            HBox headerRow = new HBox(10);
            Label bitHeader = new Label("Bit");
            bitHeader.setMinWidth(30);
            Label onHeader = new Label("On");
            onHeader.setMinWidth(30);
            Label gainHeader = new Label("Gain (0.1x - 10x)");
            headerRow.getChildren().addAll(bitHeader, onHeader, gainHeader);
            dialog.addCustomContent(headerRow);

            for (int i = 0; i < 8; i++) {
                int bitNum = 7 - i;
                final int idx = i;

                HBox row = new HBox(10);
                row.setAlignment(Pos.CENTER_LEFT);

                Label bitLabel = new Label(String.valueOf(bitNum));
                bitLabel.setMinWidth(30);

                checkBoxes[i] = new CheckBox();
                checkBoxes[i].setSelected(bitEnabled[i]);

                gainSliders[i] = new Slider(0, 200, Math.log10(bitGain[i]) * 100 + 100);
                gainSliders[i].setPrefWidth(180);

                Label gainLabel = new Label(String.format("%.2fx", bitGain[i]));
                gainLabel.setMinWidth(50);

                gainSliders[i].valueProperty().addListener((obs, oldVal, newVal) -> {
                    double g = Math.pow(10, (newVal.doubleValue() - 100) / 100.0);
                    gainLabel.setText(String.format("%.2fx", g));
                });

                row.getChildren().addAll(bitLabel, checkBoxes[i], gainSliders[i], gainLabel);
                dialog.addCustomContent(row);
            }

            node.properties.put("_bitCheckBoxes", checkBoxes);
            node.properties.put("_bitGainSliders", gainSliders);
        } else if ("BitPlanesColor".equals(node.nodeType)) {
            dialog.addDescription("Bit Planes Color: Select and adjust RGB bit planes");

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

            TabPane tabPane = new TabPane();
            tabPane.setTabClosingPolicy(TabPane.TabClosingPolicy.UNAVAILABLE);

            CheckBox[][] checkBoxes = new CheckBox[3][8];
            Slider[][] gainSliders = new Slider[3][8];

            String[] tabNames = {"Red", "Green", "Blue"};
            for (int c = 0; c < 3; c++) {
                Tab tab = new Tab(tabNames[c]);
                VBox tabContent = new VBox(5);
                tabContent.setPadding(new Insets(10));

                HBox headerRow = new HBox(10);
                Label bitHeader = new Label("Bit");
                bitHeader.setMinWidth(30);
                Label onHeader = new Label("On");
                onHeader.setMinWidth(30);
                Label gainHeader = new Label("Gain (0.1x - 10x)");
                headerRow.getChildren().addAll(bitHeader, onHeader, gainHeader);
                tabContent.getChildren().add(headerRow);

                for (int i = 0; i < 8; i++) {
                    int bitNum = 7 - i;
                    final int channel = c;
                    final int idx = i;

                    HBox row = new HBox(10);
                    row.setAlignment(Pos.CENTER_LEFT);

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

            node.properties.put("_colorBitCheckBoxes", checkBoxes);
            node.properties.put("_colorBitGainSliders", gainSliders);
        } else if (isFFTNodeType(node.nodeType)) {
            // FFT Low-Pass / High-Pass filter properties
            int currentRadius = 100;
            int currentSmoothness = 0;
            if (node.properties.containsKey("radius")) {
                currentRadius = ((Number) node.properties.get("radius")).intValue();
            }
            if (node.properties.containsKey("smoothness")) {
                currentSmoothness = ((Number) node.properties.get("smoothness")).intValue();
            }

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

            node.properties.put("_radiusSlider", radiusSlider);
            node.properties.put("_smoothnessSlider", smoothnessSlider);
        } else if ("AdaptiveThreshold".equals(node.nodeType)) {
            // AdaptiveThreshold properties
            int maxValue = node.properties.containsKey("maxValue") ?
                ((Number) node.properties.get("maxValue")).intValue() : 255;
            int methodIndex = node.properties.containsKey("methodIndex") ?
                ((Number) node.properties.get("methodIndex")).intValue() : 0;
            int typeIndex = node.properties.containsKey("typeIndex") ?
                ((Number) node.properties.get("typeIndex")).intValue() : 0;
            int blockSize = node.properties.containsKey("blockSize") ?
                ((Number) node.properties.get("blockSize")).intValue() : 11;
            double cValue = node.properties.containsKey("cValue") ?
                ((Number) node.properties.get("cValue")).doubleValue() : 2.0;

            Slider maxValueSlider = dialog.addSlider("Max Value:", 0, 255, maxValue, "%.0f");
            String[] methods = {"Mean", "Gaussian"};
            ComboBox<String> methodCombo = dialog.addComboBox("Method:", methods, methods[methodIndex]);
            String[] types = {"Binary", "Binary Inv"};
            ComboBox<String> typeCombo = dialog.addComboBox("Type:", types, types[typeIndex]);
            Spinner<Integer> blockSizeSpinner = dialog.addSpinner("Block Size:", 3, 99, blockSize);
            Slider cSlider = dialog.addSlider("C Value:", -20, 20, cValue, "%.1f");

            node.properties.put("_maxValueSlider", maxValueSlider);
            node.properties.put("_methodCombo", methodCombo);
            node.properties.put("_typeCombo", typeCombo);
            node.properties.put("_blockSizeSpinner", blockSizeSpinner);
            node.properties.put("_cSlider", cSlider);
        } else if ("AddWeighted".equals(node.nodeType)) {
            // AddWeighted properties (alpha, beta, gamma)
            double alpha = node.properties.containsKey("alpha") ?
                ((Number) node.properties.get("alpha")).doubleValue() : 0.5;
            double beta = node.properties.containsKey("beta") ?
                ((Number) node.properties.get("beta")).doubleValue() : 0.5;
            double gamma = node.properties.containsKey("gamma") ?
                ((Number) node.properties.get("gamma")).doubleValue() : 0.0;

            Slider alphaSlider = dialog.addSlider("Alpha:", 0, 2, alpha, "%.2f");
            Slider betaSlider = dialog.addSlider("Beta:", 0, 2, beta, "%.2f");
            Slider gammaSlider = dialog.addSlider("Gamma:", -100, 100, gamma, "%.1f");

            node.properties.put("_alphaSlider", alphaSlider);
            node.properties.put("_betaSlider", betaSlider);
            node.properties.put("_gammaSlider", gammaSlider);
        } else if ("BilateralFilter".equals(node.nodeType)) {
            // BilateralFilter properties
            int diameter = node.properties.containsKey("diameter") ?
                ((Number) node.properties.get("diameter")).intValue() : 9;
            double sigmaColor = node.properties.containsKey("sigmaColor") ?
                ((Number) node.properties.get("sigmaColor")).doubleValue() : 75.0;
            double sigmaSpace = node.properties.containsKey("sigmaSpace") ?
                ((Number) node.properties.get("sigmaSpace")).doubleValue() : 75.0;

            Spinner<Integer> diameterSpinner = dialog.addSpinner("Diameter:", 1, 25, diameter);
            Slider sigmaColorSlider = dialog.addSlider("Sigma Color:", 0, 200, sigmaColor, "%.0f");
            Slider sigmaSpaceSlider = dialog.addSlider("Sigma Space:", 0, 200, sigmaSpace, "%.0f");

            node.properties.put("_diameterSpinner", diameterSpinner);
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

            Slider clipLimitSlider = dialog.addSlider("Clip Limit:", 0.1, 40.0, clipLimit, "%.1f");
            Spinner<Integer> tileSizeSpinner = dialog.addSpinner("Tile Size:", 1, 32, tileSize);
            String[] colorModes = {"Grayscale", "LAB (L channel)", "HSV (V channel)"};
            ComboBox<String> colorModeCombo = dialog.addComboBox("Color Mode:", colorModes, colorModes[colorModeIndex]);

            node.properties.put("_clipLimitSlider", clipLimitSlider);
            node.properties.put("_tileSizeSpinner", tileSizeSpinner);
            node.properties.put("_colorModeCombo", colorModeCombo);
        } else if ("ColorInRange".equals(node.nodeType)) {
            // ColorInRange properties
            boolean useHSV = node.properties.containsKey("useHSV") ?
                (Boolean) node.properties.get("useHSV") : true;
            int hLow = node.properties.containsKey("hLow") ?
                ((Number) node.properties.get("hLow")).intValue() : 0;
            int hHigh = node.properties.containsKey("hHigh") ?
                ((Number) node.properties.get("hHigh")).intValue() : 180;
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

            CheckBox useHSVCheck = dialog.addCheckbox("Use HSV Color Space", useHSV);
            dialog.addDescription("H: 0-180 (Hue), S: 0-255 (Saturation), V: 0-255 (Value)");
            Slider hLowSlider = dialog.addSlider("H Low:", 0, 180, hLow, "%.0f");
            Slider hHighSlider = dialog.addSlider("H High:", 0, 180, hHigh, "%.0f");
            Slider sLowSlider = dialog.addSlider("S Low:", 0, 255, sLow, "%.0f");
            Slider sHighSlider = dialog.addSlider("S High:", 0, 255, sHigh, "%.0f");
            Slider vLowSlider = dialog.addSlider("V Low:", 0, 255, vLow, "%.0f");
            Slider vHighSlider = dialog.addSlider("V High:", 0, 255, vHigh, "%.0f");
            String[] outputModes = {"Mask Only", "Masked Color"};
            ComboBox<String> outputModeCombo = dialog.addComboBox("Output:", outputModes, outputModes[outputMode]);

            node.properties.put("_useHSVCheck", useHSVCheck);
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

            Spinner<Integer> xSpinner = dialog.addSpinner("X:", 0, 4096, cropX);
            Spinner<Integer> ySpinner = dialog.addSpinner("Y:", 0, 4096, cropY);
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
                ((Number) node.properties.get("directionIndex")).intValue() : 0;
            int scalePercent = node.properties.containsKey("scalePercent") ?
                ((Number) node.properties.get("scalePercent")).intValue() : 100;
            int delta = node.properties.containsKey("delta") ?
                ((Number) node.properties.get("delta")).intValue() : 0;

            String[] directions = {"X (Horizontal edges)", "Y (Vertical edges)"};
            ComboBox<String> directionCombo = dialog.addComboBox("Direction:", directions, directions[directionIndex]);
            Slider scaleSlider = dialog.addSlider("Scale %:", 1, 500, scalePercent, "%.0f%%");
            Slider deltaSlider = dialog.addSlider("Delta:", -128, 128, delta, "%.0f");

            node.properties.put("_directionCombo", directionCombo);
            node.properties.put("_scaleSlider", scaleSlider);
            node.properties.put("_deltaSlider", deltaSlider);
        } else if ("Laplacian".equals(node.nodeType)) {
            // Laplacian properties
            int kernelSizeIndex = node.properties.containsKey("kernelSizeIndex") ?
                ((Number) node.properties.get("kernelSizeIndex")).intValue() : 0;
            int scalePercent = node.properties.containsKey("scalePercent") ?
                ((Number) node.properties.get("scalePercent")).intValue() : 100;
            int delta = node.properties.containsKey("delta") ?
                ((Number) node.properties.get("delta")).intValue() : 0;
            boolean useAbsolute = node.properties.containsKey("useAbsolute") ?
                (Boolean) node.properties.get("useAbsolute") : true;

            String[] kernelSizes = {"1", "3", "5", "7"};
            ComboBox<String> kernelSizeCombo = dialog.addComboBox("Kernel Size:", kernelSizes, kernelSizes[kernelSizeIndex]);
            Slider scaleSlider = dialog.addSlider("Scale %:", 1, 500, scalePercent, "%.0f%%");
            Slider deltaSlider = dialog.addSlider("Delta:", -128, 128, delta, "%.0f");
            CheckBox absCheck = dialog.addCheckbox("Use Absolute Value", useAbsolute);

            node.properties.put("_kernelSizeCombo", kernelSizeCombo);
            node.properties.put("_scaleSlider", scaleSlider);
            node.properties.put("_deltaSlider", deltaSlider);
            node.properties.put("_absCheck", absCheck);
        } else if ("MorphologyEx".equals(node.nodeType)) {
            // MorphologyEx properties
            int operationIndex = node.properties.containsKey("operationIndex") ?
                ((Number) node.properties.get("operationIndex")).intValue() : 0;
            int shapeIndex = node.properties.containsKey("shapeIndex") ?
                ((Number) node.properties.get("shapeIndex")).intValue() : 0;
            int kernelWidth = node.properties.containsKey("kernelWidth") ?
                ((Number) node.properties.get("kernelWidth")).intValue() : 5;
            int kernelHeight = node.properties.containsKey("kernelHeight") ?
                ((Number) node.properties.get("kernelHeight")).intValue() : 5;
            int iterations = node.properties.containsKey("iterations") ?
                ((Number) node.properties.get("iterations")).intValue() : 1;

            String[] operations = {"Erode", "Dilate", "Open", "Close", "Gradient", "TopHat", "BlackHat"};
            ComboBox<String> opCombo = dialog.addComboBox("Operation:", operations, operations[operationIndex]);
            String[] shapes = {"Rect", "Cross", "Ellipse"};
            ComboBox<String> shapeCombo = dialog.addComboBox("Shape:", shapes, shapes[shapeIndex]);
            Spinner<Integer> widthSpinner = dialog.addSpinner("Kernel Width:", 1, 31, kernelWidth);
            Spinner<Integer> heightSpinner = dialog.addSpinner("Kernel Height:", 1, 31, kernelHeight);
            Spinner<Integer> iterSpinner = dialog.addSpinner("Iterations:", 1, 10, iterations);

            node.properties.put("_opCombo", opCombo);
            node.properties.put("_shapeCombo", shapeCombo);
            node.properties.put("_kernelWidthSpinner", widthSpinner);
            node.properties.put("_kernelHeightSpinner", heightSpinner);
            node.properties.put("_iterSpinner", iterSpinner);
        } else if ("HoughCircles".equals(node.nodeType)) {
            // HoughCircles properties
            boolean showOriginal = node.properties.containsKey("showOriginal") ?
                (Boolean) node.properties.get("showOriginal") : true;
            int minDist = node.properties.containsKey("minDist") ?
                ((Number) node.properties.get("minDist")).intValue() : 50;
            int param1 = node.properties.containsKey("param1") ?
                ((Number) node.properties.get("param1")).intValue() : 100;
            int param2 = node.properties.containsKey("param2") ?
                ((Number) node.properties.get("param2")).intValue() : 30;
            int minRadius = node.properties.containsKey("minRadius") ?
                ((Number) node.properties.get("minRadius")).intValue() : 10;
            int maxRadius = node.properties.containsKey("maxRadius") ?
                ((Number) node.properties.get("maxRadius")).intValue() : 100;
            int thickness = node.properties.containsKey("thickness") ?
                ((Number) node.properties.get("thickness")).intValue() : 2;
            String colorRGB = node.properties.containsKey("colorRGB") ?
                (String) node.properties.get("colorRGB") : "0,255,0";

            CheckBox showOrigCheck = dialog.addCheckbox("Show Original Image", showOriginal);
            Slider minDistSlider = dialog.addSlider("Min Distance:", 1, 200, minDist, "%.0f");
            Slider param1Slider = dialog.addSlider("Param1 (Canny):", 1, 300, param1, "%.0f");
            Slider param2Slider = dialog.addSlider("Param2 (Accum):", 1, 100, param2, "%.0f");
            Spinner<Integer> minRadSpinner = dialog.addSpinner("Min Radius:", 0, 500, minRadius);
            Spinner<Integer> maxRadSpinner = dialog.addSpinner("Max Radius:", 0, 500, maxRadius);
            Spinner<Integer> thickSpinner = dialog.addSpinner("Line Thickness:", 1, 10, thickness);
            TextField colorField = dialog.addTextField("Color (R,G,B):", colorRGB);

            node.properties.put("_showOrigCheck", showOrigCheck);
            node.properties.put("_minDistSlider", minDistSlider);
            node.properties.put("_param1Slider", param1Slider);
            node.properties.put("_param2Slider", param2Slider);
            node.properties.put("_minRadSpinner", minRadSpinner);
            node.properties.put("_maxRadSpinner", maxRadSpinner);
            node.properties.put("_thickSpinner", thickSpinner);
            node.properties.put("_colorField", colorField);
        } else if ("HoughLines".equals(node.nodeType)) {
            // HoughLines properties
            int threshold = node.properties.containsKey("threshold") ?
                ((Number) node.properties.get("threshold")).intValue() : 50;
            int minLineLength = node.properties.containsKey("minLineLength") ?
                ((Number) node.properties.get("minLineLength")).intValue() : 50;
            int maxLineGap = node.properties.containsKey("maxLineGap") ?
                ((Number) node.properties.get("maxLineGap")).intValue() : 10;
            int thickness = node.properties.containsKey("thickness") ?
                ((Number) node.properties.get("thickness")).intValue() : 2;
            String colorRGB = node.properties.containsKey("colorRGB") ?
                (String) node.properties.get("colorRGB") : "0,0,255";

            Slider threshSlider = dialog.addSlider("Threshold:", 1, 200, threshold, "%.0f");
            Slider minLenSlider = dialog.addSlider("Min Line Length:", 1, 200, minLineLength, "%.0f");
            Slider maxGapSlider = dialog.addSlider("Max Line Gap:", 1, 100, maxLineGap, "%.0f");
            Spinner<Integer> thickSpinner = dialog.addSpinner("Line Thickness:", 1, 10, thickness);
            TextField colorField = dialog.addTextField("Color (R,G,B):", colorRGB);

            node.properties.put("_threshSlider", threshSlider);
            node.properties.put("_minLenSlider", minLenSlider);
            node.properties.put("_maxGapSlider", maxGapSlider);
            node.properties.put("_thickSpinner", thickSpinner);
            node.properties.put("_colorField", colorField);
        } else if ("HarrisCorners".equals(node.nodeType)) {
            // HarrisCorners properties
            boolean showOriginal = node.properties.containsKey("showOriginal") ?
                (Boolean) node.properties.get("showOriginal") : true;
            int blockSize = node.properties.containsKey("blockSize") ?
                ((Number) node.properties.get("blockSize")).intValue() : 2;
            int ksize = node.properties.containsKey("ksize") ?
                ((Number) node.properties.get("ksize")).intValue() : 3;
            int kPercent = node.properties.containsKey("kPercent") ?
                ((Number) node.properties.get("kPercent")).intValue() : 4;
            int thresholdPercent = node.properties.containsKey("thresholdPercent") ?
                ((Number) node.properties.get("thresholdPercent")).intValue() : 1;
            int markerSize = node.properties.containsKey("markerSize") ?
                ((Number) node.properties.get("markerSize")).intValue() : 5;
            String colorRGB = node.properties.containsKey("colorRGB") ?
                (String) node.properties.get("colorRGB") : "255,0,0";

            CheckBox showOrigCheck = dialog.addCheckbox("Show Original Image", showOriginal);
            Spinner<Integer> blockSpinner = dialog.addSpinner("Block Size:", 2, 10, blockSize);
            Spinner<Integer> ksizeSpinner = dialog.addSpinner("Aperture Size:", 3, 31, ksize);
            Slider kSlider = dialog.addSlider("K (% / 100):", 1, 20, kPercent, "%.0f");
            Slider threshSlider = dialog.addSlider("Threshold (% of max):", 1, 50, thresholdPercent, "%.0f");
            Spinner<Integer> markerSpinner = dialog.addSpinner("Marker Size:", 1, 20, markerSize);
            TextField colorField = dialog.addTextField("Color (R,G,B):", colorRGB);

            node.properties.put("_showOrigCheck", showOrigCheck);
            node.properties.put("_blockSpinner", blockSpinner);
            node.properties.put("_ksizeSpinner", ksizeSpinner);
            node.properties.put("_kSlider", kSlider);
            node.properties.put("_threshSlider", threshSlider);
            node.properties.put("_markerSpinner", markerSpinner);
            node.properties.put("_colorField", colorField);
        } else if ("Contours".equals(node.nodeType)) {
            // Contours properties
            boolean showOriginal = node.properties.containsKey("showOriginal") ?
                (Boolean) node.properties.get("showOriginal") : false;
            int thickness = node.properties.containsKey("thickness") ?
                ((Number) node.properties.get("thickness")).intValue() : 2;
            String colorRGB = node.properties.containsKey("colorRGB") ?
                (String) node.properties.get("colorRGB") : "0,255,0";

            CheckBox showOrigCheck = dialog.addCheckbox("Show Original Image", showOriginal);
            Spinner<Integer> thickSpinner = dialog.addSpinner("Line Thickness:", 1, 10, thickness);
            TextField colorField = dialog.addTextField("Color (R,G,B):", colorRGB);

            node.properties.put("_showOrigCheck", showOrigCheck);
            node.properties.put("_thickSpinner", thickSpinner);
            node.properties.put("_colorField", colorField);
        } else if ("Rectangle".equals(node.nodeType)) {
            // Rectangle properties
            int x1 = node.properties.containsKey("x1") ?
                ((Number) node.properties.get("x1")).intValue() : 50;
            int y1 = node.properties.containsKey("y1") ?
                ((Number) node.properties.get("y1")).intValue() : 50;
            int x2 = node.properties.containsKey("x2") ?
                ((Number) node.properties.get("x2")).intValue() : 200;
            int y2 = node.properties.containsKey("y2") ?
                ((Number) node.properties.get("y2")).intValue() : 200;
            String colorRGB = node.properties.containsKey("colorRGB") ?
                (String) node.properties.get("colorRGB") : "0,255,0";
            int thickness = node.properties.containsKey("thickness") ?
                ((Number) node.properties.get("thickness")).intValue() : 2;
            boolean filled = node.properties.containsKey("filled") ?
                (Boolean) node.properties.get("filled") : false;

            Spinner<Integer> x1Spinner = dialog.addSpinner("X1:", 0, 4096, x1);
            Spinner<Integer> y1Spinner = dialog.addSpinner("Y1:", 0, 4096, y1);
            Spinner<Integer> x2Spinner = dialog.addSpinner("X2:", 0, 4096, x2);
            Spinner<Integer> y2Spinner = dialog.addSpinner("Y2:", 0, 4096, y2);
            TextField colorField = dialog.addTextField("Color (R,G,B):", colorRGB);
            Spinner<Integer> thickSpinner = dialog.addSpinner("Thickness:", 1, 50, thickness);
            CheckBox filledCheck = dialog.addCheckbox("Filled", filled);

            node.properties.put("_x1Spinner", x1Spinner);
            node.properties.put("_y1Spinner", y1Spinner);
            node.properties.put("_x2Spinner", x2Spinner);
            node.properties.put("_y2Spinner", y2Spinner);
            node.properties.put("_colorField", colorField);
            node.properties.put("_thickSpinner", thickSpinner);
            node.properties.put("_filledCheck", filledCheck);
        } else if ("Circle".equals(node.nodeType)) {
            // Circle properties
            int centerX = node.properties.containsKey("centerX") ?
                ((Number) node.properties.get("centerX")).intValue() : 100;
            int centerY = node.properties.containsKey("centerY") ?
                ((Number) node.properties.get("centerY")).intValue() : 100;
            int radius = node.properties.containsKey("radius") ?
                ((Number) node.properties.get("radius")).intValue() : 50;
            String colorRGB = node.properties.containsKey("colorRGB") ?
                (String) node.properties.get("colorRGB") : "0,255,0";
            int thickness = node.properties.containsKey("thickness") ?
                ((Number) node.properties.get("thickness")).intValue() : 2;
            boolean filled = node.properties.containsKey("filled") ?
                (Boolean) node.properties.get("filled") : false;

            Spinner<Integer> cxSpinner = dialog.addSpinner("Center X:", 0, 4096, centerX);
            Spinner<Integer> cySpinner = dialog.addSpinner("Center Y:", 0, 4096, centerY);
            Spinner<Integer> radSpinner = dialog.addSpinner("Radius:", 1, 2000, radius);
            TextField colorField = dialog.addTextField("Color (R,G,B):", colorRGB);
            Spinner<Integer> thickSpinner = dialog.addSpinner("Thickness:", 1, 50, thickness);
            CheckBox filledCheck = dialog.addCheckbox("Filled", filled);

            node.properties.put("_cxSpinner", cxSpinner);
            node.properties.put("_cySpinner", cySpinner);
            node.properties.put("_radSpinner", radSpinner);
            node.properties.put("_colorField", colorField);
            node.properties.put("_thickSpinner", thickSpinner);
            node.properties.put("_filledCheck", filledCheck);
        } else if ("Ellipse".equals(node.nodeType)) {
            // Ellipse properties
            int centerX = node.properties.containsKey("centerX") ?
                ((Number) node.properties.get("centerX")).intValue() : 100;
            int centerY = node.properties.containsKey("centerY") ?
                ((Number) node.properties.get("centerY")).intValue() : 100;
            int axisX = node.properties.containsKey("axisX") ?
                ((Number) node.properties.get("axisX")).intValue() : 60;
            int axisY = node.properties.containsKey("axisY") ?
                ((Number) node.properties.get("axisY")).intValue() : 40;
            int angle = node.properties.containsKey("angle") ?
                ((Number) node.properties.get("angle")).intValue() : 0;
            String colorRGB = node.properties.containsKey("colorRGB") ?
                (String) node.properties.get("colorRGB") : "0,255,0";
            int thickness = node.properties.containsKey("thickness") ?
                ((Number) node.properties.get("thickness")).intValue() : 2;
            boolean filled = node.properties.containsKey("filled") ?
                (Boolean) node.properties.get("filled") : false;

            Spinner<Integer> cxSpinner = dialog.addSpinner("Center X:", 0, 4096, centerX);
            Spinner<Integer> cySpinner = dialog.addSpinner("Center Y:", 0, 4096, centerY);
            Spinner<Integer> axSpinner = dialog.addSpinner("Axis X:", 1, 2000, axisX);
            Spinner<Integer> aySpinner = dialog.addSpinner("Axis Y:", 1, 2000, axisY);
            Slider angleSlider = dialog.addSlider("Angle:", 0, 360, angle, "%.0f°");
            TextField colorField = dialog.addTextField("Color (R,G,B):", colorRGB);
            Spinner<Integer> thickSpinner = dialog.addSpinner("Thickness:", 1, 50, thickness);
            CheckBox filledCheck = dialog.addCheckbox("Filled", filled);

            node.properties.put("_cxSpinner", cxSpinner);
            node.properties.put("_cySpinner", cySpinner);
            node.properties.put("_axSpinner", axSpinner);
            node.properties.put("_aySpinner", aySpinner);
            node.properties.put("_angleSlider", angleSlider);
            node.properties.put("_colorField", colorField);
            node.properties.put("_thickSpinner", thickSpinner);
            node.properties.put("_filledCheck", filledCheck);
        } else if ("Grayscale".equals(node.nodeType)) {
            // Grayscale/Color Convert properties
            int conversionIndex = node.properties.containsKey("conversionIndex") ?
                ((Number) node.properties.get("conversionIndex")).intValue() : 0;

            String[] conversions = {"BGR to Gray", "Gray to BGR", "BGR to HSV", "HSV to BGR",
                "BGR to LAB", "LAB to BGR", "BGR to YCrCb", "YCrCb to BGR"};
            ComboBox<String> convCombo = dialog.addComboBox("Conversion:", conversions, conversions[conversionIndex]);

            node.properties.put("_convCombo", convCombo);
        } else if ("FileSource".equals(node.nodeType)) {
            // FileSource properties
            String filePath = node.properties.containsKey("filePath") ?
                (String) node.properties.get("filePath") : "";
            int fpsIndex = node.properties.containsKey("fpsIndex") ?
                ((Number) node.properties.get("fpsIndex")).intValue() : 2;

            TextField fileField = dialog.addTextField("File Path:", filePath);
            String[] fpsOptions = {"1", "15", "30", "60"};
            ComboBox<String> fpsCombo = dialog.addComboBox("FPS:", fpsOptions, fpsOptions[fpsIndex]);

            // Add browse button
            Button browseBtn = new Button("Browse...");
            browseBtn.setOnAction(e -> {
                javafx.stage.FileChooser fileChooser = new javafx.stage.FileChooser();
                fileChooser.setTitle("Select Image or Video File");
                fileChooser.getExtensionFilters().addAll(
                    new javafx.stage.FileChooser.ExtensionFilter("Image/Video", "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.mp4", "*.avi", "*.mov"),
                    new javafx.stage.FileChooser.ExtensionFilter("All Files", "*.*"));
                java.io.File selected = fileChooser.showOpenDialog(stage);
                if (selected != null) {
                    fileField.setText(selected.getAbsolutePath());
                }
            });
            dialog.addCustomContent(browseBtn);

            node.properties.put("_fileField", fileField);
            node.properties.put("_fpsCombo", fpsCombo);
        } else if ("WebcamSource".equals(node.nodeType)) {
            // WebcamSource properties
            int cameraIndex = node.properties.containsKey("cameraIndex") ?
                ((Number) node.properties.get("cameraIndex")).intValue() : 0;
            int fpsIndex = node.properties.containsKey("fpsIndex") ?
                ((Number) node.properties.get("fpsIndex")).intValue() : 2;

            Spinner<Integer> camSpinner = dialog.addSpinner("Camera Index:", 0, 10, cameraIndex);
            String[] fpsOptions = {"1", "15", "30", "60"};
            ComboBox<String> fpsCombo = dialog.addComboBox("FPS:", fpsOptions, fpsOptions[fpsIndex]);

            node.properties.put("_camSpinner", camSpinner);
            node.properties.put("_fpsCombo", fpsCombo);
        }

        // Add "Queues in Sync" checkbox for dual-input nodes
        CheckBox syncCheckBox = null;
        boolean isDualInput = node.hasDualInput || isDualInputNodeType(node.nodeType);
        if (isDualInput) {
            syncCheckBox = dialog.addCheckbox("Queues in Sync", node.queuesInSync);
            dialog.addDescription("When enabled, wait for new data on both\ninputs before processing (synchronized mode).");
        }

        // Set OK handler to save values
        final CheckBox finalSyncCheckBox = syncCheckBox;
        final TextField finalPipelineFileField = pipelineFileField;
        dialog.setOnOk(() -> {
            node.label = dialog.getNameValue();

            // Save properties handled by the helper class (covers all nodes with main branch differences)
            FXNodePropertiesHelper.savePropertiesForNode(node);

            // Handle container pipeline file path
            if (node.isContainer && finalPipelineFileField != null) {
                String newPath = finalPipelineFileField.getText().trim();
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
                node.properties.remove("_gainSlider");
                node.properties.remove("_gainLogRange");
            }

            // Handle FFT filter properties
            if (isFFTNodeType(node.nodeType) && node.properties.containsKey("_radiusSlider")) {
                Slider radiusSlider = (Slider) node.properties.get("_radiusSlider");
                Slider smoothnessSlider = (Slider) node.properties.get("_smoothnessSlider");
                node.properties.put("radius", (int) radiusSlider.getValue());
                node.properties.put("smoothness", (int) smoothnessSlider.getValue());
                node.properties.remove("_radiusSlider");
                node.properties.remove("_smoothnessSlider");
            }

            // Handle MedianBlur properties
            if ("MedianBlur".equals(node.nodeType) && node.properties.containsKey("_ksizeSlider")) {
                Slider ksizeSlider = (Slider) node.properties.get("_ksizeSlider");
                int ksize = (int) ksizeSlider.getValue();
                if (ksize % 2 == 0) ksize++;
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
                @SuppressWarnings("unchecked")
                Spinner<Integer> blockSizeSpinner = (Spinner<Integer>) node.properties.get("_blockSizeSpinner");
                Slider cSlider = (Slider) node.properties.get("_cSlider");

                node.properties.put("maxValue", (int) maxValueSlider.getValue());
                String[] methods = {"Mean", "Gaussian"};
                int methodIndex = 0;
                for (int i = 0; i < methods.length; i++) {
                    if (methods[i].equals(methodCombo.getValue())) { methodIndex = i; break; }
                }
                node.properties.put("methodIndex", methodIndex);
                String[] types = {"Binary", "Binary Inv"};
                int typeIndex = 0;
                for (int i = 0; i < types.length; i++) {
                    if (types[i].equals(typeCombo.getValue())) { typeIndex = i; break; }
                }
                node.properties.put("typeIndex", typeIndex);
                int bs = blockSizeSpinner.getValue();
                if (bs % 2 == 0) bs++;
                if (bs < 3) bs = 3;
                node.properties.put("blockSize", bs);
                node.properties.put("cValue", cSlider.getValue());

                node.properties.remove("_maxValueSlider");
                node.properties.remove("_methodCombo");
                node.properties.remove("_typeCombo");
                node.properties.remove("_blockSizeSpinner");
                node.properties.remove("_cSlider");
            }

            // Handle AddWeighted properties
            if ("AddWeighted".equals(node.nodeType) && node.properties.containsKey("_alphaSlider")) {
                Slider alphaSlider = (Slider) node.properties.get("_alphaSlider");
                Slider betaSlider = (Slider) node.properties.get("_betaSlider");
                Slider gammaSlider = (Slider) node.properties.get("_gammaSlider");
                node.properties.put("alpha", alphaSlider.getValue());
                node.properties.put("beta", betaSlider.getValue());
                node.properties.put("gamma", gammaSlider.getValue());
                node.properties.remove("_alphaSlider");
                node.properties.remove("_betaSlider");
                node.properties.remove("_gammaSlider");
            }

            // Handle BilateralFilter properties
            if ("BilateralFilter".equals(node.nodeType) && node.properties.containsKey("_diameterSpinner")) {
                @SuppressWarnings("unchecked")
                Spinner<Integer> diameterSpinner = (Spinner<Integer>) node.properties.get("_diameterSpinner");
                Slider sigmaColorSlider = (Slider) node.properties.get("_sigmaColorSlider");
                Slider sigmaSpaceSlider = (Slider) node.properties.get("_sigmaSpaceSlider");
                node.properties.put("diameter", diameterSpinner.getValue());
                node.properties.put("sigmaColor", sigmaColorSlider.getValue());
                node.properties.put("sigmaSpace", sigmaSpaceSlider.getValue());
                node.properties.remove("_diameterSpinner");
                node.properties.remove("_sigmaColorSlider");
                node.properties.remove("_sigmaSpaceSlider");
            }

            // Handle CLAHE properties
            if ("CLAHE".equals(node.nodeType) && node.properties.containsKey("_clipLimitSlider")) {
                Slider clipLimitSlider = (Slider) node.properties.get("_clipLimitSlider");
                @SuppressWarnings("unchecked")
                Spinner<Integer> tileSizeSpinner = (Spinner<Integer>) node.properties.get("_tileSizeSpinner");
                @SuppressWarnings("unchecked")
                ComboBox<String> colorModeCombo = (ComboBox<String>) node.properties.get("_colorModeCombo");
                node.properties.put("clipLimit", clipLimitSlider.getValue());
                node.properties.put("tileSize", tileSizeSpinner.getValue());
                String[] colorModes = {"Grayscale", "LAB (L channel)", "HSV (V channel)"};
                int colorModeIndex = 0;
                for (int i = 0; i < colorModes.length; i++) {
                    if (colorModes[i].equals(colorModeCombo.getValue())) { colorModeIndex = i; break; }
                }
                node.properties.put("colorModeIndex", colorModeIndex);
                node.properties.remove("_clipLimitSlider");
                node.properties.remove("_tileSizeSpinner");
                node.properties.remove("_colorModeCombo");
            }

            // Handle ColorInRange properties
            if ("ColorInRange".equals(node.nodeType) && node.properties.containsKey("_useHSVCheck")) {
                CheckBox useHSVCheck = (CheckBox) node.properties.get("_useHSVCheck");
                Slider hLowSlider = (Slider) node.properties.get("_hLowSlider");
                Slider hHighSlider = (Slider) node.properties.get("_hHighSlider");
                Slider sLowSlider = (Slider) node.properties.get("_sLowSlider");
                Slider sHighSlider = (Slider) node.properties.get("_sHighSlider");
                Slider vLowSlider = (Slider) node.properties.get("_vLowSlider");
                Slider vHighSlider = (Slider) node.properties.get("_vHighSlider");
                @SuppressWarnings("unchecked")
                ComboBox<String> outputModeCombo = (ComboBox<String>) node.properties.get("_outputModeCombo");

                node.properties.put("useHSV", useHSVCheck.isSelected());
                node.properties.put("hLow", (int) hLowSlider.getValue());
                node.properties.put("hHigh", (int) hHighSlider.getValue());
                node.properties.put("sLow", (int) sLowSlider.getValue());
                node.properties.put("sHigh", (int) sHighSlider.getValue());
                node.properties.put("vLow", (int) vLowSlider.getValue());
                node.properties.put("vHigh", (int) vHighSlider.getValue());
                String[] outputModes = {"Mask Only", "Masked Color"};
                int outputMode = 0;
                for (int i = 0; i < outputModes.length; i++) {
                    if (outputModes[i].equals(outputModeCombo.getValue())) { outputMode = i; break; }
                }
                node.properties.put("outputMode", outputMode);

                node.properties.remove("_useHSVCheck");
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
                int kernelSizeIndex = 1;
                for (int i = 0; i < kernelSizes.length; i++) {
                    if (kernelSizes[i].equals(kernelSizeCombo.getValue())) { kernelSizeIndex = i; break; }
                }
                node.properties.put("kernelSizeIndex", kernelSizeIndex);
                node.properties.remove("_dxGroup");
                node.properties.remove("_dyGroup");
                node.properties.remove("_kernelSizeCombo");
            }

            // Handle Scharr properties
            if ("Scharr".equals(node.nodeType) && node.properties.containsKey("_directionCombo")) {
                @SuppressWarnings("unchecked")
                ComboBox<String> directionCombo = (ComboBox<String>) node.properties.get("_directionCombo");
                Slider scaleSlider = (Slider) node.properties.get("_scaleSlider");
                Slider deltaSlider = (Slider) node.properties.get("_deltaSlider");
                String[] directions = {"X (Horizontal edges)", "Y (Vertical edges)"};
                int directionIndex = 0;
                for (int i = 0; i < directions.length; i++) {
                    if (directions[i].equals(directionCombo.getValue())) { directionIndex = i; break; }
                }
                node.properties.put("directionIndex", directionIndex);
                node.properties.put("scalePercent", (int) scaleSlider.getValue());
                node.properties.put("delta", (int) deltaSlider.getValue());
                node.properties.remove("_directionCombo");
                node.properties.remove("_scaleSlider");
                node.properties.remove("_deltaSlider");
            }

            // Handle Laplacian properties
            if ("Laplacian".equals(node.nodeType) && node.properties.containsKey("_kernelSizeCombo")) {
                @SuppressWarnings("unchecked")
                ComboBox<String> kernelSizeCombo = (ComboBox<String>) node.properties.get("_kernelSizeCombo");
                Slider scaleSlider = (Slider) node.properties.get("_scaleSlider");
                Slider deltaSlider = (Slider) node.properties.get("_deltaSlider");
                CheckBox absCheck = (CheckBox) node.properties.get("_absCheck");
                String[] kernelSizes = {"1", "3", "5", "7"};
                int kernelSizeIndex = 0;
                for (int i = 0; i < kernelSizes.length; i++) {
                    if (kernelSizes[i].equals(kernelSizeCombo.getValue())) { kernelSizeIndex = i; break; }
                }
                node.properties.put("kernelSizeIndex", kernelSizeIndex);
                node.properties.put("scalePercent", (int) scaleSlider.getValue());
                node.properties.put("delta", (int) deltaSlider.getValue());
                node.properties.put("useAbsolute", absCheck.isSelected());
                node.properties.remove("_kernelSizeCombo");
                node.properties.remove("_scaleSlider");
                node.properties.remove("_deltaSlider");
                node.properties.remove("_absCheck");
            }

            // Handle MorphologyEx properties
            if ("MorphologyEx".equals(node.nodeType) && node.properties.containsKey("_opCombo")) {
                @SuppressWarnings("unchecked")
                ComboBox<String> opCombo = (ComboBox<String>) node.properties.get("_opCombo");
                @SuppressWarnings("unchecked")
                ComboBox<String> shapeCombo = (ComboBox<String>) node.properties.get("_shapeCombo");
                @SuppressWarnings("unchecked")
                Spinner<Integer> widthSpinner = (Spinner<Integer>) node.properties.get("_kernelWidthSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> heightSpinner = (Spinner<Integer>) node.properties.get("_kernelHeightSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> iterSpinner = (Spinner<Integer>) node.properties.get("_iterSpinner");
                String[] operations = {"Erode", "Dilate", "Open", "Close", "Gradient", "TopHat", "BlackHat"};
                int opIndex = 0;
                for (int i = 0; i < operations.length; i++) {
                    if (operations[i].equals(opCombo.getValue())) { opIndex = i; break; }
                }
                String[] shapes = {"Rect", "Cross", "Ellipse"};
                int shapeIndex = 0;
                for (int i = 0; i < shapes.length; i++) {
                    if (shapes[i].equals(shapeCombo.getValue())) { shapeIndex = i; break; }
                }
                node.properties.put("operationIndex", opIndex);
                node.properties.put("shapeIndex", shapeIndex);
                node.properties.put("kernelWidth", widthSpinner.getValue());
                node.properties.put("kernelHeight", heightSpinner.getValue());
                node.properties.put("iterations", iterSpinner.getValue());
                node.properties.remove("_opCombo");
                node.properties.remove("_shapeCombo");
                node.properties.remove("_kernelWidthSpinner");
                node.properties.remove("_kernelHeightSpinner");
                node.properties.remove("_iterSpinner");
            }

            // Handle HoughCircles properties
            if ("HoughCircles".equals(node.nodeType) && node.properties.containsKey("_showOrigCheck")) {
                CheckBox showOrigCheck = (CheckBox) node.properties.get("_showOrigCheck");
                Slider minDistSlider = (Slider) node.properties.get("_minDistSlider");
                Slider param1Slider = (Slider) node.properties.get("_param1Slider");
                Slider param2Slider = (Slider) node.properties.get("_param2Slider");
                @SuppressWarnings("unchecked")
                Spinner<Integer> minRadSpinner = (Spinner<Integer>) node.properties.get("_minRadSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> maxRadSpinner = (Spinner<Integer>) node.properties.get("_maxRadSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> thickSpinner = (Spinner<Integer>) node.properties.get("_thickSpinner");
                TextField colorField = (TextField) node.properties.get("_colorField");

                node.properties.put("showOriginal", showOrigCheck.isSelected());
                node.properties.put("minDist", (int) minDistSlider.getValue());
                node.properties.put("param1", (int) param1Slider.getValue());
                node.properties.put("param2", (int) param2Slider.getValue());
                node.properties.put("minRadius", minRadSpinner.getValue());
                node.properties.put("maxRadius", maxRadSpinner.getValue());
                node.properties.put("thickness", thickSpinner.getValue());
                node.properties.put("colorRGB", colorField.getText());

                node.properties.remove("_showOrigCheck");
                node.properties.remove("_minDistSlider");
                node.properties.remove("_param1Slider");
                node.properties.remove("_param2Slider");
                node.properties.remove("_minRadSpinner");
                node.properties.remove("_maxRadSpinner");
                node.properties.remove("_thickSpinner");
                node.properties.remove("_colorField");
            }

            // Handle HoughLines properties
            if ("HoughLines".equals(node.nodeType) && node.properties.containsKey("_threshSlider")) {
                Slider threshSlider = (Slider) node.properties.get("_threshSlider");
                Slider minLenSlider = (Slider) node.properties.get("_minLenSlider");
                Slider maxGapSlider = (Slider) node.properties.get("_maxGapSlider");
                @SuppressWarnings("unchecked")
                Spinner<Integer> thickSpinner = (Spinner<Integer>) node.properties.get("_thickSpinner");
                TextField colorField = (TextField) node.properties.get("_colorField");

                node.properties.put("threshold", (int) threshSlider.getValue());
                node.properties.put("minLineLength", (int) minLenSlider.getValue());
                node.properties.put("maxLineGap", (int) maxGapSlider.getValue());
                node.properties.put("thickness", thickSpinner.getValue());
                node.properties.put("colorRGB", colorField.getText());

                node.properties.remove("_threshSlider");
                node.properties.remove("_minLenSlider");
                node.properties.remove("_maxGapSlider");
                node.properties.remove("_thickSpinner");
                node.properties.remove("_colorField");
            }

            // Handle HarrisCorners properties
            if ("HarrisCorners".equals(node.nodeType) && node.properties.containsKey("_showOrigCheck")) {
                CheckBox showOrigCheck = (CheckBox) node.properties.get("_showOrigCheck");
                @SuppressWarnings("unchecked")
                Spinner<Integer> blockSpinner = (Spinner<Integer>) node.properties.get("_blockSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> ksizeSpinner = (Spinner<Integer>) node.properties.get("_ksizeSpinner");
                Slider kSlider = (Slider) node.properties.get("_kSlider");
                Slider threshSlider = (Slider) node.properties.get("_threshSlider");
                @SuppressWarnings("unchecked")
                Spinner<Integer> markerSpinner = (Spinner<Integer>) node.properties.get("_markerSpinner");
                TextField colorField = (TextField) node.properties.get("_colorField");

                node.properties.put("showOriginal", showOrigCheck.isSelected());
                node.properties.put("blockSize", blockSpinner.getValue());
                node.properties.put("ksize", ksizeSpinner.getValue());
                node.properties.put("kPercent", (int) kSlider.getValue());
                node.properties.put("thresholdPercent", (int) threshSlider.getValue());
                node.properties.put("markerSize", markerSpinner.getValue());
                node.properties.put("colorRGB", colorField.getText());

                node.properties.remove("_showOrigCheck");
                node.properties.remove("_blockSpinner");
                node.properties.remove("_ksizeSpinner");
                node.properties.remove("_kSlider");
                node.properties.remove("_threshSlider");
                node.properties.remove("_markerSpinner");
                node.properties.remove("_colorField");
            }

            // Handle Contours properties
            if ("Contours".equals(node.nodeType) && node.properties.containsKey("_showOrigCheck")) {
                CheckBox showOrigCheck = (CheckBox) node.properties.get("_showOrigCheck");
                @SuppressWarnings("unchecked")
                Spinner<Integer> thickSpinner = (Spinner<Integer>) node.properties.get("_thickSpinner");
                TextField colorField = (TextField) node.properties.get("_colorField");

                node.properties.put("showOriginal", showOrigCheck.isSelected());
                node.properties.put("thickness", thickSpinner.getValue());
                node.properties.put("colorRGB", colorField.getText());

                node.properties.remove("_showOrigCheck");
                node.properties.remove("_thickSpinner");
                node.properties.remove("_colorField");
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
                TextField colorField = (TextField) node.properties.get("_colorField");
                @SuppressWarnings("unchecked")
                Spinner<Integer> thickSpinner = (Spinner<Integer>) node.properties.get("_thickSpinner");
                CheckBox filledCheck = (CheckBox) node.properties.get("_filledCheck");

                node.properties.put("x1", x1Spinner.getValue());
                node.properties.put("y1", y1Spinner.getValue());
                node.properties.put("x2", x2Spinner.getValue());
                node.properties.put("y2", y2Spinner.getValue());
                node.properties.put("colorRGB", colorField.getText());
                node.properties.put("thickness", thickSpinner.getValue());
                node.properties.put("filled", filledCheck.isSelected());

                node.properties.remove("_x1Spinner");
                node.properties.remove("_y1Spinner");
                node.properties.remove("_x2Spinner");
                node.properties.remove("_y2Spinner");
                node.properties.remove("_colorField");
                node.properties.remove("_thickSpinner");
                node.properties.remove("_filledCheck");
            }

            // Handle Circle properties
            if ("Circle".equals(node.nodeType) && node.properties.containsKey("_cxSpinner")) {
                @SuppressWarnings("unchecked")
                Spinner<Integer> cxSpinner = (Spinner<Integer>) node.properties.get("_cxSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> cySpinner = (Spinner<Integer>) node.properties.get("_cySpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> radSpinner = (Spinner<Integer>) node.properties.get("_radSpinner");
                TextField colorField = (TextField) node.properties.get("_colorField");
                @SuppressWarnings("unchecked")
                Spinner<Integer> thickSpinner = (Spinner<Integer>) node.properties.get("_thickSpinner");
                CheckBox filledCheck = (CheckBox) node.properties.get("_filledCheck");

                node.properties.put("centerX", cxSpinner.getValue());
                node.properties.put("centerY", cySpinner.getValue());
                node.properties.put("radius", radSpinner.getValue());
                node.properties.put("colorRGB", colorField.getText());
                node.properties.put("thickness", thickSpinner.getValue());
                node.properties.put("filled", filledCheck.isSelected());

                node.properties.remove("_cxSpinner");
                node.properties.remove("_cySpinner");
                node.properties.remove("_radSpinner");
                node.properties.remove("_colorField");
                node.properties.remove("_thickSpinner");
                node.properties.remove("_filledCheck");
            }

            // Handle Ellipse properties
            if ("Ellipse".equals(node.nodeType) && node.properties.containsKey("_cxSpinner")) {
                @SuppressWarnings("unchecked")
                Spinner<Integer> cxSpinner = (Spinner<Integer>) node.properties.get("_cxSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> cySpinner = (Spinner<Integer>) node.properties.get("_cySpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> axSpinner = (Spinner<Integer>) node.properties.get("_axSpinner");
                @SuppressWarnings("unchecked")
                Spinner<Integer> aySpinner = (Spinner<Integer>) node.properties.get("_aySpinner");
                Slider angleSlider = (Slider) node.properties.get("_angleSlider");
                TextField colorField = (TextField) node.properties.get("_colorField");
                @SuppressWarnings("unchecked")
                Spinner<Integer> thickSpinner = (Spinner<Integer>) node.properties.get("_thickSpinner");
                CheckBox filledCheck = (CheckBox) node.properties.get("_filledCheck");

                node.properties.put("centerX", cxSpinner.getValue());
                node.properties.put("centerY", cySpinner.getValue());
                node.properties.put("axisX", axSpinner.getValue());
                node.properties.put("axisY", aySpinner.getValue());
                node.properties.put("angle", (int) angleSlider.getValue());
                node.properties.put("colorRGB", colorField.getText());
                node.properties.put("thickness", thickSpinner.getValue());
                node.properties.put("filled", filledCheck.isSelected());

                node.properties.remove("_cxSpinner");
                node.properties.remove("_cySpinner");
                node.properties.remove("_axSpinner");
                node.properties.remove("_aySpinner");
                node.properties.remove("_angleSlider");
                node.properties.remove("_colorField");
                node.properties.remove("_thickSpinner");
                node.properties.remove("_filledCheck");
            }

            // Handle Grayscale properties
            if ("Grayscale".equals(node.nodeType) && node.properties.containsKey("_convCombo")) {
                @SuppressWarnings("unchecked")
                ComboBox<String> convCombo = (ComboBox<String>) node.properties.get("_convCombo");
                String[] conversions = {"BGR to Gray", "Gray to BGR", "BGR to HSV", "HSV to BGR",
                    "BGR to LAB", "LAB to BGR", "BGR to YCrCb", "YCrCb to BGR"};
                int conversionIndex = 0;
                for (int i = 0; i < conversions.length; i++) {
                    if (conversions[i].equals(convCombo.getValue())) { conversionIndex = i; break; }
                }
                node.properties.put("conversionIndex", conversionIndex);
                node.properties.remove("_convCombo");
            }

            // Handle FileSource properties
            if ("FileSource".equals(node.nodeType) && node.properties.containsKey("_fileField")) {
                TextField fileField = (TextField) node.properties.get("_fileField");
                @SuppressWarnings("unchecked")
                ComboBox<String> fpsCombo = (ComboBox<String>) node.properties.get("_fpsCombo");
                node.properties.put("filePath", fileField.getText());
                String[] fpsOptions = {"1", "15", "30", "60"};
                int fpsIndex = 2;
                for (int i = 0; i < fpsOptions.length; i++) {
                    if (fpsOptions[i].equals(fpsCombo.getValue())) { fpsIndex = i; break; }
                }
                node.properties.put("fpsIndex", fpsIndex);
                node.properties.remove("_fileField");
                node.properties.remove("_fpsCombo");
            }

            // Handle WebcamSource properties
            if ("WebcamSource".equals(node.nodeType) && node.properties.containsKey("_camSpinner")) {
                @SuppressWarnings("unchecked")
                Spinner<Integer> camSpinner = (Spinner<Integer>) node.properties.get("_camSpinner");
                @SuppressWarnings("unchecked")
                ComboBox<String> fpsCombo = (ComboBox<String>) node.properties.get("_fpsCombo");
                node.properties.put("cameraIndex", camSpinner.getValue());
                String[] fpsOptions = {"1", "15", "30", "60"};
                int fpsIndex = 2;
                for (int i = 0; i < fpsOptions.length; i++) {
                    if (fpsOptions[i].equals(fpsCombo.getValue())) { fpsIndex = i; break; }
                }
                node.properties.put("fpsIndex", fpsIndex);
                node.properties.remove("_camSpinner");
                node.properties.remove("_fpsCombo");
            }

            // Handle dual-input "Queues in Sync" property
            if (isDualInput && finalSyncCheckBox != null) {
                node.queuesInSync = finalSyncCheckBox.isSelected();
            }

            paintCanvas();
            notifyModified();
        });

        dialog.showAndWaitForResult();
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

    /**
     * Check if a node type is a dual-input node by type name.
     */
    private boolean isDualInputNodeType(String nodeType) {
        return "AddClamp".equals(nodeType) ||
               "SubtractClamp".equals(nodeType) ||
               "AddWeighted".equals(nodeType) ||
               "BitwiseAnd".equals(nodeType) ||
               "BitwiseOr".equals(nodeType) ||
               "BitwiseXor".equals(nodeType);
    }

    private FXNode getNodeAt(double x, double y) {
        for (int i = nodes.size() - 1; i >= 0; i--) {
            FXNode node = nodes.get(i);
            if (node.contains(x, y)) {
                return node;
            }
        }
        return null;
    }

    /**
     * Find a connection near the given point.
     * Uses distance-to-bezier-curve calculation.
     */
    private FXConnection getConnectionAt(double px, double py, double tolerance) {
        for (FXConnection conn : connections) {
            double[] startPt = conn.source.getOutputPoint(conn.sourceOutputIndex);
            double[] endPt = conn.target.getInputPoint(conn.targetInputIndex);
            if (startPt != null && endPt != null) {
                if (isPointNearBezier(px, py, startPt[0], startPt[1], endPt[0], endPt[1], tolerance)) {
                    return conn;
                }
            }
        }
        return null;
    }

    /**
     * Check if a point is near a bezier curve (same curve shape as rendered connections).
     */
    private boolean isPointNearBezier(double px, double py,
                                       double startX, double startY,
                                       double endX, double endY,
                                       double tolerance) {
        // Use the same control point calculation as NodeRenderer.renderConnection
        double ctrlOffset = Math.abs(endX - startX) / 2;
        if (ctrlOffset < 30) ctrlOffset = 30;
        double ctrl1X = startX + ctrlOffset;
        double ctrl1Y = startY;
        double ctrl2X = endX - ctrlOffset;
        double ctrl2Y = endY;

        // Sample the bezier curve at multiple points and check distance
        int samples = 20;
        for (int i = 0; i <= samples; i++) {
            double t = (double) i / samples;
            // Cubic bezier formula
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

    /**
     * Check if a connection's bezier curve intersects a rectangle.
     */
    private boolean connectionIntersectsRect(FXConnection conn, double x1, double y1, double x2, double y2) {
        double[] startPt = conn.source.getOutputPoint(conn.sourceOutputIndex);
        double[] endPt = conn.target.getInputPoint(conn.targetInputIndex);
        if (startPt == null || endPt == null) return false;

        double startX = startPt[0], startY = startPt[1];
        double endX = endPt[0], endY = endPt[1];

        // Use same bezier control points as rendering
        double ctrlOffset = Math.abs(endX - startX) / 2;
        if (ctrlOffset < 30) ctrlOffset = 30;
        double ctrl1X = startX + ctrlOffset;
        double ctrl1Y = startY;
        double ctrl2X = endX - ctrlOffset;
        double ctrl2Y = endY;

        // Sample the bezier and check if any point is inside the rect
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

    private void addNode(String nodeType) {
        // Calculate position in visible area
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

    private void deleteSelected() {
        // Remove selected connections
        connections.removeAll(selectedConnections);

        // Remove connections to/from selected nodes
        for (FXNode node : selectedNodes) {
            connections.removeIf(c -> c.source == node || c.target == node);
        }

        // Remove selected nodes
        nodes.removeAll(selectedNodes);

        selectedNodes.clear();
        selectedConnections.clear();
        paintCanvas();
        updateStatus();
        notifyModified();
    }

    private void filterToolbarButtons() {
        String filter = searchBox.getText().toLowerCase().trim();

        toolbarContent.getChildren().clear();

        // Container I/O nodes are NOT shown in toolbar - they are auto-created via ensureBoundaryNodes()
        // Only show other categories
        for (String category : FXNodeRegistry.getCategoriesExcluding("Container I/O")) {
            boolean categoryAdded = false;
            for (FXNodeRegistry.NodeType nodeType : FXNodeRegistry.getNodesInCategory(category)) {
                if (filter.isEmpty() || nodeType.displayName.toLowerCase().contains(filter)
                        || nodeType.getButtonName().toLowerCase().contains(filter)) {
                    if (!categoryAdded) {
                        addToolbarCategory(category);
                        categoryAdded = true;
                    }
                    final String typeName = nodeType.name;
                    addToolbarButton(nodeType.getButtonName(), () -> addNode(typeName));
                }
            }
        }
    }

    private void updateStatus() {
        statusLabel.setText("Nodes: " + nodes.size());
    }

    private void notifyModified() {
        if (onModified != null) {
            onModified.run();
        }
    }

    private void paintCanvas() {
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

        // Draw connections
        for (FXConnection conn : connections) {
            // Skip connections with null source or target (can happen with malformed data)
            if (conn.source == null || conn.target == null) {
                continue;
            }
            boolean isSelected = selectedConnections.contains(conn);
            double[] startPt = conn.source.getOutputPoint(conn.sourceOutputIndex);
            double[] endPt = conn.target.getInputPoint(conn.targetInputIndex);
            if (startPt != null && endPt != null) {
                NodeRenderer.renderConnection(gc, startPt[0], startPt[1], endPt[0], endPt[1], isSelected,
                    conn.queueSize, conn.totalFrames);
            }
        }

        // Draw in-progress connection (forward: from output)
        if (isDrawingConnection && connectionSource != null) {
            double[] startPt = connectionSource.getOutputPoint(connectionOutputIndex);
            if (startPt != null) {
                gc.setStroke(Color.BLUE);
                gc.setLineWidth(2);
                gc.strokeLine(startPt[0], startPt[1], connectionEndX, connectionEndY);
            }
        }

        // Draw in-progress reverse connection (from input)
        if (isDrawingReverseConnection && connectionTarget != null) {
            double[] endPt = connectionTarget.getInputPoint(connectionInputIndex);
            if (endPt != null) {
                gc.setStroke(Color.BLUE);
                gc.setLineWidth(2);
                gc.strokeLine(connectionEndX, connectionEndY, endPt[0], endPt[1]);
            }
        }

        // Draw nodes
        for (FXNode node : nodes) {
            boolean isSelected = selectedNodes.contains(node);
            // Build output counters array from node fields
            int[] outputCounters = new int[] { node.outputCount1, node.outputCount2, node.outputCount3, node.outputCount4 };
            NodeRenderer.renderNode(gc, node.x, node.y, node.width, node.height,
                node.label, isSelected, node.enabled, node.backgroundColor,
                node.hasInput, node.hasDualInput, node.outputCount,
                node.thumbnail, node.isContainer,
                node.inputCount, node.inputCount2, outputCounters, node.nodeType, node.isBoundaryNode);

            // Draw stats line (Pri/Work/FPS) - always show (including after load)
            // Source nodes (no input, not boundary) show FPS; boundary nodes and processing nodes don't
            boolean isSourceNode = !node.hasInput && !node.isBoundaryNode;
            if (isSourceNode) {
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
    }
}
