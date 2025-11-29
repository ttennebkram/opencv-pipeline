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
            private int debugCounter = 0;

            @Override
            public void handle(long now) {
                if (now - lastUpdate >= REPAINT_INTERVAL_NS) {
                    if (isPipelineRunning != null && isPipelineRunning.get()) {
                        debugCounter++;
                        if (debugCounter % 10 == 1) {
                            // Print debug info every 10 frames (~1 second)
                            System.out.println("ContainerEditor repaint #" + debugCounter +
                                " nodes=" + nodes.size() +
                                " firstNode=" + (nodes.isEmpty() ? "none" : nodes.get(0).label +
                                    " in=" + nodes.get(0).inputCount +
                                    " out=" + nodes.get(0).outputCount1 +
                                    " thumb=" + (nodes.get(0).thumbnail != null)));
                        }
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
            // Update button state after toggling
            updatePipelineButtonState();
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
            showNodeProperties(node);
        }
    }

    private void showNodeProperties(FXNode node) {
        // Simple properties dialog
        TextInputDialog dialog = new TextInputDialog(node.label);
        dialog.setTitle("Node Properties");
        dialog.setHeaderText("Edit node: " + node.nodeType);
        dialog.setContentText("Name:");
        dialog.showAndWait().ifPresent(name -> {
            node.label = name;
            paintCanvas();
            notifyModified();
        });
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
                node.inputCount, outputCounters, node.nodeType, node.isBoundaryNode);

            // Draw stats line (Pri/Work/FPS) - only when there's work being done
            if (node.workUnitsCompleted > 0) {
                if (!node.hasInput) {
                    // Source node - show FPS as well
                    NodeRenderer.drawSourceStatsLine(gc, node.x + 22, node.y + node.height - 8,
                        node.threadPriority, node.workUnitsCompleted, node.effectiveFps);
                } else {
                    // Processing node
                    NodeRenderer.drawStatsLine(gc, node.x + 22, node.y + node.height - 8,
                        node.threadPriority, node.workUnitsCompleted);
                }
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
