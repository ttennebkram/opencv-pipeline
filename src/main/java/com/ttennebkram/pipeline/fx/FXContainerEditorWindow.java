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
    private java.util.function.Supplier<Integer> getThreadCount;

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
    private FXConnection yankingConnection = null;  // Connection being yanked (detached from one end)
    private boolean yankingFromTarget = false;       // True if yanking from target end, false if from source end

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

    // Tooltip support
    private Tooltip canvasTooltip = new Tooltip();

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
            inputNode = new FXNode("Container Input", "ContainerInput", 50, 50);
            inputNode.hasInput = false;  // Source node - no input
            inputNode.outputCount = 1;
            inputNode.isBoundaryNode = true;
            inputNode.backgroundColor = NodeRenderer.COLOR_CONTAINER_NODE;
            nodes.add(inputNode);
        }

        if (outputNode == null) {
            // ContainerOutput has an input but no output
            outputNode = new FXNode("Container Output", "ContainerOutput", 400, 50);
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
            } else if (e.getCode() == javafx.scene.input.KeyCode.UP) {
                moveSelectedNodes(0, -10);
                e.consume();
            } else if (e.getCode() == javafx.scene.input.KeyCode.DOWN) {
                moveSelectedNodes(0, 10);
                e.consume();
            } else if (e.getCode() == javafx.scene.input.KeyCode.LEFT) {
                moveSelectedNodes(-10, 0);
                e.consume();
            } else if (e.getCode() == javafx.scene.input.KeyCode.RIGHT) {
                moveSelectedNodes(10, 0);
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

        // Intercept arrow keys before ScrollPane consumes them for scrolling
        // Use event filter (capture phase) to get the event before ScrollPane
        canvasScrollPane.addEventFilter(javafx.scene.input.KeyEvent.KEY_PRESSED, e -> {
            if (e.getCode() == javafx.scene.input.KeyCode.UP ||
                e.getCode() == javafx.scene.input.KeyCode.DOWN ||
                e.getCode() == javafx.scene.input.KeyCode.LEFT ||
                e.getCode() == javafx.scene.input.KeyCode.RIGHT) {
                // Only consume if we have selected items to move
                if (!selectedNodes.isEmpty() || !selectedConnections.isEmpty()) {
                    if (e.getCode() == javafx.scene.input.KeyCode.UP) {
                        moveSelectedNodes(0, -10);
                    } else if (e.getCode() == javafx.scene.input.KeyCode.DOWN) {
                        moveSelectedNodes(0, 10);
                    } else if (e.getCode() == javafx.scene.input.KeyCode.LEFT) {
                        moveSelectedNodes(-10, 0);
                    } else if (e.getCode() == javafx.scene.input.KeyCode.RIGHT) {
                        moveSelectedNodes(10, 0);
                    }
                    e.consume();  // Prevent ScrollPane from scrolling
                }
            }
        });

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
                startStopBtn.setStyle("-fx-base: #F08080;");  // Light coral when running
            } else {
                startStopBtn.setText("Start Pipeline");
                startStopBtn.setStyle("-fx-base: #90EE90;");  // Light green when not running
            }
            // Update status bar label
            if (pipelineStatusLabel != null) {
                if (running) {
                    int threadCount = (getThreadCount != null) ? getThreadCount.get() : 0;
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

    public void setGetThreadCount(java.util.function.Supplier<Integer> supplier) {
        this.getThreadCount = supplier;
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
        canvas.setOnMouseMoved(this::handleMouseMoved);
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

    private void handleMouseMoved(javafx.scene.input.MouseEvent e) {
        double canvasX = e.getX() / zoomLevel;
        double canvasY = e.getY() / zoomLevel;

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

        // Update cursor - show hand for connection points
        if (connectionPointNode != null) {
            canvas.setCursor(javafx.scene.Cursor.HAND);
        } else {
            canvas.setCursor(javafx.scene.Cursor.DEFAULT);
        }

        // Update tooltip based on what we're hovering over
        String tooltipText = null;
        if (connectionPointNode != null) {
            // Show connection point tooltip
            tooltipText = getConnectionPointTooltip(connectionPointNode, isInputPoint, connectionPointIndex);
        }

        // Install or uninstall tooltip based on hover state
        if (tooltipText != null) {
            canvasTooltip.setText(tooltipText);
            Tooltip.install(canvas, canvasTooltip);
        } else {
            Tooltip.uninstall(canvas, canvasTooltip);
        }
    }

    /**
     * Get tooltip text for a connection point on a node.
     * Simplified version for container editor - just covers common node types.
     */
    private String getConnectionPointTooltip(FXNode node, boolean isInput, int index) {
        String nodeType = node.nodeType;

        if (isInput) {
            // Input tooltips
            if (node.hasDualInput) {
                return "Input " + (index + 1) + " (Mat)";
            } else {
                return "Input (Mat)";
            }
        } else {
            // Output tooltips
            if (node.outputCount > 1) {
                return "Output " + (index + 1) + " (Mat)";
            } else {
                // Container boundary nodes
                if ("ContainerInput".equals(nodeType)) {
                    return "Output: Container input data (Mat)";
                } else if ("ContainerOutput".equals(nodeType)) {
                    return "Input: Data to send to parent (Mat)";
                }
                return "Output (Mat)";
            }
        }
    }

    private void handleMousePressed(javafx.scene.input.MouseEvent e) {
        double canvasX = e.getX() / zoomLevel;
        double canvasY = e.getY() / zoomLevel;

        double tolerance = NodeRenderer.CONNECTION_RADIUS + 9;

        // First, check for clicks on free endpoints of dangling connections
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

        // Next, check ALL nodes for connection point clicks (connection points are on edges,
        // so the click might not be "inside" the node bounds)
        for (FXNode node : nodes) {
            int outputIdx = node.getOutputPointAt(canvasX, canvasY, tolerance);
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
                    notifyModified();
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

            int inputIdx = node.getInputPointAt(canvasX, canvasY, tolerance);
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
                    notifyModified();
                    paintCanvas();
                    return;
                }
                // No existing connection - start a "reverse" connection (user will drag to an output)
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
        } else if (isDrawingReverseConnection) {
            connectionEndX = canvasX;
            connectionEndY = canvasY;
            paintCanvas();
        } else if (dragNode != null) {
            isDragging = true;
            // Calculate delta from drag node's movement
            double newDragNodeX = canvasX - dragOffsetX;
            double newDragNodeY = canvasY - dragOffsetY;
            double deltaX = newDragNodeX - dragNode.x;
            double deltaY = newDragNodeY - dragNode.y;

            // Move all selected nodes by the same delta
            for (FXNode node : selectedNodes) {
                node.x += deltaX;
                node.y += deltaY;
            }

            // Also move free endpoints of selected dangling connections
            for (FXConnection conn : selectedConnections) {
                if (conn.source == null) {
                    conn.freeSourceX += deltaX;
                    conn.freeSourceY += deltaY;
                }
                if (conn.target == null) {
                    conn.freeTargetX += deltaX;
                    conn.freeTargetY += deltaY;
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

        if (isDrawingConnection) {
            boolean connected = false;
            double tolerance = NodeRenderer.CONNECTION_RADIUS + 9;

            if (yankingConnection != null) {
                // We're reconnecting a yanked connection (preserving queue data)
                if (yankingFromTarget) {
                    // Yanking from target end - try to reconnect target to a new input
                    for (FXNode node : nodes) {
                        if (node != yankingConnection.source) {
                            int inputIdx = node.getInputPointAt(canvasX, canvasY, tolerance);
                            if (inputIdx >= 0) {
                                // Check if there's already a connection to this input - reject drop if occupied
                                FXConnection existingConn = findConnectionToInput(node, inputIdx);
                                if (existingConn != null && existingConn != yankingConnection) {
                                    // Input occupied - leave yanked connection dangling
                                    break;
                                }
                                // Reconnect the yanked connection
                                yankingConnection.reconnectTarget(node, inputIdx);
                                connected = true;
                                notifyModified();
                                break;
                            }
                        }
                    }
                    if (!connected) {
                        // Leave connection with dangling target - push away from nearby connection points
                        double[] pushed = pushAwayFromConnectionPoints(canvasX, canvasY, 30);
                        yankingConnection.freeTargetX = pushed[0];
                        yankingConnection.freeTargetY = pushed[1];
                        notifyModified();
                    }
                } else {
                    // Yanking from source end - try to reconnect source to a new output
                    for (FXNode node : nodes) {
                        if (node != yankingConnection.target) {
                            int outputIdx = node.getOutputPointAt(canvasX, canvasY, tolerance);
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
                                notifyModified();
                                break;
                            }
                        }
                    }
                    if (!connected) {
                        // Leave connection with dangling source - push away from nearby connection points
                        double[] pushed = pushAwayFromConnectionPoints(canvasX, canvasY, 30);
                        yankingConnection.freeSourceX = pushed[0];
                        yankingConnection.freeSourceY = pushed[1];
                        notifyModified();
                    }
                }
                yankingConnection = null;
            } else if (connectionSource != null) {
                // Creating a new connection (not yanking)
                for (FXNode targetNode : nodes) {
                    if (targetNode != connectionSource) {
                        int inputIdx = targetNode.getInputPointAt(canvasX, canvasY, tolerance);
                        if (inputIdx >= 0) {
                            // Check if there's already a connection to this input - reject the drop
                            FXConnection existingConn = findConnectionToInput(targetNode, inputIdx);
                            if (existingConn != null) {
                                // Input already occupied - create dangling connection instead
                                break;
                            }

                            // Create connection
                            FXConnection conn = new FXConnection(connectionSource, connectionOutputIndex, targetNode, inputIdx);
                            connections.add(conn);
                            notifyModified();
                            connected = true;
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
                    notifyModified();
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
        nestedEditor.setGetThreadCount(getThreadCount);
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
            // Use getStartPoint/getEndPoint to handle dangling connections
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
     * Returns the first connection found to that node's input index,
     * or a target-dangling connection whose free end is near the input point.
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
        paintCanvas();
        updateStatus();
        notifyModified();
    }

    private void moveSelectedNodes(double dx, double dy) {
        if (selectedNodes.isEmpty() && selectedConnections.isEmpty()) {
            return;
        }

        // Move selected nodes
        for (FXNode node : selectedNodes) {
            node.x += dx;
            node.y += dy;
        }

        // Move free endpoints of selected connections (dangling connections)
        for (FXConnection conn : selectedConnections) {
            if (conn.source == null) {
                conn.freeSourceX += dx;
                conn.freeSourceY += dy;
            }
            if (conn.target == null) {
                conn.freeTargetX += dx;
                conn.freeTargetY += dy;
            }
        }

        paintCanvas();
        notifyModified();
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
            // Add Connector/Queue button to Utility category
            if (category.equals("Utility")) {
                if (filter.isEmpty() || "connector".contains(filter) || "queue".contains(filter)) {
                    if (!categoryAdded) {
                        addToolbarCategory(category);
                    }
                    addToolbarButton("Connector/Queue", this::addConnectorQueue);
                }
            }
        }
    }

    /**
     * Add a standalone Connector/Queue (a dangling connection with both ends free).
     */
    private void addConnectorQueue() {
        double x = 100;
        double y = 100 + nodes.size() * 30;  // Offset based on node count

        FXConnection conn = FXConnection.createDangling(x, y, x + 100, y);
        connections.add(conn);

        selectedNodes.clear();
        selectedConnections.clear();
        selectedConnections.add(conn);

        paintCanvas();
        notifyModified();
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

        // Draw connections (including dangling connections with free endpoints)
        for (FXConnection conn : connections) {
            boolean isSelected = selectedConnections.contains(conn);
            double[] start = conn.getStartPoint();
            double[] end = conn.getEndPoint();
            if (start != null && end != null) {
                NodeRenderer.renderConnection(gc, start[0], start[1], end[0], end[1], isSelected,
                    conn.queueSize, conn.totalFrames);
            }
        }

        // Draw in-progress connection (forward: from output)
        // Only for NEW connections, not when yanking existing ones
        // Yanked connections are already in the connections list and render via their free endpoints
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
