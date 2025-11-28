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
    private static final Color COLOR_SELECTION_BOX = Color.rgb(0, 120, 215);
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

    // Connection drawing state
    private FXNode connectionSource = null;
    private int connectionOutputIndex = 0;
    private double connectionEndX, connectionEndY;
    private boolean isDrawingConnection = false;

    // Zoom
    private double zoomLevel = 1.0;
    private static final int[] ZOOM_LEVELS = {25, 50, 75, 100, 125, 150, 200};

    public FXContainerEditorWindow(Stage parentStage, FXNode containerNode, Runnable onModified) {
        this.parentStage = parentStage;
        this.containerNode = containerNode;
        this.onModified = onModified;
        this.nodes = containerNode.innerNodes;
        this.connections = containerNode.innerConnections;

        createWindow();
    }

    private void createWindow() {
        stage = new Stage();
        stage.setTitle("Container: " + containerNode.label);
        stage.initOwner(parentStage);
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

        Scene scene = new Scene(root, 1000, 600);

        // Keyboard handler
        scene.setOnKeyPressed(e -> {
            if (e.getCode() == javafx.scene.input.KeyCode.DELETE ||
                e.getCode() == javafx.scene.input.KeyCode.BACK_SPACE) {
                deleteSelected();
                e.consume();
            }
        });

        stage.setScene(scene);

        // Position near parent
        stage.setX(parentStage.getX() + 50);
        stage.setY(parentStage.getY() + 50);

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
        toolbar.setPrefWidth(180);

        // Search box with clear button inside using StackPane
        searchBox = new TextField();
        searchBox.setPromptText("Search nodes...");
        searchBox.textProperty().addListener((obs, oldVal, newVal) -> filterToolbarButtons());
        // Add padding on right for clear button
        searchBox.setStyle("-fx-padding: 2 22 2 5;");

        Button clearSearchBtn = new Button("\u00D7"); // Unicode multiplication sign (looks like x)
        clearSearchBtn.setStyle("-fx-font-size: 12px; -fx-padding: 0 5 0 5; -fx-background-color: #cccccc; -fx-background-radius: 8; -fx-cursor: hand; -fx-min-width: 16; -fx-min-height: 16;");
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

        // Populate with all node types (including sources for use cases like loading reference images)
        for (String category : FXNodeRegistry.getCategories()) {
            boolean hasNodes = false;
            for (FXNodeRegistry.NodeType nodeType : FXNodeRegistry.getNodesInCategory(category)) {
                if (!hasNodes) {
                    addToolbarCategory(category);
                    hasNodes = true;
                }
                final String typeName = nodeType.name;
                addToolbarButton(nodeType.displayName, () -> addNode(typeName));
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
        canvasScrollPane.setPannable(true);

        return canvasScrollPane;
    }

    private VBox createPreviewPane() {
        VBox previewPane = new VBox(5);
        previewPane.setPadding(new Insets(10));
        previewPane.setStyle("-fx-background-color: #f0f0f0;");

        // Start/Stop pipeline button at top
        startStopBtn = new Button("Start Pipeline");
        startStopBtn.setStyle("-fx-background-color: rgb(100, 180, 100); -fx-font-weight: bold;");
        startStopBtn.setOnAction(e -> togglePipeline());
        startStopBtn.setMaxWidth(Double.MAX_VALUE);

        // Instructions label
        Label instructionsLabel = new Label("Instructions:");
        instructionsLabel.setStyle("-fx-font-weight: bold; -fx-padding: 5 0 0 0;");

        Label instructions = new Label(
            "- Click node name to create\n" +
            "- Drag nodes to move\n" +
            "- Click circles to connect\n" +
            "- Double-click for properties\n" +
            "- Click node to see preview\n" +
            "- Edit while running for live updates"
        );
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

        previewPane.getChildren().addAll(startStopBtn, instructionsLabel, instructions, previewLabel, imageContainer);

        return previewPane;
    }

    private void togglePipeline() {
        if (onStartPipeline != null && onStopPipeline != null && isPipelineRunning != null) {
            if (isPipelineRunning.get()) {
                onStopPipeline.run();
            } else {
                onStartPipeline.run();
            }
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

        FXNode clickedNode = getNodeAt(canvasX, canvasY);

        if (clickedNode != null) {
            // Check if clicking on output point
            int outputIdx = clickedNode.getOutputPointAt(canvasX, canvasY, 10);
            if (outputIdx >= 0) {
                connectionSource = clickedNode;
                connectionOutputIndex = outputIdx;
                connectionEndX = canvasX;
                connectionEndY = canvasY;
                isDrawingConnection = true;
                return;
            }

            // Check if clicking on checkbox
            if (clickedNode.isOnCheckbox(canvasX, canvasY)) {
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
            selectedNodes.clear();
            selectedConnections.clear();
            paintCanvas();
        }
    }

    private void handleMouseDragged(javafx.scene.input.MouseEvent e) {
        double canvasX = e.getX() / zoomLevel;
        double canvasY = e.getY() / zoomLevel;

        if (isDrawingConnection) {
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
        }
    }

    private void handleMouseReleased(javafx.scene.input.MouseEvent e) {
        double canvasX = e.getX() / zoomLevel;
        double canvasY = e.getY() / zoomLevel;

        if (isDrawingConnection && connectionSource != null) {
            // Check if released over an input point
            FXNode targetNode = getNodeAt(canvasX, canvasY);
            if (targetNode != null && targetNode != connectionSource) {
                int inputIdx = targetNode.getInputPointAt(canvasX, canvasY, 10);
                if (inputIdx >= 0) {
                    // Create connection
                    FXConnection conn = new FXConnection(connectionSource, connectionOutputIndex, targetNode, inputIdx);
                    connections.add(conn);
                    notifyModified();
                }
            }
            isDrawingConnection = false;
            connectionSource = null;
            paintCanvas();
        }

        if (isDragging) {
            notifyModified();
        }
        dragNode = null;
        isDragging = false;
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

        for (String category : FXNodeRegistry.getCategories()) {
            boolean categoryAdded = false;
            for (FXNodeRegistry.NodeType nodeType : FXNodeRegistry.getNodesInCategory(category)) {
                if (filter.isEmpty() || nodeType.displayName.toLowerCase().contains(filter)) {
                    if (!categoryAdded) {
                        addToolbarCategory(category);
                        categoryAdded = true;
                    }
                    final String typeName = nodeType.name;
                    addToolbarButton(nodeType.displayName, () -> addNode(typeName));
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
            boolean isSelected = selectedConnections.contains(conn);
            double[] startPt = conn.source.getOutputPoint(conn.sourceOutputIndex);
            double[] endPt = conn.target.getInputPoint(conn.targetInputIndex);
            if (startPt != null && endPt != null) {
                NodeRenderer.renderConnection(gc, startPt[0], startPt[1], endPt[0], endPt[1], isSelected);
            }
        }

        // Draw in-progress connection
        if (isDrawingConnection && connectionSource != null) {
            double[] startPt = connectionSource.getOutputPoint(connectionOutputIndex);
            if (startPt != null) {
                gc.setStroke(Color.BLUE);
                gc.setLineWidth(2);
                gc.strokeLine(startPt[0], startPt[1], connectionEndX, connectionEndY);
            }
        }

        // Draw nodes
        for (FXNode node : nodes) {
            boolean isSelected = selectedNodes.contains(node);
            NodeRenderer.renderNode(gc, node.x, node.y, node.width, node.height,
                node.label, isSelected, node.enabled, node.backgroundColor,
                node.hasInput, node.hasDualInput, node.outputCount,
                node.thumbnail, node.isContainer, node.nodeType);
        }

        gc.restore();
    }
}
