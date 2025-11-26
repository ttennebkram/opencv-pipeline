package com.ttennebkram.pipeline.ui;

import org.eclipse.swt.SWT;
import org.eclipse.swt.custom.SashForm;
import org.eclipse.swt.custom.ScrolledComposite;
import org.eclipse.swt.events.*;
import org.eclipse.swt.graphics.*;
import org.eclipse.swt.layout.*;
import org.eclipse.swt.widgets.*;

import com.ttennebkram.pipeline.model.Connection;
import com.ttennebkram.pipeline.nodes.*;
import com.ttennebkram.pipeline.registry.NodeRegistry;
import com.ttennebkram.pipeline.serialization.PipelineSerializer;

import org.opencv.core.Mat;

import java.io.File;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Editor window for editing the internal pipeline of a ContainerNode.
 * Non-modal window that allows adding/connecting processing nodes inside the container.
 * Extends PipelineCanvasBase for shared canvas/editing functionality.
 */
public class ContainerEditorWindow extends PipelineCanvasBase {

    private ContainerNode container;
    private Shell parentShell;

    // Node lists reference the container's internal lists
    private List<PipelineNode> nodes;
    private List<Connection> connections;

    // Toolbar builder
    private ToolbarBuilder toolbarBuilder;

    // Callback when container is modified
    private Runnable onModified;

    // Callback to get base directory for relative paths
    private java.util.function.Supplier<String> basePathSupplier;

    // Callback to trigger global save
    private Runnable onRequestGlobalSave;

    // Pipeline control callbacks (shared with main editor)
    private Runnable onStartPipeline;
    private Runnable onStopPipeline;
    private java.util.function.Supplier<Boolean> isPipelineRunning;

    // Start/stop button (needs to update with pipeline state)
    private Button startStopBtn;

    public ContainerEditorWindow(Shell parentShell, Display display, ContainerNode container) {
        this.parentShell = parentShell;
        this.display = display;
        this.shell = null; // Will be created in createWindow()
        this.container = container;

        // Reference the container's internal lists
        this.nodes = container.getChildNodes();
        this.connections = container.getChildConnections();

        createWindow();
    }

    // ========== Abstract method implementations from PipelineCanvasBase ==========

    @Override
    protected List<PipelineNode> getNodes() {
        return nodes;
    }

    @Override
    protected List<Connection> getConnections() {
        return connections;
    }

    @Override
    protected void redrawCanvas() {
        if (canvas != null && !canvas.isDisposed()) {
            canvas.redraw();
        }
    }

    @Override
    protected void notifyModified() {
        if (onModified != null) {
            onModified.run();
        }
        updateNodeCount();
    }

    @Override
    protected void addNodeAt(String nodeType, int x, int y) {
        ProcessingNode node = NodeRegistry.createProcessingNode(nodeType, display, shell, x, y);
        if (node != null) {
            nodes.add(node);
            notifyModified();
            redrawCanvas();
        }
    }

    @Override
    protected void deleteSelected() {
        deleteSelectedImpl();
    }

    public void setOnModified(Runnable callback) {
        this.onModified = callback;
    }

    public void setBasePathSupplier(java.util.function.Supplier<String> supplier) {
        this.basePathSupplier = supplier;
    }

    public void setOnRequestGlobalSave(Runnable callback) {
        this.onRequestGlobalSave = callback;
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

    /**
     * Update the start/stop button to reflect the current pipeline state.
     * Call this from the main editor when pipeline state changes.
     */
    public void updatePipelineButtonState() {
        if (startStopBtn == null || startStopBtn.isDisposed()) {
            return;
        }

        boolean running = isPipelineRunning != null && isPipelineRunning.get();
        if (running) {
            startStopBtn.setText("Stop Pipeline");
            startStopBtn.setBackground(new Color(200, 100, 100)); // Red for stop
        } else {
            startStopBtn.setText("Start Pipeline");
            startStopBtn.setBackground(new Color(100, 180, 100)); // Green for start
        }
    }


    private String getBasePath() {
        if (basePathSupplier != null) {
            return basePathSupplier.get();
        }
        return null;
    }

    private void createWindow() {
        shell = new Shell(display, SWT.SHELL_TRIM);
        shell.setText("Container: " + container.getContainerName());
        shell.setSize(900, 600);

        // Same layout as main editor: GridLayout with toolbar + SashForm
        shell.setLayout(new GridLayout(2, false));

        // Left: Toolbar (same structure as main editor)
        createToolbar(shell);

        // Right: SashForm containing canvas and preview (resizable between them)
        SashForm rightSash = new SashForm(shell, SWT.HORIZONTAL);
        rightSash.setLayoutData(new GridData(SWT.FILL, SWT.FILL, true, true));

        // Canvas for internal pipeline
        createCanvas(rightSash);

        // Preview panel
        createPreviewPanel(rightSash);

        // Set sash weights: canvas | preview
        rightSash.setWeights(new int[]{700, 300});

        // Handle window close
        shell.addDisposeListener(e -> {
            if (previewImage != null && !previewImage.isDisposed()) {
                previewImage.dispose();
            }
        });

        // Start preview update timer (updates preview from selected node while pipeline is running)
        startPreviewTimer();

        // Add keyboard shortcuts
        shell.addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                // Cmd-S (Mac) or Ctrl-S (Windows/Linux)
                if (e.character == 's' && (e.stateMask & SWT.MOD1) != 0) {
                    saveContainerPipeline();
                    e.doit = false;
                }
                // Esc to close window
                else if (e.keyCode == SWT.ESC) {
                    shell.close();
                    e.doit = false;
                }
            }
        });

        // Center on parent
        if (parentShell != null) {
            Rectangle parentBounds = parentShell.getBounds();
            Rectangle shellBounds = shell.getBounds();
            int x = parentBounds.x + (parentBounds.width - shellBounds.width) / 2;
            int y = parentBounds.y + (parentBounds.height - shellBounds.height) / 2;
            shell.setLocation(x, y);
        }
    }

    private void createToolbar(Composite parent) {
        toolbarBuilder = new ToolbarBuilder(display, shell);
        toolbarBuilder.setAddFileSourceNode(() -> addFileSourceNode());
        toolbarBuilder.setAddWebcamSourceNode(() -> addWebcamSourceNode());
        toolbarBuilder.setAddBlankSourceNode(() -> addBlankSourceNode());
        toolbarBuilder.setAddNodeByType(typeName -> addNodeOfType(typeName));
        toolbarBuilder.build(parent);
    }

    private void addFileSourceNode() {
        int[] pos = getNewNodePosition();
        FileSourceNode node = new FileSourceNode(shell, display, canvas, pos[0], pos[1]);
        nodes.add(node);
        notifyModified();
        canvas.redraw();
    }

    private void addWebcamSourceNode() {
        int[] pos = getNewNodePosition();
        WebcamSourceNode node = new WebcamSourceNode(shell, display, canvas, pos[0], pos[1]);
        nodes.add(node);
        notifyModified();
        canvas.redraw();
    }

    private void addBlankSourceNode() {
        int[] pos = getNewNodePosition();
        BlankSourceNode node = new BlankSourceNode(shell, display, pos[0], pos[1]);
        nodes.add(node);
        notifyModified();
        canvas.redraw();
    }

    private int[] getNewNodePosition() {
        int newX = 300;
        int newY = 150;
        if (!nodes.isEmpty()) {
            newX = 200 + (nodes.size() * 30) % 300;
            newY = 100 + (nodes.size() * 30) % 200;
        }
        return new int[]{newX, newY};
    }

    private void addNodeOfType(String typeName) {
        int[] pos = getNewNodePosition();
        PipelineNode node = NodeRegistry.createNode(typeName, display, shell, pos[0], pos[1]);
        if (node != null) {
            nodes.add(node);
            notifyModified();
            canvas.redraw();
        }
    }

    private void createCanvas(Composite parent) {
        // Container for canvas + status bar
        Composite canvasContainer = new Composite(parent, SWT.NONE);
        canvasContainer.setLayout(new GridLayout(1, false));
        ((GridLayout)canvasContainer.getLayout()).marginHeight = 0;
        ((GridLayout)canvasContainer.getLayout()).marginWidth = 0;
        ((GridLayout)canvasContainer.getLayout()).verticalSpacing = 0;

        scrolledCanvas = new ScrolledComposite(canvasContainer, SWT.H_SCROLL | SWT.V_SCROLL | SWT.BORDER);
        scrolledCanvas.setLayoutData(new GridData(SWT.FILL, SWT.FILL, true, true));
        scrolledCanvas.setExpandHorizontal(true);
        scrolledCanvas.setExpandVertical(true);

        canvas = new Canvas(scrolledCanvas, SWT.DOUBLE_BUFFERED);
        canvas.setBackground(display.getSystemColor(SWT.COLOR_WHITE));

        scrolledCanvas.setContent(canvas);
        scrolledCanvas.setMinSize(1200, 800);

        // Mouse wheel zoom (Cmd/Ctrl + scroll) - stays centered on view
        scrolledCanvas.addListener(SWT.MouseVerticalWheel, e -> {
            if ((e.stateMask & SWT.MOD1) != 0) {
                // Cmd/Ctrl + wheel = zoom
                int direction = e.count > 0 ? 1 : -1;
                int currentIdx = zoomCombo.getSelectionIndex();
                int newIdx = currentIdx + direction;
                if (newIdx >= 0 && newIdx < ZOOM_LEVELS.length) {
                    zoomCombo.select(newIdx);
                    setZoomLevelCentered(ZOOM_LEVELS[newIdx] / 100.0);
                }
                e.doit = false;
            }
        });

        // Status bar at bottom of canvas
        Composite statusComp = new Composite(canvasContainer, SWT.NONE);
        statusComp.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));
        statusComp.setLayout(new GridLayout(2, false));
        ((GridLayout)statusComp.getLayout()).marginHeight = 2;
        ((GridLayout)statusComp.getLayout()).marginWidth = 5;
        statusComp.setBackground(new Color(160, 160, 160));

        // Node count on the left
        nodeCountLabel = new Label(statusComp, SWT.NONE);
        updateNodeCount();
        nodeCountLabel.setLayoutData(new GridData(SWT.LEFT, SWT.CENTER, true, false));
        nodeCountLabel.setBackground(new Color(160, 160, 160));
        nodeCountLabel.setForeground(display.getSystemColor(SWT.COLOR_BLACK));

        // Zoom combo on the right
        zoomCombo = new Combo(statusComp, SWT.DROP_DOWN | SWT.READ_ONLY);
        String[] zoomItems = new String[ZOOM_LEVELS.length];
        int defaultIndex = 3; // 100%
        for (int i = 0; i < ZOOM_LEVELS.length; i++) {
            zoomItems[i] = ZOOM_LEVELS[i] + "%";
            if (ZOOM_LEVELS[i] == 100) defaultIndex = i;
        }
        zoomCombo.setItems(zoomItems);
        zoomCombo.select(defaultIndex);
        zoomCombo.setBackground(display.getSystemColor(SWT.COLOR_LIST_BACKGROUND));
        zoomCombo.setForeground(display.getSystemColor(SWT.COLOR_LIST_FOREGROUND));
        GridData comboGd = new GridData(SWT.RIGHT, SWT.CENTER, false, false);
        comboGd.widthHint = 75;
        zoomCombo.setLayoutData(comboGd);
        zoomCombo.addListener(SWT.Selection, e -> {
            int idx = zoomCombo.getSelectionIndex();
            if (idx >= 0 && idx < ZOOM_LEVELS.length) {
                setZoomLevelCentered(ZOOM_LEVELS[idx] / 100.0);
            }
        });

        // Paint handler
        canvas.addPaintListener(e -> paintCanvas(e.gc));

        // Mouse handlers
        canvas.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseDown(MouseEvent e) {
                handleMouseDown(e);
            }

            @Override
            public void mouseUp(MouseEvent e) {
                handleMouseUp(e);
            }

            @Override
            public void mouseDoubleClick(MouseEvent e) {
                handleDoubleClick(e);
            }
        });

        canvas.addMouseMoveListener(e -> handleMouseMove(e));

        // Keyboard handler for delete and arrow keys
        canvas.addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                if (e.keyCode == SWT.DEL || e.keyCode == SWT.BS) {
                    deleteSelected();
                } else if (e.keyCode == SWT.ARROW_UP || e.keyCode == SWT.ARROW_DOWN ||
                           e.keyCode == SWT.ARROW_LEFT || e.keyCode == SWT.ARROW_RIGHT) {
                    moveSelectedNodes(e.keyCode);
                }
            }
        });
    }

    @Override
    protected void updateCanvasSize() {
        int baseWidth = 1200;
        int baseHeight = 800;
        int scaledWidth = (int) (baseWidth * zoomLevel);
        int scaledHeight = (int) (baseHeight * zoomLevel);
        canvas.setSize(scaledWidth, scaledHeight);
        scrolledCanvas.setMinSize(scaledWidth, scaledHeight);
    }

    @Override
    protected void updateNodeCount() {
        if (nodeCountLabel != null && !nodeCountLabel.isDisposed()) {
            // Count includes boundary nodes + child nodes
            int nodeCount = nodes.size() + 2; // +2 for boundary input and output
            int connectionCount = connections.size();
            nodeCountLabel.setText(String.format("Nodes: %d  Connections: %d", nodeCount, connectionCount));
        }
    }

    private void createPreviewPanel(Composite parent) {
        Composite previewPanel = new Composite(parent, SWT.BORDER);
        previewPanel.setLayout(new GridLayout(1, false));

        // Save and Cancel buttons at top
        Composite buttonPanel = new Composite(previewPanel, SWT.NONE);
        buttonPanel.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));
        buttonPanel.setLayout(new GridLayout(2, true));

        Button saveButton = new Button(buttonPanel, SWT.PUSH);
        saveButton.setText("Save");
        saveButton.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));
        saveButton.addListener(SWT.Selection, e -> saveContainerPipeline());

        Button cancelButton = new Button(buttonPanel, SWT.PUSH);
        cancelButton.setText("Cancel");
        cancelButton.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));
        cancelButton.addListener(SWT.Selection, e -> shell.close());

        // Start/Stop Pipeline button (shared with main editor)
        startStopBtn = new Button(previewPanel, SWT.PUSH);
        startStopBtn.setText("Start Pipeline");
        startStopBtn.setBackground(new Color(100, 180, 100)); // Green for start
        startStopBtn.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));
        startStopBtn.addListener(SWT.Selection, e -> {
            boolean running = isPipelineRunning != null && isPipelineRunning.get();
            if (running) {
                if (onStopPipeline != null) {
                    onStopPipeline.run();
                }
            } else {
                if (onStartPipeline != null) {
                    onStartPipeline.run();
                }
            }
        });

        // Separator
        Label separator = new Label(previewPanel, SWT.SEPARATOR | SWT.HORIZONTAL);
        separator.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        Label previewLabel = new Label(previewPanel, SWT.NONE);
        previewLabel.setText("Preview");
        previewLabel.setFont(new Font(display, "Arial", 10, SWT.BOLD));

        previewCanvas = new Canvas(previewPanel, SWT.BORDER | SWT.DOUBLE_BUFFERED);
        previewCanvas.setLayoutData(new GridData(SWT.FILL, SWT.FILL, true, true));
        previewCanvas.setBackground(display.getSystemColor(SWT.COLOR_BLACK));

        previewCanvas.addPaintListener(e -> {
            if (previewImage != null && !previewImage.isDisposed()) {
                Rectangle bounds = previewCanvas.getClientArea();
                Rectangle imgBounds = previewImage.getBounds();
                // Scale to fit
                double scale = Math.min((double) bounds.width / imgBounds.width,
                                       (double) bounds.height / imgBounds.height);
                int w = (int) (imgBounds.width * scale);
                int h = (int) (imgBounds.height * scale);
                int x = (bounds.width - w) / 2;
                int y = (bounds.height - h) / 2;
                e.gc.drawImage(previewImage, 0, 0, imgBounds.width, imgBounds.height, x, y, w, h);
            }
        });
    }

    private void paintCanvas(GC gc) {
        gc.setAntialias(SWT.ON);

        // Draw grid
        Rectangle bounds = canvas.getClientArea();
        int gridSize = 20;
        int scaledGrid = (int) Math.round(gridSize * zoomLevel);
        if (scaledGrid < 1) scaledGrid = 1;
        gc.setForeground(new Color(230, 230, 230));
        gc.setLineWidth(1);
        for (int x = 0; x <= bounds.width; x += scaledGrid) {
            gc.drawLine(x, 0, x, bounds.height);
        }
        for (int y = 0; y <= bounds.height; y += scaledGrid) {
            gc.drawLine(0, y, bounds.width, y);
        }

        // Apply zoom transform
        Transform transform = new Transform(display);
        transform.scale((float) zoomLevel, (float) zoomLevel);
        gc.setTransform(transform);

        // Draw connections with bezier curves (same style as main editor)
        for (Connection conn : connections) {
            Point start = conn.source.getOutputPoint(conn.outputIndex);
            Point end = getConnectionTargetPoint(conn);

            if (selectedConnections.contains(conn)) {
                gc.setLineWidth(3);
                gc.setForeground(display.getSystemColor(SWT.COLOR_CYAN));
            } else {
                gc.setLineWidth(2);
                gc.setForeground(display.getSystemColor(SWT.COLOR_DARK_GRAY));
            }

            // Calculate bezier control points (same algorithm as main editor)
            int[] cp = calculateBezierControlPoints(start, end);

            Path path = new Path(display);
            path.moveTo(start.x, start.y);
            path.cubicTo(cp[0], cp[1], cp[2], cp[3], end.x, end.y);
            gc.drawPath(path);
            path.dispose();

            // Draw arrow from second control point direction
            drawArrow(gc, new Point(cp[2], cp[3]), end);

            // Draw queue stats (uses shared method with backpressure coloring)
            drawQueueStats(gc, conn, start, end, cp);
        }

        // Draw connection being created
        if (connectionSource != null && connectionEndPoint != null) {
            Point start = connectionSource.getOutputPoint(connectionSourceOutputIndex);
            gc.setLineWidth(2);
            gc.setForeground(display.getSystemColor(SWT.COLOR_BLUE));
            gc.drawLine(start.x, start.y, connectionEndPoint.x, connectionEndPoint.y);
        }

        // Draw boundary nodes first (they're fixed)
        ContainerInputNode boundaryInput = container.getBoundaryInput();
        ContainerOutputNode boundaryOutput = container.getBoundaryOutput();

        boundaryInput.paint(gc);
        if (selectedNodes.contains(boundaryInput) || boundaryInput == selectedNode) {
            boundaryInput.drawSelectionHighlight(gc, true);
        }
        boundaryOutput.paint(gc);
        if (selectedNodes.contains(boundaryOutput) || boundaryOutput == selectedNode) {
            boundaryOutput.drawSelectionHighlight(gc, true);
        }

        // Draw child nodes
        for (PipelineNode node : nodes) {
            node.paint(gc);
            if (selectedNodes.contains(node) || node == selectedNode) {
                node.drawSelectionHighlight(gc, true);
            }
        }

        // Draw selection box if dragging
        if (isSelectionBoxDragging && selectionBoxStart != null && selectionBoxEnd != null) {
            int boxX = Math.min(selectionBoxStart.x, selectionBoxEnd.x);
            int boxY = Math.min(selectionBoxStart.y, selectionBoxEnd.y);
            int boxWidth = Math.abs(selectionBoxEnd.x - selectionBoxStart.x);
            int boxHeight = Math.abs(selectionBoxEnd.y - selectionBoxStart.y);

            // Draw selection box with semi-transparent fill
            gc.setBackground(new Color(0, 120, 215));
            gc.setAlpha(30);
            gc.fillRectangle(boxX, boxY, boxWidth, boxHeight);
            gc.setAlpha(255);

            // Draw selection box border
            gc.setForeground(new Color(0, 120, 215));
            gc.setLineWidth(1);
            gc.drawRectangle(boxX, boxY, boxWidth, boxHeight);
        }

        transform.dispose();
    }

    private void handleMouseDown(MouseEvent e) {
        Point click = toCanvasPoint(e.x, e.y);
        canvas.setFocus();

        ContainerInputNode boundaryInput = container.getBoundaryInput();
        ContainerOutputNode boundaryOutput = container.getBoundaryOutput();

        // First: Check output points for starting connections (before node body)
        // Check boundary input's output point first
        int outputIdx = boundaryInput.getOutputIndexNear(click.x, click.y);
        if (outputIdx >= 0) {
            connectionSource = boundaryInput;
            connectionSourceOutputIndex = outputIdx;
            connectionEndPoint = click;
            return;
        }

        // Check child nodes' output points
        for (int i = nodes.size() - 1; i >= 0; i--) {
            PipelineNode node = nodes.get(i);
            outputIdx = node.getOutputIndexNear(click.x, click.y);
            if (outputIdx >= 0) {
                connectionSource = node;
                connectionSourceOutputIndex = outputIdx;
                connectionEndPoint = click;
                return;
            }
        }

        // Second: Check input points - if connected, yank the connection to allow reconnecting
        // Check boundary output's input point
        if (boundaryOutput.isNearInputPoint(click.x, click.y)) {
            Connection toYank = findConnectionToTarget(boundaryOutput, 1);
            if (toYank != null) {
                connectionSource = toYank.source;
                connectionSourceOutputIndex = toYank.outputIndex;
                yankedOriginalTarget = toYank.target;
                yankedOriginalInputIndex = toYank.inputIndex;
                connectionEndPoint = click;
                connections.remove(toYank);
                notifyModified();
                canvas.redraw();
                return;
            }
        }

        // Check child nodes' input points
        for (PipelineNode node : nodes) {
            if (node.isNearInputPoint(click.x, click.y)) {
                Connection toYank = findConnectionToTarget(node, 1);
                if (toYank != null) {
                    connectionSource = toYank.source;
                    connectionSourceOutputIndex = toYank.outputIndex;
                    yankedOriginalTarget = toYank.target;
                    yankedOriginalInputIndex = toYank.inputIndex;
                    connectionEndPoint = click;
                    connections.remove(toYank);
                    notifyModified();
                    canvas.redraw();
                    return;
                }
            }
            if (node.hasDualInput() && node.isNearInputPoint2(click.x, click.y)) {
                Connection toYank = findConnectionToTarget(node, 2);
                if (toYank != null) {
                    connectionSource = toYank.source;
                    connectionSourceOutputIndex = toYank.outputIndex;
                    yankedOriginalTarget = toYank.target;
                    yankedOriginalInputIndex = toYank.inputIndex;
                    connectionEndPoint = click;
                    connections.remove(toYank);
                    notifyModified();
                    canvas.redraw();
                    return;
                }
            }
        }

        // Third: Check for node body clicks (for selection and dragging)
        for (int i = nodes.size() - 1; i >= 0; i--) {
            PipelineNode node = nodes.get(i);
            if (node.containsPoint(click)) {
                // If node is not already selected, clear selection and select only this node
                // If node IS already selected, keep current selection (for dragging multiple)
                if (!selectedNodes.contains(node)) {
                    selectedNodes.clear();
                    selectedConnections.clear();
                    selectedNodes.add(node);
                }
                selectedNode = node;
                dragOffset = new Point(click.x - node.getX(), click.y - node.getY());
                isDragging = true;
                updatePreviewFromNode(node);
                canvas.redraw();
                return;
            }
        }

        // Check boundary input body (can be dragged)
        if (boundaryInput.containsPoint(click)) {
            if (!selectedNodes.contains(boundaryInput)) {
                selectedNodes.clear();
                selectedConnections.clear();
                selectedNodes.add(boundaryInput);
            }
            selectedNode = boundaryInput;
            dragOffset = new Point(click.x - boundaryInput.getX(), click.y - boundaryInput.getY());
            isDragging = true;
            updatePreviewFromNode(boundaryInput);
            canvas.redraw();
            return;
        }

        // Check boundary output body (can be dragged)
        if (boundaryOutput.containsPoint(click)) {
            if (!selectedNodes.contains(boundaryOutput)) {
                selectedNodes.clear();
                selectedConnections.clear();
                selectedNodes.add(boundaryOutput);
            }
            selectedNode = boundaryOutput;
            dragOffset = new Point(click.x - boundaryOutput.getX(), click.y - boundaryOutput.getY());
            isDragging = true;
            updatePreviewFromNode(boundaryOutput);
            canvas.redraw();
            return;
        }

        // Check for connection line selection (clicking on the line itself)
        for (Connection conn : connections) {
            if (isNearConnectionLine(conn, click)) {
                selectedNodes.clear();
                selectedConnections.clear();
                selectedConnections.add(conn);
                selectedNode = null;
                canvas.redraw();
                return;
            }
        }

        // Clicked on empty space - start selection box
        selectedNode = null;
        selectedNodes.clear();
        selectedConnections.clear();
        selectionBoxStart = click;
        selectionBoxEnd = click;
        isSelectionBoxDragging = true;
        canvas.redraw();
    }

    /**
     * Find a connection originating from the given source node's output.
     */
    private Connection findConnectionFromSource(PipelineNode source, int outputIndex) {
        for (Connection conn : connections) {
            if (conn.source == source && conn.outputIndex == outputIndex) {
                return conn;
            }
        }
        return null;
    }

    private void handleMouseMove(MouseEvent e) {
        Point click = toCanvasPoint(e.x, e.y);

        // Check for tooltip on connection points
        updateConnectionTooltip(click.x, click.y);

        // Update connection endpoint
        if (connectionSource != null) {
            connectionEndPoint = click;
            canvas.redraw();
            return;
        }

        // Drag selected node(s) (including boundary nodes)
        if (isDragging && selectedNode != null && dragOffset != null) {
            // Calculate the delta movement
            int deltaX = click.x - dragOffset.x - selectedNode.getX();
            int deltaY = click.y - dragOffset.y - selectedNode.getY();

            // Move all selected nodes by the same delta if dragging a multi-selection
            if (selectedNodes.contains(selectedNode) && selectedNodes.size() > 1) {
                for (PipelineNode node : selectedNodes) {
                    node.setX(node.getX() + deltaX);
                    node.setY(node.getY() + deltaY);
                }
            } else {
                // Single node drag
                selectedNode.setX(click.x - dragOffset.x);
                selectedNode.setY(click.y - dragOffset.y);
            }
            notifyModified();
            canvas.redraw();
        }

        // Update selection box
        if (isSelectionBoxDragging) {
            selectionBoxEnd = click;
            canvas.redraw();
        }
    }

    /**
     * Update canvas tooltip based on mouse position over connection points.
     */
    private void updateConnectionTooltip(int canvasX, int canvasY) {
        String tooltip = null;

        ContainerInputNode boundaryInput = container.getBoundaryInput();
        ContainerOutputNode boundaryOutput = container.getBoundaryOutput();

        // Check boundary input's output point
        int outputIndex = boundaryInput.getOutputIndexNear(canvasX, canvasY);
        if (outputIndex >= 0) {
            tooltip = boundaryInput.getOutputTooltip(outputIndex);
        }

        // Check boundary output's input point
        if (tooltip == null && boundaryOutput.isNearInputPoint(canvasX, canvasY)) {
            tooltip = boundaryOutput.getInputTooltip();
        }

        // Check all child nodes for connection point hover
        if (tooltip == null) {
            for (PipelineNode node : nodes) {
                // Check output points
                outputIndex = node.getOutputIndexNear(canvasX, canvasY);
                if (outputIndex >= 0) {
                    tooltip = node.getOutputTooltip(outputIndex);
                    break;
                }

                // Check primary input point
                if (node.isNearInputPoint(canvasX, canvasY)) {
                    tooltip = node.getInputTooltip();
                    break;
                }

                // Check secondary input point (dual-input nodes)
                if (node.isNearInputPoint2(canvasX, canvasY)) {
                    tooltip = node.getInput2Tooltip();
                    break;
                }
            }
        }

        // Update canvas tooltip
        String currentTooltip = canvas.getToolTipText();
        if (tooltip == null) {
            if (currentTooltip != null) {
                canvas.setToolTipText(null);
            }
        } else {
            if (!tooltip.equals(currentTooltip)) {
                canvas.setToolTipText(tooltip);
            }
        }
    }

    private void handleMouseUp(MouseEvent e) {
        Point click = toCanvasPoint(e.x, e.y);

        // Complete connection if drawing one
        if (connectionSource != null && connectionEndPoint != null) {
            boolean connectionMade = false;

            // Check if dropping on boundary output's input
            ContainerOutputNode boundaryOutput = container.getBoundaryOutput();
            if (boundaryOutput.isNearInputPoint(click.x, click.y)) {
                createConnection(connectionSource, boundaryOutput, 1, connectionSourceOutputIndex);
                connectionMade = true;
            } else {
                // Check child nodes
                for (PipelineNode node : nodes) {
                    if (node.isNearInputPoint(click.x, click.y)) {
                        createConnection(connectionSource, node, 1, connectionSourceOutputIndex);
                        connectionMade = true;
                        break;
                    }
                    if (node.hasDualInput() && node.isNearInputPoint2(click.x, click.y)) {
                        createConnection(connectionSource, node, 2, connectionSourceOutputIndex);
                        connectionMade = true;
                        break;
                    }
                }
            }

            // If this was a yanked connection and no new connection was made, restore original
            if (!connectionMade && yankedOriginalTarget != null) {
                createConnection(connectionSource, yankedOriginalTarget, yankedOriginalInputIndex, connectionSourceOutputIndex);
            }

            connectionSource = null;
            connectionEndPoint = null;
            yankedOriginalTarget = null;
            yankedOriginalInputIndex = 0;
            canvas.redraw();
        }

        // Handle selection box completion
        if (isSelectionBoxDragging && selectionBoxStart != null && selectionBoxEnd != null) {
            // Calculate box bounds
            int boxX = Math.min(selectionBoxStart.x, selectionBoxEnd.x);
            int boxY = Math.min(selectionBoxStart.y, selectionBoxEnd.y);
            int boxWidth = Math.abs(selectionBoxEnd.x - selectionBoxStart.x);
            int boxHeight = Math.abs(selectionBoxEnd.y - selectionBoxStart.y);

            // Select child nodes that are completely inside the selection box
            for (PipelineNode node : nodes) {
                if (node.getX() >= boxX && node.getY() >= boxY &&
                    node.getX() + node.getWidth() <= boxX + boxWidth &&
                    node.getY() + node.getHeight() <= boxY + boxHeight) {
                    selectedNodes.add(node);
                }
            }

            // Also check boundary nodes
            ContainerInputNode boundaryInput = container.getBoundaryInput();
            ContainerOutputNode boundaryOutput = container.getBoundaryOutput();

            if (boundaryInput.getX() >= boxX && boundaryInput.getY() >= boxY &&
                boundaryInput.getX() + boundaryInput.getWidth() <= boxX + boxWidth &&
                boundaryInput.getY() + boundaryInput.getHeight() <= boxY + boxHeight) {
                selectedNodes.add(boundaryInput);
            }

            if (boundaryOutput.getX() >= boxX && boundaryOutput.getY() >= boxY &&
                boundaryOutput.getX() + boundaryOutput.getWidth() <= boxX + boxWidth &&
                boundaryOutput.getY() + boundaryOutput.getHeight() <= boxY + boxHeight) {
                selectedNodes.add(boundaryOutput);
            }

            // Select connections that are completely inside selection box
            for (Connection conn : connections) {
                Point start = conn.source.getOutputPoint(conn.outputIndex);
                Point end = getConnectionTargetPoint(conn);
                if (start.x >= boxX && start.x <= boxX + boxWidth &&
                    start.y >= boxY && start.y <= boxY + boxHeight &&
                    end.x >= boxX && end.x <= boxX + boxWidth &&
                    end.y >= boxY && end.y <= boxY + boxHeight) {
                    selectedConnections.add(conn);
                }
            }

            // Clear selection box state
            selectionBoxStart = null;
            selectionBoxEnd = null;
            isSelectionBoxDragging = false;
            canvas.redraw();
        }

        isDragging = false;
        dragOffset = null;
    }

    private void createConnection(PipelineNode source, PipelineNode target, int inputIndex, int outputIndex) {
        // Don't connect to self
        if (source == target) {
            return;
        }

        // Check if connection already exists
        for (Connection conn : connections) {
            if (conn.source == source && conn.target == target &&
                conn.inputIndex == inputIndex && conn.outputIndex == outputIndex) {
                return;
            }
        }

        Connection conn = new Connection(source, target, inputIndex, outputIndex);
        connections.add(conn);
        notifyModified();
    }

    private void handleDoubleClick(MouseEvent e) {
        Point click = toCanvasPoint(e.x, e.y);

        for (PipelineNode node : nodes) {
            if (node.containsPoint(click)) {
                if (node instanceof ProcessingNode) {
                    // Temporarily set node's shell to this container editor shell
                    // so the dialog appears as a child of this window
                    ProcessingNode pn = (ProcessingNode) node;
                    Shell originalShell = pn.getShell();
                    pn.setShell(shell);
                    pn.showPropertiesDialog();
                    pn.setShell(originalShell);
                }
                return;
            }
        }
    }

    private void deleteSelectedImpl() {
        if (selectedNode != null) {
            // Don't delete boundary nodes
            if (selectedNode instanceof ContainerInputNode || selectedNode instanceof ContainerOutputNode) {
                return;
            }

            // Remove connections to/from this node
            connections.removeIf(conn -> conn.source == selectedNode || conn.target == selectedNode);

            // Remove node
            nodes.remove(selectedNode);
            selectedNode = null;
            selectedNodes.clear();
            notifyModified();
            canvas.redraw();
        }

        // Remove selected connections
        connections.removeAll(selectedConnections);
        selectedConnections.clear();
        notifyModified();
        canvas.redraw();
    }

    @Override
    protected void updatePreviewFromNode(PipelineNode node) {
        Mat outputMat = node.getOutputMat();
        if (outputMat != null && !outputMat.empty()) {
            // Convert to SWT Image
            Mat rgb = new Mat();
            if (outputMat.channels() == 3) {
                Imgproc.cvtColor(outputMat, rgb, Imgproc.COLOR_BGR2RGB);
            } else if (outputMat.channels() == 1) {
                Imgproc.cvtColor(outputMat, rgb, Imgproc.COLOR_GRAY2RGB);
            } else {
                rgb = outputMat.clone();
            }

            int w = rgb.width();
            int h = rgb.height();
            byte[] data = new byte[w * h * 3];
            rgb.get(0, 0, data);

            PaletteData palette = new PaletteData(0xFF0000, 0x00FF00, 0x0000FF);
            ImageData imageData = new ImageData(w, h, 24, palette);

            int bytesPerLine = imageData.bytesPerLine;
            for (int row = 0; row < h; row++) {
                int srcOffset = row * w * 3;
                int dstOffset = row * bytesPerLine;
                for (int col = 0; col < w; col++) {
                    int srcIdx = srcOffset + col * 3;
                    int dstIdx = dstOffset + col * 3;
                    imageData.data[dstIdx] = data[srcIdx];
                    imageData.data[dstIdx + 1] = data[srcIdx + 1];
                    imageData.data[dstIdx + 2] = data[srcIdx + 2];
                }
            }

            if (previewImage != null && !previewImage.isDisposed()) {
                previewImage.dispose();
            }
            previewImage = new Image(display, imageData);
            rgb.release();

            previewCanvas.redraw();
        }
    }

    /**
     * Start a timer that periodically updates the preview from the selected node.
     * This allows the preview to update while the pipeline is running.
     */
    private void startPreviewTimer() {
        final int PREVIEW_UPDATE_INTERVAL = 100; // milliseconds

        Runnable timerRunnable = new Runnable() {
            @Override
            public void run() {
                if (shell == null || shell.isDisposed()) {
                    return;
                }

                // Update preview from selected node if one is selected
                if (selectedNode != null) {
                    updatePreviewFromNode(selectedNode);
                } else if (selectedNodes.size() == 1) {
                    // Single node in selection set
                    PipelineNode node = selectedNodes.iterator().next();
                    updatePreviewFromNode(node);
                } else {
                    // No selection - show output boundary node's preview if available
                    ContainerOutputNode boundaryOutput = container.getBoundaryOutput();
                    if (boundaryOutput != null) {
                        updatePreviewFromNode(boundaryOutput);
                    }
                }

                // Also redraw the canvas to update node thumbnails
                if (canvas != null && !canvas.isDisposed()) {
                    canvas.redraw();
                }

                // Reschedule the timer
                if (!shell.isDisposed()) {
                    display.timerExec(PREVIEW_UPDATE_INTERVAL, this);
                }
            }
        };

        // Start the timer
        display.timerExec(PREVIEW_UPDATE_INTERVAL, timerRunnable);
    }

    public void open() {
        shell.open();
    }

    public boolean isDisposed() {
        return shell == null || shell.isDisposed();
    }

    public void close() {
        if (shell != null && !shell.isDisposed()) {
            shell.close();
        }
    }

    public Shell getShell() {
        return shell;
    }

    public ContainerNode getContainer() {
        return container;
    }

    /**
     * Refresh the canvas display (call when container's internal state changes).
     */
    public void refresh() {
        if (canvas != null && !canvas.isDisposed()) {
            canvas.redraw();
        }
    }

    /**
     * Save the container's internal pipeline to its file.
     * If no file is set, prompts for a file name.
     */
    public void saveContainerPipeline() {
        String filePath = container.getPipelineFilePath();

        if (filePath == null || filePath.isEmpty()) {
            // Prompt for file name
            FileDialog dialog = new FileDialog(shell, SWT.SAVE);
            dialog.setText("Save Container Pipeline");
            dialog.setFilterExtensions(new String[]{"*.json"});
            dialog.setFilterNames(new String[]{"Pipeline Files (*.json)"});

            // Suggest a filename based on container name
            String suggestedName = container.getContainerName().replaceAll("[^a-zA-Z0-9_-]", "_") + "_pipeline.json";
            dialog.setFileName(suggestedName);

            // Start in same directory as parent pipeline if available
            String basePath = getBasePath();
            if (basePath != null) {
                File baseFile = new File(basePath);
                if (baseFile.getParentFile() != null) {
                    dialog.setFilterPath(baseFile.getParentFile().getAbsolutePath());
                }
            }

            filePath = dialog.open();
            if (filePath == null) {
                return; // User cancelled
            }

            if (!filePath.toLowerCase().endsWith(".json")) {
                filePath += ".json";
            }

            // Store as relative path if possible
            String relativePath = makeRelativePath(filePath);
            container.setPipelineFilePath(relativePath);
            notifyModified(); // Parent needs to save the reference
        } else {
            // Resolve relative path to absolute
            filePath = resolveFilePath(filePath);
        }

        // Save the internal pipeline
        saveToFile(filePath);
    }

    /**
     * Save the internal pipeline to the specified file.
     */
    private void saveToFile(String filePath) {
        try {
            // Build list of all nodes including boundary nodes
            List<PipelineNode> allNodes = new ArrayList<>();
            allNodes.add(container.getBoundaryInput());
            allNodes.addAll(nodes);
            allNodes.add(container.getBoundaryOutput());

            // Use PipelineSerializer to save
            PipelineSerializer.save(filePath, allNodes, connections,
                new java.util.ArrayList<>(), new java.util.ArrayList<>(), new java.util.ArrayList<>());

            shell.setText("Container: " + container.getContainerName());
            System.out.println("Saved container pipeline to: " + filePath);

            // Also trigger the parent pipeline save to persist the container's file reference
            if (onRequestGlobalSave != null) {
                onRequestGlobalSave.run();
            }

            // Redraw the canvas to ensure display is up-to-date
            canvas.redraw();

        } catch (Exception e) {
            MessageBox error = new MessageBox(shell, SWT.ICON_ERROR | SWT.OK);
            error.setText("Save Error");
            error.setMessage("Failed to save container pipeline:\n" + e.getMessage());
            error.open();
        }
    }

    /**
     * Make a path relative to the parent pipeline's directory.
     */
    private String makeRelativePath(String absolutePath) {
        String basePath = getBasePath();
        if (basePath == null) {
            return absolutePath; // Can't make relative without base
        }

        File baseFile = new File(basePath);
        File baseDir = baseFile.getParentFile();
        if (baseDir == null) {
            return absolutePath;
        }

        File targetFile = new File(absolutePath);
        String baseDirPath = baseDir.getAbsolutePath();
        String targetPath = targetFile.getAbsolutePath();

        // If target is in same directory or subdirectory, make relative
        if (targetPath.startsWith(baseDirPath)) {
            String relative = targetPath.substring(baseDirPath.length());
            if (relative.startsWith(File.separator)) {
                relative = relative.substring(1);
            }
            return relative;
        }

        return absolutePath; // Can't make relative, use absolute
    }

    /**
     * Resolve a potentially relative path to absolute.
     */
    private String resolveFilePath(String path) {
        File file = new File(path);
        if (file.isAbsolute()) {
            return path;
        }

        String basePath = getBasePath();
        if (basePath != null) {
            File baseFile = new File(basePath);
            File baseDir = baseFile.getParentFile();
            if (baseDir != null) {
                return new File(baseDir, path).getAbsolutePath();
            }
        }

        return path;
    }

    /**
     * Get the absolute file path for this container's pipeline.
     */
    public String getAbsoluteFilePath() {
        String path = container.getPipelineFilePath();
        if (path == null || path.isEmpty()) {
            return null;
        }
        return resolveFilePath(path);
    }
}
