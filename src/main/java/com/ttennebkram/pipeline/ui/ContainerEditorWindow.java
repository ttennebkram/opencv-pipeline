package com.ttennebkram.pipeline.ui;

import org.eclipse.swt.SWT;
import org.eclipse.swt.custom.SashForm;
import org.eclipse.swt.custom.ScrolledComposite;
import org.eclipse.swt.events.*;
import org.eclipse.swt.graphics.*;
import org.eclipse.swt.layout.*;
import org.eclipse.swt.widgets.*;

import com.ttennebkram.pipeline.model.Connection;
import com.ttennebkram.pipeline.model.DanglingConnection;
import com.ttennebkram.pipeline.model.FreeConnection;
import com.ttennebkram.pipeline.model.ReverseDanglingConnection;
import com.ttennebkram.pipeline.nodes.*;
import com.ttennebkram.pipeline.registry.NodeRegistry;
import com.ttennebkram.pipeline.serialization.PipelineSerializer;
import com.ttennebkram.pipeline.ui.HelpBrowser;

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

    // =========================== COLOR CONSTANTS ===========================
    // Status bar colors
    private static final int[] COLOR_STATUS_BAR_BG = {160, 160, 160};        // Gray status bar background
    private static final int[] COLOR_STATUS_STOPPED = {180, 0, 0};           // Red text for stopped
    private static final int[] COLOR_STATUS_RUNNING = {0, 128, 0};           // Green text for running

    // Pipeline control button colors
    private static final int[] COLOR_START_BUTTON = {100, 180, 100};         // Green start button
    private static final int[] COLOR_STOP_BUTTON = {200, 100, 100};          // Red stop button

    // Canvas colors
    private static final int[] COLOR_GRID_LINES = {230, 230, 230};           // Light gray grid
    private static final int[] COLOR_SELECTION_BOX = {0, 120, 215};          // Blue selection box
    // ========================================================================

    private ContainerNode container;
    private Shell parentShell;

    // Node lists reference the container's internal lists
    private List<PipelineNode> nodes;
    private List<Connection> connections;

    // Dangling connections (one end free)
    private List<DanglingConnection> danglingConnections = new ArrayList<>();
    private List<ReverseDanglingConnection> reverseDanglingConnections = new ArrayList<>();
    private List<FreeConnection> freeConnections = new ArrayList<>();

    // Selection sets for dangling connections
    private Set<DanglingConnection> selectedDanglingConnections = new HashSet<>();
    private Set<ReverseDanglingConnection> selectedReverseDanglingConnections = new HashSet<>();
    private Set<FreeConnection> selectedFreeConnections = new HashSet<>();

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

    // Callback to open nested container editor (from parent - used for window tracking)
    private java.util.function.Consumer<ContainerNode> onOpenNestedContainer;

    // List of nested container editor windows opened from this window
    private List<ContainerEditorWindow> nestedContainerWindows = new java.util.ArrayList<>();

    // Start/stop button (needs to update with pipeline state)
    private Button startStopBtn;

    // Pipeline status label
    private Label pipelineStatusLabel;

    public ContainerEditorWindow(Shell parentShell, Display display, ContainerNode container) {
        this.parentShell = parentShell;
        this.display = display;
        this.shell = null; // Will be created in createWindow()
        this.container = container;

        // Reference the container's internal lists
        this.nodes = container.getChildNodes();
        this.connections = container.getChildConnections();

        createWindow();

        // Wire up change callbacks for existing nodes
        for (PipelineNode node : nodes) {
            if (node instanceof ProcessingNode) {
                ((ProcessingNode) node).setOnChanged(() -> { notifyModified(); canvas.redraw(); });
            }
        }
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
            node.setOnChanged(() -> { notifyModified(); canvas.redraw(); });
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

    public void setOnOpenNestedContainer(java.util.function.Consumer<ContainerNode> callback) {
        this.onOpenNestedContainer = callback;
    }

    /**
     * Open a nested container editor from within this container.
     * Sets up the proper base path (this container's file) and save callbacks.
     */
    private void openNestedContainerEditor(ContainerNode nestedContainer) {
        // Check if already open
        for (ContainerEditorWindow window : nestedContainerWindows) {
            if (!window.isDisposed() && window.getContainer() == nestedContainer) {
                window.getShell().setActive();
                window.getShell().setFocus();
                return;
            }
        }

        // Clean up disposed windows
        nestedContainerWindows.removeIf(ContainerEditorWindow::isDisposed);

        // Create new window with this container's shell as parent
        ContainerEditorWindow window = new ContainerEditorWindow(shell, display, nestedContainer);
        window.setOnModified(() -> {
            notifyModified();
            canvas.redraw();
        });

        // Set base path to THIS container's file (not the main pipeline)
        window.setBasePathSupplier(() -> getAbsoluteFilePath());

        // When nested container requests save, save THIS container's pipeline
        window.setOnRequestGlobalSave(() -> saveContainerPipeline());

        // Wire up pipeline control callbacks (pass through to parent)
        window.setOnStartPipeline(onStartPipeline);
        window.setOnStopPipeline(onStopPipeline);
        window.setIsPipelineRunning(isPipelineRunning);

        // Recursively wire up for further nesting
        window.setOnOpenNestedContainer(deeplyNested -> window.openNestedContainerEditor(deeplyNested));

        nestedContainerWindows.add(window);
        window.open();
        window.updatePipelineButtonState();

        // Also notify the parent editor's callback for window tracking
        if (onOpenNestedContainer != null) {
            onOpenNestedContainer.accept(nestedContainer);
        }
    }

    /**
     * Update the start/stop button to reflect the current pipeline state.
     * Call this from the main editor when pipeline state changes.
     */
    public void updatePipelineButtonState() {
        boolean running = isPipelineRunning != null && isPipelineRunning.get();

        // Update start/stop button
        if (startStopBtn != null && !startStopBtn.isDisposed()) {
            if (running) {
                startStopBtn.setText("Stop Pipeline");
                startStopBtn.setBackground(new Color(COLOR_STOP_BUTTON[0], COLOR_STOP_BUTTON[1], COLOR_STOP_BUTTON[2]));
            } else {
                startStopBtn.setText("Start Pipeline");
                startStopBtn.setBackground(new Color(COLOR_START_BUTTON[0], COLOR_START_BUTTON[1], COLOR_START_BUTTON[2]));
            }
        }

        // Update pipeline status label with thread count
        if (pipelineStatusLabel != null && !pipelineStatusLabel.isDisposed()) {
            if (running) {
                // Show immediate status with thread count
                int threadCount = countThreadsForContainer();
                String text = "Pipeline Running (" + threadCount + " threads)";
                pipelineStatusLabel.setText(text);
                pipelineStatusLabel.setForeground(new Color(COLOR_STATUS_RUNNING[0], COLOR_STATUS_RUNNING[1], COLOR_STATUS_RUNNING[2]));

                // Schedule a delayed update to get accurate thread count after startup
                display.timerExec(200, () -> {
                    if (pipelineStatusLabel != null && !pipelineStatusLabel.isDisposed()) {
                        boolean stillRunning = isPipelineRunning != null && isPipelineRunning.get();
                        if (stillRunning) {
                            int updatedCount = countThreadsForContainer();
                            String updatedText = "Pipeline Running (" + updatedCount + " threads)";
                            pipelineStatusLabel.setText(updatedText);
                        }
                    }
                });
            } else {
                pipelineStatusLabel.setText("Pipeline Stopped");
                pipelineStatusLabel.setForeground(new Color(COLOR_STATUS_STOPPED[0], COLOR_STATUS_STOPPED[1], COLOR_STATUS_STOPPED[2]));
            }
        }
    }

    /**
     * Count active threads for this container and all nested containers.
     * Counts: boundary input, child nodes, boundary output, plus nested container threads.
     */
    private int countThreadsForContainer() {
        int count = 0;

        // Count boundary input
        if (container.getBoundaryInput() != null && container.getBoundaryInput().hasActiveThread()) {
            count++;
        }

        // Count child nodes and recurse into nested containers
        for (PipelineNode node : nodes) {
            if (node.hasActiveThread()) {
                count++;
            }
            // Recurse into nested containers
            if (node instanceof ContainerNode) {
                count += countThreadsForNestedContainer((ContainerNode) node);
            }
        }

        // Count boundary output
        if (container.getBoundaryOutput() != null && container.getBoundaryOutput().hasActiveThread()) {
            count++;
        }

        System.out.println("[" + PipelineNode.timestamp() + "] Container " + container.getContainerName() +
            " thread count: " + count + " (boundaryIn=" + (container.getBoundaryInput() != null && container.getBoundaryInput().hasActiveThread()) +
            ", childNodes=" + nodes.size() +
            ", boundaryOut=" + (container.getBoundaryOutput() != null && container.getBoundaryOutput().hasActiveThread()) + ")");

        return count;
    }

    /**
     * Recursively count threads in a nested container.
     */
    private int countThreadsForNestedContainer(ContainerNode nested) {
        int count = 0;

        // Count boundary nodes
        if (nested.getBoundaryInput() != null && nested.getBoundaryInput().hasActiveThread()) {
            count++;
        }
        if (nested.getBoundaryOutput() != null && nested.getBoundaryOutput().hasActiveThread()) {
            count++;
        }

        // Count child nodes and recurse
        for (PipelineNode child : nested.getChildNodes()) {
            if (child.hasActiveThread()) {
                count++;
            }
            if (child instanceof ContainerNode) {
                count += countThreadsForNestedContainer((ContainerNode) child);
            }
        }

        return count;
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

        // Size to match parent window, clamped to fit on screen
        Rectangle screenBounds = display.getPrimaryMonitor().getClientArea();
        int targetWidth = 1200;
        int targetHeight = 800;
        if (parentShell != null) {
            Rectangle parentBounds = parentShell.getBounds();
            targetWidth = parentBounds.width;
            targetHeight = parentBounds.height;
        }
        // Clamp to screen size with some margin
        int maxWidth = screenBounds.width - 50;
        int maxHeight = screenBounds.height - 50;
        targetWidth = Math.min(targetWidth, maxWidth);
        targetHeight = Math.min(targetHeight, maxHeight);
        shell.setSize(targetWidth, targetHeight);

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

        // Offset from parent (down and to the right), clamped to screen
        if (parentShell != null) {
            Rectangle parentBounds = parentShell.getBounds();
            Rectangle shellBounds = shell.getBounds();
            int offsetX = 60;
            int offsetY = 60;
            int x = parentBounds.x + offsetX;
            int y = parentBounds.y + offsetY;
            // Clamp to screen bounds
            Rectangle screenBounds2 = display.getPrimaryMonitor().getClientArea();
            if (x + shellBounds.width > screenBounds2.x + screenBounds2.width) {
                x = screenBounds2.x + screenBounds2.width - shellBounds.width;
            }
            if (y + shellBounds.height > screenBounds2.y + screenBounds2.height) {
                y = screenBounds2.y + screenBounds2.height - shellBounds.height;
            }
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
        statusComp.setLayout(new GridLayout(3, false));
        ((GridLayout)statusComp.getLayout()).marginHeight = 2;
        ((GridLayout)statusComp.getLayout()).marginWidth = 5;
        statusComp.setBackground(new Color(COLOR_STATUS_BAR_BG[0], COLOR_STATUS_BAR_BG[1], COLOR_STATUS_BAR_BG[2]));

        // Node count on the left
        nodeCountLabel = new Label(statusComp, SWT.NONE);
        updateNodeCount();
        nodeCountLabel.setLayoutData(new GridData(SWT.LEFT, SWT.CENTER, false, false));
        nodeCountLabel.setBackground(new Color(COLOR_STATUS_BAR_BG[0], COLOR_STATUS_BAR_BG[1], COLOR_STATUS_BAR_BG[2]));
        nodeCountLabel.setForeground(display.getSystemColor(SWT.COLOR_BLACK));

        // Pipeline status in center
        pipelineStatusLabel = new Label(statusComp, SWT.NONE);
        pipelineStatusLabel.setText("Pipeline Stopped");
        GridData statusGd = new GridData(SWT.CENTER, SWT.CENTER, true, false);
        statusGd.widthHint = 200; // Reserve space for thread count text
        pipelineStatusLabel.setLayoutData(statusGd);
        pipelineStatusLabel.setBackground(new Color(COLOR_STATUS_BAR_BG[0], COLOR_STATUS_BAR_BG[1], COLOR_STATUS_BAR_BG[2]));
        pipelineStatusLabel.setForeground(new Color(COLOR_STATUS_STOPPED[0], COLOR_STATUS_STOPPED[1], COLOR_STATUS_STOPPED[2]));

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

        // Right-click context menu
        canvas.addMenuDetectListener(e -> handleRightClick(e));

        // Keyboard handler for delete, arrow keys, and shortcuts
        canvas.addKeyListener(new KeyAdapter() {
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
                else if (e.keyCode == SWT.DEL || e.keyCode == SWT.BS) {
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
        startStopBtn.setBackground(new Color(COLOR_START_BUTTON[0], COLOR_START_BUTTON[1], COLOR_START_BUTTON[2]));
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
                try {
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
                } catch (Exception ex) {
                    // Image may have become invalid during drawing
                    System.err.println("Warning: Failed to draw preview image: " + ex.getMessage());
                }
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
        gc.setForeground(new Color(COLOR_GRID_LINES[0], COLOR_GRID_LINES[1], COLOR_GRID_LINES[2]));
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

        // Selection highlight color - matches node selection
        Color selectionColor = new Color(PipelineNode.SELECTION_COLOR_R, PipelineNode.SELECTION_COLOR_G, PipelineNode.SELECTION_COLOR_B);

        // Draw connections with bezier curves (same style as main editor)
        for (Connection conn : connections) {
            Point start = conn.source.getOutputPoint(conn.outputIndex);
            Point end = getConnectionTargetPoint(conn);

            if (selectedConnections.contains(conn)) {
                gc.setLineWidth(3);
                gc.setForeground(selectionColor);
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
        // Draw connection being dragged from output (forward)
        if (connectionSource != null && connectionEndPoint != null) {
            Point start = connectionSource.getOutputPoint(connectionSourceOutputIndex);
            gc.setLineWidth(2);
            gc.setForeground(display.getSystemColor(SWT.COLOR_BLUE));
            gc.drawLine(start.x, start.y, connectionEndPoint.x, connectionEndPoint.y);
        }

        // Draw connection being dragged from target (reverse - after yanking from output)
        if (connectionTarget != null && connectionEndPoint != null) {
            Point end = targetInputIndex == 2 && connectionTarget.hasDualInput()
                ? connectionTarget.getInputPoint2()
                : connectionTarget.getInputPoint();
            gc.setLineWidth(2);
            gc.setForeground(display.getSystemColor(SWT.COLOR_BLUE));
            gc.drawLine(connectionEndPoint.x, connectionEndPoint.y, end.x, end.y);
        }

        // Draw free connection being dragged (both ends unattached)
        if (freeConnectionFixedEnd != null && connectionEndPoint != null) {
            gc.setForeground(display.getSystemColor(SWT.COLOR_GRAY));
            gc.setLineStyle(SWT.LINE_DASH);
            gc.setLineWidth(2);
            if (draggingFreeConnectionSource) {
                // Dragging source end - connectionEndPoint is source, freeConnectionFixedEnd is target (arrow end)
                gc.drawLine(connectionEndPoint.x, connectionEndPoint.y, freeConnectionFixedEnd.x, freeConnectionFixedEnd.y);
                drawArrow(gc, connectionEndPoint, freeConnectionFixedEnd);
                // Draw circle at fixed arrow end
                gc.setBackground(display.getSystemColor(SWT.COLOR_GRAY));
                gc.fillOval(freeConnectionFixedEnd.x - 4, freeConnectionFixedEnd.y - 4, 8, 8);
            } else {
                // Dragging target end - freeConnectionFixedEnd is source, connectionEndPoint is target (arrow end)
                gc.drawLine(freeConnectionFixedEnd.x, freeConnectionFixedEnd.y, connectionEndPoint.x, connectionEndPoint.y);
                drawArrow(gc, freeConnectionFixedEnd, connectionEndPoint);
                // Draw circle at fixed source end
                gc.setBackground(display.getSystemColor(SWT.COLOR_GRAY));
                gc.fillOval(freeConnectionFixedEnd.x - 4, freeConnectionFixedEnd.y - 4, 8, 8);
            }
            gc.setLineStyle(SWT.LINE_SOLID);
        }

        // Draw dangling connections (source connected, target free) as dashed lines
        gc.setLineStyle(SWT.LINE_DASH);
        gc.setLineWidth(2);
        for (DanglingConnection dc : danglingConnections) {
            boolean isSelected = selectedDanglingConnections.contains(dc);
            gc.setForeground(isSelected ? selectionColor : display.getSystemColor(SWT.COLOR_GRAY));
            Point start = dc.source.getOutputPoint(dc.outputIndex);
            gc.drawLine(start.x, start.y, dc.freeEnd.x, dc.freeEnd.y);
            // Draw small circle at free end
            gc.setBackground(isSelected ? selectionColor : display.getSystemColor(SWT.COLOR_GRAY));
            gc.fillOval(dc.freeEnd.x - 4, dc.freeEnd.y - 4, 8, 8);
        }

        // Draw reverse dangling connections (target connected, source free) as dashed lines
        for (ReverseDanglingConnection rdc : reverseDanglingConnections) {
            boolean isSelected = selectedReverseDanglingConnections.contains(rdc);
            gc.setForeground(isSelected ? selectionColor : display.getSystemColor(SWT.COLOR_GRAY));
            // Use correct input point based on inputIndex
            Point end = (rdc.inputIndex == 2 && rdc.target.hasDualInput())
                ? rdc.target.getInputPoint2()
                : rdc.target.getInputPoint();
            gc.drawLine(rdc.freeEnd.x, rdc.freeEnd.y, end.x, end.y);
            // Draw small circle at free end
            gc.setBackground(isSelected ? selectionColor : display.getSystemColor(SWT.COLOR_GRAY));
            gc.fillOval(rdc.freeEnd.x - 4, rdc.freeEnd.y - 4, 8, 8);
        }

        // Draw free connections (both ends free) as dashed lines with circles at both ends
        for (FreeConnection fc : freeConnections) {
            boolean isSelected = selectedFreeConnections.contains(fc);
            gc.setForeground(isSelected ? selectionColor : display.getSystemColor(SWT.COLOR_GRAY));
            gc.drawLine(fc.startEnd.x, fc.startEnd.y, fc.arrowEnd.x, fc.arrowEnd.y);
            drawArrow(gc, fc.startEnd, fc.arrowEnd);
            // Draw small circles at both ends
            gc.setBackground(isSelected ? selectionColor : display.getSystemColor(SWT.COLOR_GRAY));
            gc.fillOval(fc.startEnd.x - 4, fc.startEnd.y - 4, 8, 8);
            gc.fillOval(fc.arrowEnd.x - 4, fc.arrowEnd.y - 4, 8, 8);
        }
        gc.setLineStyle(SWT.LINE_SOLID);

        // Dispose selection color after use
        selectionColor.dispose();

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
            gc.setBackground(new Color(COLOR_SELECTION_BOX[0], COLOR_SELECTION_BOX[1], COLOR_SELECTION_BOX[2]));
            gc.setAlpha(30);
            gc.fillRectangle(boxX, boxY, boxWidth, boxHeight);
            gc.setAlpha(255);

            // Draw selection box border
            gc.setForeground(new Color(COLOR_SELECTION_BOX[0], COLOR_SELECTION_BOX[1], COLOR_SELECTION_BOX[2]));
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

        int radius = 8;

        // Check if clicking on a dangling connection's free end (to pick it up)
        DanglingConnection danglingToRemove = null;
        for (DanglingConnection dc : danglingConnections) {
            double dist = Math.sqrt(Math.pow(click.x - dc.freeEnd.x, 2) + Math.pow(click.y - dc.freeEnd.y, 2));
            if (dist <= radius) {
                // Pick up this dangling connection
                connectionSource = dc.source;
                connectionSourceOutputIndex = dc.outputIndex;
                connectionTarget = null;
                connectionEndPoint = click;
                danglingToRemove = dc;
                break;
            }
        }
        if (danglingToRemove != null) {
            danglingConnections.remove(danglingToRemove);
            canvas.redraw();
            return;
        }

        // Check if clicking on a reverse dangling connection's free end (to pick it up)
        ReverseDanglingConnection reverseDanglingToRemove = null;
        for (ReverseDanglingConnection rdc : reverseDanglingConnections) {
            double dist = Math.sqrt(Math.pow(click.x - rdc.freeEnd.x, 2) + Math.pow(click.y - rdc.freeEnd.y, 2));
            if (dist <= radius) {
                // Pick up this reverse dangling connection - preserve input index
                connectionTarget = rdc.target;
                targetInputIndex = rdc.inputIndex; // Preserve original input index
                connectionSource = null;
                connectionEndPoint = click;
                reverseDanglingToRemove = rdc;
                break;
            }
        }
        if (reverseDanglingToRemove != null) {
            reverseDanglingConnections.remove(reverseDanglingToRemove);
            canvas.redraw();
            return;
        }

        // Check if clicking on a FreeConnection's start end (non-arrow end)
        FreeConnection freeStartToRemove = null;
        Point freeStartOtherEnd = null;
        for (FreeConnection fc : freeConnections) {
            double dist = Math.sqrt(Math.pow(click.x - fc.startEnd.x, 2) + Math.pow(click.y - fc.startEnd.y, 2));
            if (dist <= radius) {
                freeStartToRemove = fc;
                freeStartOtherEnd = fc.arrowEnd;
                break;
            }
        }
        if (freeStartToRemove != null) {
            freeConnections.remove(freeStartToRemove);
            // Set up dragging from the start end - arrow end stays fixed
            freeConnectionFixedEnd = freeStartOtherEnd;
            draggingFreeConnectionSource = true; // We're dragging the start/source end
            connectionEndPoint = click;
            connectionSource = null;
            connectionTarget = null;
            canvas.redraw();
            return;
        }

        // Check if clicking on a FreeConnection's arrow end
        FreeConnection freeArrowToRemove = null;
        Point freeArrowOtherEnd = null;
        for (FreeConnection fc : freeConnections) {
            double dist = Math.sqrt(Math.pow(click.x - fc.arrowEnd.x, 2) + Math.pow(click.y - fc.arrowEnd.y, 2));
            if (dist <= radius) {
                freeArrowToRemove = fc;
                freeArrowOtherEnd = fc.startEnd;
                break;
            }
        }
        if (freeArrowToRemove != null) {
            freeConnections.remove(freeArrowToRemove);
            // Set up dragging from the arrow end - start end stays fixed
            freeConnectionFixedEnd = freeArrowOtherEnd;
            draggingFreeConnectionSource = false; // We're dragging the arrow/target end
            connectionEndPoint = click;
            connectionSource = null;
            connectionTarget = null;
            canvas.redraw();
            return;
        }

        // First: Check output points - if connected, yank the connection; otherwise start new
        // Check boundary input's output point first
        int outputIdx = boundaryInput.getOutputIndexNear(click.x, click.y);
        if (outputIdx >= 0) {
            // Check if there's an existing connection from this output to yank
            Connection toYank = findConnectionFromSource(boundaryInput, outputIdx);
            if (toYank != null) {
                // Yank the connection - start dragging from the target end
                // Clear any forward connection state first
                connectionSource = null;
                yankedOriginalTarget = null;
                // Set reverse connection state
                connectionTarget = toYank.target;
                targetInputIndex = toYank.inputIndex;
                // Save original source to restore if dropped on empty space
                yankedOriginalSource = toYank.source;
                yankedOriginalOutputIndex = toYank.outputIndex;
                connectionEndPoint = click;
                connections.remove(toYank);
                notifyModified();
                canvas.redraw();
                return;
            }
            // Check if there's a dangling connection from this output to yank
            DanglingConnection danglingToYank = findDanglingFromSource(boundaryInput, outputIdx);
            if (danglingToYank != null) {
                // Yank off the source end - use freeConnectionFixedEnd pattern
                // The arrow end stays fixed, we drag the source end
                danglingConnections.remove(danglingToYank);
                freeConnectionFixedEnd = danglingToYank.freeEnd; // The arrow end stays put
                draggingFreeConnectionSource = true; // We're dragging the source end
                connectionEndPoint = click;
                connectionSource = null;
                connectionTarget = null;
                canvas.redraw();
                return;
            }
            // No existing connection - start a new connection
            connectionTarget = null; // Clear reverse state
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
                // Check if there's an existing connection from this output to yank
                Connection toYank = findConnectionFromSource(node, outputIdx);
                if (toYank != null) {
                    // Yank the connection - start dragging from the target end
                    // Clear any forward connection state first
                    connectionSource = null;
                    yankedOriginalTarget = null;
                    // Set reverse connection state
                    connectionTarget = toYank.target;
                    targetInputIndex = toYank.inputIndex;
                    // Save original source to restore if dropped on empty space
                    yankedOriginalSource = toYank.source;
                    yankedOriginalOutputIndex = toYank.outputIndex;
                    connectionEndPoint = click;
                    connections.remove(toYank);
                    notifyModified();
                    canvas.redraw();
                    return;
                }
                // Check if there's a dangling connection from this output to yank
                DanglingConnection danglingToYank = findDanglingFromSource(node, outputIdx);
                if (danglingToYank != null) {
                    // Yank off the source end - use freeConnectionFixedEnd pattern
                    // The arrow end stays fixed, we drag the source end
                    danglingConnections.remove(danglingToYank);
                    freeConnectionFixedEnd = danglingToYank.freeEnd; // The arrow end stays put
                    draggingFreeConnectionSource = true; // We're dragging the source end
                    connectionEndPoint = click;
                    connectionSource = null;
                    connectionTarget = null;
                    canvas.redraw();
                    return;
                }
                // No existing connection - start a new connection
                connectionTarget = null; // Clear reverse state
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
                // Clear any reverse connection state first
                connectionTarget = null;
                yankedOriginalSource = null;
                // Set forward connection state
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
            // Check if there's a reverse dangling connection to this input to yank
            ReverseDanglingConnection reverseDanglingToYank = findReverseDanglingToTarget(boundaryOutput, 1);
            if (reverseDanglingToYank != null) {
                // Pick up the reverse dangling connection - use freeConnectionFixedEnd pattern
                // The source end stays fixed, we drag the arrow end
                reverseDanglingConnections.remove(reverseDanglingToYank);
                freeConnectionFixedEnd = reverseDanglingToYank.freeEnd; // The source end stays put
                draggingFreeConnectionSource = false; // We're dragging the arrow/target end
                connectionEndPoint = click;
                connectionTarget = null;
                connectionSource = null;
                canvas.redraw();
                return;
            }
            // No existing connection - start a new reverse connection (dragging from input to find output)
            connectionTarget = boundaryOutput;
            targetInputIndex = 1;
            connectionSource = null;
            connectionEndPoint = click;
            canvas.redraw();
            return;
        }

        // Check child nodes' input points
        for (PipelineNode node : nodes) {
            if (node.isNearInputPoint(click.x, click.y)) {
                Connection toYank = findConnectionToTarget(node, 1);
                if (toYank != null) {
                    // Clear any reverse connection state first
                    connectionTarget = null;
                    yankedOriginalSource = null;
                    // Set forward connection state
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
                // Check if there's a reverse dangling connection to this input (input1) to yank
                ReverseDanglingConnection reverseDanglingToYank = findReverseDanglingToTarget(node, 1);
                if (reverseDanglingToYank != null) {
                    // Pick up the reverse dangling connection - use freeConnectionFixedEnd pattern
                    // The source end stays fixed, we drag the arrow end
                    reverseDanglingConnections.remove(reverseDanglingToYank);
                    freeConnectionFixedEnd = reverseDanglingToYank.freeEnd; // The source end stays put
                    draggingFreeConnectionSource = false; // We're dragging the arrow/target end
                    connectionEndPoint = click;
                    connectionTarget = null;
                    connectionSource = null;
                    canvas.redraw();
                    return;
                }
                // No existing connection - start a new reverse connection (dragging from input to find output)
                connectionTarget = node;
                targetInputIndex = 1;
                connectionSource = null;
                connectionEndPoint = click;
                canvas.redraw();
                return;
            }
            if (node.hasDualInput() && node.isNearInputPoint2(click.x, click.y)) {
                Connection toYank = findConnectionToTarget(node, 2);
                if (toYank != null) {
                    // Clear any reverse connection state first
                    connectionTarget = null;
                    yankedOriginalSource = null;
                    // Set forward connection state
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
                // Check if there's a reverse dangling connection to this input (input2) to yank
                ReverseDanglingConnection reverseDanglingToYank2 = findReverseDanglingToTarget(node, 2);
                if (reverseDanglingToYank2 != null) {
                    // Pick up the reverse dangling connection - use freeConnectionFixedEnd pattern
                    // The source end stays fixed, we drag the arrow end
                    reverseDanglingConnections.remove(reverseDanglingToYank2);
                    freeConnectionFixedEnd = reverseDanglingToYank2.freeEnd; // The source end stays put
                    draggingFreeConnectionSource = false; // We're dragging the arrow/target end
                    connectionEndPoint = click;
                    connectionTarget = null;
                    connectionSource = null;
                    canvas.redraw();
                    return;
                }
                // No existing connection - start a new reverse connection (dragging from input2 to find output)
                connectionTarget = node;
                targetInputIndex = 2;
                connectionSource = null;
                connectionEndPoint = click;
                canvas.redraw();
                return;
            }
        }

        // Third: Check for node body clicks (for selection and dragging)
        for (int i = nodes.size() - 1; i >= 0; i--) {
            PipelineNode node = nodes.get(i);
            if (node.containsPoint(click)) {
                // Check if clicking on enabled checkbox (ProcessingNode or SourceNode)
                if (node instanceof ProcessingNode) {
                    ProcessingNode pNode = (ProcessingNode) node;
                    if (pNode.isOnEnabledCheckbox(click)) {
                        pNode.toggleEnabled();
                        notifyModified();
                        canvas.redraw();
                        return;
                    }
                } else if (node instanceof SourceNode) {
                    SourceNode sNode = (SourceNode) node;
                    if (sNode.isOnEnabledCheckbox(click)) {
                        sNode.toggleEnabled();
                        notifyModified();
                        canvas.redraw();
                        return;
                    }
                }
                // Check if clicking on help icon - open help browser
                if (node.isOnHelpIcon(click)) {
                    if (HelpBrowser.hasHelp(node.getClass())) {
                        HelpBrowser.openForNode(shell, node.getClass());
                        return;
                    }
                }
                // Check if clicking on container icon - open nested container editor
                if (node instanceof ContainerNode) {
                    ContainerNode nestedContainer = (ContainerNode) node;
                    if (nestedContainer.isOnContainerIcon(click)) {
                        openNestedContainerEditor(nestedContainer);
                        return;
                    }
                }
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
                selectedDanglingConnections.clear();
                selectedReverseDanglingConnections.clear();
                selectedFreeConnections.clear();
                selectedConnections.add(conn);
                selectedNode = null;
                canvas.redraw();
                return;
            }
        }

        // Check for dangling connection line selection
        for (DanglingConnection dc : danglingConnections) {
            if (isNearDanglingLine(dc, click)) {
                selectedNodes.clear();
                selectedConnections.clear();
                selectedDanglingConnections.clear();
                selectedReverseDanglingConnections.clear();
                selectedFreeConnections.clear();
                selectedDanglingConnections.add(dc);
                selectedNode = null;
                canvas.redraw();
                return;
            }
        }

        // Check for reverse dangling connection line selection
        for (ReverseDanglingConnection rdc : reverseDanglingConnections) {
            if (isNearReverseDanglingLine(rdc, click)) {
                selectedNodes.clear();
                selectedConnections.clear();
                selectedDanglingConnections.clear();
                selectedReverseDanglingConnections.clear();
                selectedFreeConnections.clear();
                selectedReverseDanglingConnections.add(rdc);
                selectedNode = null;
                canvas.redraw();
                return;
            }
        }

        // Check for free connection line selection
        for (FreeConnection fc : freeConnections) {
            if (isNearFreeConnectionLine(fc, click)) {
                selectedNodes.clear();
                selectedConnections.clear();
                selectedDanglingConnections.clear();
                selectedReverseDanglingConnections.clear();
                selectedFreeConnections.clear();
                selectedFreeConnections.add(fc);
                selectedNode = null;
                canvas.redraw();
                return;
            }
        }

        // Clicked on empty space - start selection box
        selectedNode = null;
        selectedNodes.clear();
        selectedConnections.clear();
        selectedDanglingConnections.clear();
        selectedReverseDanglingConnections.clear();
        selectedFreeConnections.clear();
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

    /**
     * Find a dangling connection originating from the given source node's output.
     */
    private DanglingConnection findDanglingFromSource(PipelineNode source, int outputIndex) {
        for (DanglingConnection dc : danglingConnections) {
            if (dc.source == source && dc.outputIndex == outputIndex) {
                return dc;
            }
        }
        return null;
    }

    /**
     * Find a reverse dangling connection targeting the given node and input index.
     */
    private ReverseDanglingConnection findReverseDanglingToTarget(PipelineNode target, int inputIndex) {
        for (ReverseDanglingConnection rdc : reverseDanglingConnections) {
            if (rdc.target == target && rdc.inputIndex == inputIndex) {
                return rdc;
            }
        }
        return null;
    }

    /**
     * Check if a point is near a dangling connection line.
     */
    private boolean isNearDanglingLine(DanglingConnection dc, Point click) {
        Point start = dc.source.getOutputPoint(dc.outputIndex);
        Point end = dc.freeEnd;
        return pointToLineDistance(click, start, end) < 8;
    }

    /**
     * Check if a point is near a reverse dangling connection line.
     */
    private boolean isNearReverseDanglingLine(ReverseDanglingConnection rdc, Point click) {
        Point start = rdc.freeEnd;
        Point end = (rdc.inputIndex == 2 && rdc.target.hasDualInput())
            ? rdc.target.getInputPoint2()
            : rdc.target.getInputPoint();
        return pointToLineDistance(click, start, end) < 8;
    }

    /**
     * Check if a point is near a free connection line.
     */
    private boolean isNearFreeConnectionLine(FreeConnection fc, Point click) {
        return pointToLineDistance(click, fc.startEnd, fc.arrowEnd) < 8;
    }

    /**
     * Calculate distance from point to line segment.
     */
    private double pointToLineDistance(Point p, Point lineStart, Point lineEnd) {
        double dx = lineEnd.x - lineStart.x;
        double dy = lineEnd.y - lineStart.y;
        double lengthSquared = dx * dx + dy * dy;

        if (lengthSquared == 0) {
            return Math.sqrt(Math.pow(p.x - lineStart.x, 2) + Math.pow(p.y - lineStart.y, 2));
        }

        double t = ((p.x - lineStart.x) * dx + (p.y - lineStart.y) * dy) / lengthSquared;
        t = Math.max(0, Math.min(1, t));

        double closestX = lineStart.x + t * dx;
        double closestY = lineStart.y + t * dy;

        return Math.sqrt(Math.pow(p.x - closestX, 2) + Math.pow(p.y - closestY, 2));
    }

    private void handleMouseMove(MouseEvent e) {
        Point click = toCanvasPoint(e.x, e.y);

        // Check for tooltip on connection points
        updateConnectionTooltip(click.x, click.y);

        // Update connection endpoint (forward drag from output)
        if (connectionSource != null) {
            connectionEndPoint = click;
            canvas.redraw();
            return;
        }

        // Update connection endpoint (reverse drag from target after yanking from output)
        if (connectionTarget != null) {
            connectionEndPoint = click;
            canvas.redraw();
            return;
        }

        // Update connection endpoint (free connection dragging - both ends unattached)
        if (freeConnectionFixedEnd != null) {
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
     * Update canvas tooltip and cursor based on mouse position over connection points and icons.
     */
    private void updateConnectionTooltip(int canvasX, int canvasY) {
        String tooltip = null;
        boolean showHandCursor = false;
        Point p = new Point(canvasX, canvasY);

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
                // Check enabled checkbox (ProcessingNode only in containers)
                if (node.containsPoint(p) && node instanceof ProcessingNode) {
                    ProcessingNode pNode = (ProcessingNode) node;
                    if (pNode.isOnEnabledCheckbox(p)) {
                        tooltip = pNode.isEnabled() ? "Click to disable node" : "Click to enable node";
                        showHandCursor = true;
                        break;
                    }
                }

                // Check help icon first (only if mouse is within node bounds for efficiency)
                if (node.containsPoint(p) && node.isOnHelpIcon(p)) {
                    if (HelpBrowser.hasHelp(node.getClass())) {
                        tooltip = "Help";
                        showHandCursor = true;
                    } else {
                        tooltip = "Help (not yet available)";
                    }
                    break;
                }

                // Check container icon for nested ContainerNodes
                if (node instanceof ContainerNode) {
                    ContainerNode nestedContainer = (ContainerNode) node;
                    if (nestedContainer.isOnContainerIcon(p)) {
                        tooltip = "Edit Container's Sub-diagram";
                        showHandCursor = true;
                        break;
                    }
                }

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

        // Update cursor
        Cursor currentCursor = canvas.getCursor();
        if (showHandCursor) {
            if (currentCursor == null || currentCursor.equals(display.getSystemCursor(SWT.CURSOR_ARROW))) {
                canvas.setCursor(display.getSystemCursor(SWT.CURSOR_HAND));
            }
        } else {
            if (currentCursor != null && currentCursor.equals(display.getSystemCursor(SWT.CURSOR_HAND))) {
                canvas.setCursor(null);
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

            // If dropped on empty space, create a dangling connection
            if (!connectionMade && connectionSource != null) {
                danglingConnections.add(new DanglingConnection(connectionSource, connectionSourceOutputIndex, click));
                notifyModified();
            }

            connectionSource = null;
            connectionEndPoint = null;
            yankedOriginalTarget = null;
            yankedOriginalInputIndex = 0;
            canvas.redraw();
        }

        // Handle reverse connection (dragging from target end after yanking from output)
        if (connectionTarget != null && connectionEndPoint != null) {
            boolean connectionMade = false;
            PipelineNode sourceNode = null;
            int sourceOutputIndex = 0;

            ContainerInputNode boundaryInput = container.getBoundaryInput();

            // Check boundary input's output point
            int outputIdx = boundaryInput.getOutputIndexNear(click.x, click.y);
            if (outputIdx >= 0) {
                sourceNode = boundaryInput;
                sourceOutputIndex = outputIdx;
                connectionMade = true;
            }

            // Check child nodes' output points
            if (!connectionMade) {
                for (PipelineNode node : nodes) {
                    if (node != connectionTarget) {
                        outputIdx = node.getOutputIndexNear(click.x, click.y);
                        if (outputIdx >= 0) {
                            sourceNode = node;
                            sourceOutputIndex = outputIdx;
                            connectionMade = true;
                            break;
                        }
                    }
                }
            }

            if (connectionMade && sourceNode != null) {
                createConnection(sourceNode, connectionTarget, targetInputIndex, sourceOutputIndex);
            } else if (connectionTarget != null) {
                // If dropped on empty space, create a reverse dangling connection (preserving inputIndex)
                reverseDanglingConnections.add(new ReverseDanglingConnection(connectionTarget, targetInputIndex, click));
                notifyModified();
            }

            connectionTarget = null;
            connectionEndPoint = null;
            targetInputIndex = 1;
            yankedOriginalSource = null;
            yankedOriginalOutputIndex = 0;
            canvas.redraw();
        }

        // Handle free connection dragging (both ends unattached)
        if (freeConnectionFixedEnd != null && connectionEndPoint != null) {
            ContainerInputNode boundaryInput = container.getBoundaryInput();
            ContainerOutputNode boundaryOutput = container.getBoundaryOutput();

            if (draggingFreeConnectionSource) {
                // Dragging the source end - check if dropped on output point
                PipelineNode sourceNode = null;
                int sourceOutputIndex = 0;
                boolean connected = false;

                // Check boundary input's output point
                int outputIdx = boundaryInput.getOutputIndexNear(click.x, click.y);
                if (outputIdx >= 0) {
                    sourceNode = boundaryInput;
                    sourceOutputIndex = outputIdx;
                    connected = true;
                }

                // Check child nodes' output points
                if (!connected) {
                    for (PipelineNode node : nodes) {
                        outputIdx = node.getOutputIndexNear(click.x, click.y);
                        if (outputIdx >= 0) {
                            sourceNode = node;
                            sourceOutputIndex = outputIdx;
                            connected = true;
                            break;
                        }
                    }
                }

                if (connected && sourceNode != null) {
                    // Connected to output point - create DanglingConnection with output index
                    danglingConnections.add(new DanglingConnection(sourceNode, sourceOutputIndex, freeConnectionFixedEnd));
                    notifyModified();
                } else {
                    // Not connected - create FreeConnection (both ends free)
                    // connectionEndPoint is the source end, freeConnectionFixedEnd is the arrow end
                    freeConnections.add(new FreeConnection(click, freeConnectionFixedEnd));
                    notifyModified();
                }
            } else {
                // Dragging the target end - check if dropped on input point
                PipelineNode targetNode = null;
                int inputIdx = 1;
                boolean connected = false;

                // Check boundary output's input point
                if (boundaryOutput.isNearInputPoint(click.x, click.y)) {
                    targetNode = boundaryOutput;
                    inputIdx = 1;
                    connected = true;
                }

                // Check child nodes' input points
                if (!connected) {
                    for (PipelineNode node : nodes) {
                        if (node.isNearInputPoint(click.x, click.y)) {
                            targetNode = node;
                            inputIdx = 1;
                            connected = true;
                            break;
                        }
                        if (node.hasDualInput() && node.isNearInputPoint2(click.x, click.y)) {
                            targetNode = node;
                            inputIdx = 2;
                            connected = true;
                            break;
                        }
                    }
                }

                if (connected && targetNode != null) {
                    // Connected to input point - create ReverseDanglingConnection with correct inputIndex
                    reverseDanglingConnections.add(new ReverseDanglingConnection(targetNode, inputIdx, freeConnectionFixedEnd));
                    notifyModified();
                } else {
                    // Not connected - create FreeConnection (both ends free)
                    // freeConnectionFixedEnd is the source end, connectionEndPoint is the arrow end
                    freeConnections.add(new FreeConnection(freeConnectionFixedEnd, click));
                    notifyModified();
                }
            }

            freeConnectionFixedEnd = null;
            draggingFreeConnectionSource = false;
            connectionEndPoint = null;
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

            // Select dangling connections that are completely inside selection box
            for (DanglingConnection dc : danglingConnections) {
                Point start = dc.source.getOutputPoint(dc.outputIndex);
                Point end = dc.freeEnd;
                if (start.x >= boxX && start.x <= boxX + boxWidth &&
                    start.y >= boxY && start.y <= boxY + boxHeight &&
                    end.x >= boxX && end.x <= boxX + boxWidth &&
                    end.y >= boxY && end.y <= boxY + boxHeight) {
                    selectedDanglingConnections.add(dc);
                }
            }

            // Select reverse dangling connections that are completely inside selection box
            for (ReverseDanglingConnection rdc : reverseDanglingConnections) {
                Point start = rdc.freeEnd;
                Point end = (rdc.inputIndex == 2 && rdc.target.hasDualInput())
                    ? rdc.target.getInputPoint2()
                    : rdc.target.getInputPoint();
                if (start.x >= boxX && start.x <= boxX + boxWidth &&
                    start.y >= boxY && start.y <= boxY + boxHeight &&
                    end.x >= boxX && end.x <= boxX + boxWidth &&
                    end.y >= boxY && end.y <= boxY + boxHeight) {
                    selectedReverseDanglingConnections.add(rdc);
                }
            }

            // Select free connections that are completely inside selection box
            for (FreeConnection fc : freeConnections) {
                if (fc.startEnd.x >= boxX && fc.startEnd.x <= boxX + boxWidth &&
                    fc.startEnd.y >= boxY && fc.startEnd.y <= boxY + boxHeight &&
                    fc.arrowEnd.x >= boxX && fc.arrowEnd.x <= boxX + boxWidth &&
                    fc.arrowEnd.y >= boxY && fc.arrowEnd.y <= boxY + boxHeight) {
                    selectedFreeConnections.add(fc);
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
                // Reset drag state - mouseDown fired before doubleClick and may have set isDragging
                isDragging = false;
                dragOffset = null;

                // ContainerNodes: double-click opens nested container editor
                if (node instanceof ContainerNode) {
                    openNestedContainerEditor((ContainerNode) node);
                } else if (node instanceof ProcessingNode) {
                    // Other ProcessingNodes: double-click opens properties dialog
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

    private void handleRightClick(MenuDetectEvent e) {
        // Convert to canvas coordinates accounting for zoom
        Point screenPoint = display.map(null, canvas, new Point(e.x, e.y));
        Point clickPoint = toCanvasPoint(screenPoint.x, screenPoint.y);

        for (PipelineNode node : nodes) {
            if (node.containsPoint(clickPoint)) {
                // Show context menu for the node
                Menu contextMenu = new Menu(canvas);

                // Node name header (disabled)
                MenuItem nameItem = new MenuItem(contextMenu, SWT.PUSH);
                nameItem.setText(node.getDisplayLabel());
                nameItem.setEnabled(false);
                new MenuItem(contextMenu, SWT.SEPARATOR);

                // ContainerNode: Edit Container Contents first, then Properties
                if (node instanceof ContainerNode) {
                    ContainerNode nestedContainer = (ContainerNode) node;

                    MenuItem editContentsItem = new MenuItem(contextMenu, SWT.PUSH);
                    editContentsItem.setText("Edit Container Contents...");
                    editContentsItem.addListener(SWT.Selection, evt -> {
                        openNestedContainerEditor(nestedContainer);
                    });

                    MenuItem propsItem = new MenuItem(contextMenu, SWT.PUSH);
                    propsItem.setText("Properties...");
                    propsItem.addListener(SWT.Selection, evt -> {
                        nestedContainer.showPropertiesDialog();
                    });

                    new MenuItem(contextMenu, SWT.SEPARATOR);
                } else if (node instanceof ProcessingNode) {
                    // Other ProcessingNodes: Edit Properties
                    MenuItem editItem = new MenuItem(contextMenu, SWT.PUSH);
                    editItem.setText("Properties...");
                    editItem.addListener(SWT.Selection, evt -> {
                        // Reset drag state in case right-click happened while dragging
                        isDragging = false;
                        dragOffset = null;

                        ProcessingNode pn = (ProcessingNode) node;
                        Shell originalShell = pn.getShell();
                        pn.setShell(shell);
                        pn.showPropertiesDialog();
                        pn.setShell(originalShell);
                    });

                    new MenuItem(contextMenu, SWT.SEPARATOR);
                }

                // Delete Node option
                MenuItem deleteItem = new MenuItem(contextMenu, SWT.PUSH);
                deleteItem.setText("Delete Node");
                deleteItem.addListener(SWT.Selection, evt -> {
                    // Remove all connections involving this node
                    connections.removeIf(c -> c.source == node || c.target == node);
                    danglingConnections.removeIf(dc -> dc.source == node);
                    reverseDanglingConnections.removeIf(rdc -> rdc.target == node);
                    nodes.remove(node);
                    notifyModified();
                    canvas.redraw();
                });

                contextMenu.setLocation(e.x, e.y);
                contextMenu.setVisible(true);
                return;
            }
        }

        // Check if right-clicked on a connection
        for (Connection conn : connections) {
            Point start = conn.source.getOutputPoint(conn.outputIndex);
            Point end = getConnectionTargetPoint(conn);
            if (isNearConnectionLine(conn, clickPoint)) {
                // Show context menu for the connection
                Menu contextMenu = new Menu(canvas);

                MenuItem deleteItem = new MenuItem(contextMenu, SWT.PUSH);
                deleteItem.setText("Delete Connection");
                deleteItem.addListener(SWT.Selection, evt -> {
                    connections.remove(conn);
                    notifyModified();
                    canvas.redraw();
                });

                contextMenu.setLocation(e.x, e.y);
                contextMenu.setVisible(true);
                return;
            }
        }
    }

    private void deleteSelectedImpl() {
        boolean modified = false;

        if (selectedNode != null) {
            // Don't delete boundary nodes
            if (selectedNode instanceof ContainerInputNode || selectedNode instanceof ContainerOutputNode) {
                return;
            }

            // Remove connections to/from this node
            connections.removeIf(conn -> conn.source == selectedNode || conn.target == selectedNode);
            // Remove dangling connections from this node
            danglingConnections.removeIf(dc -> dc.source == selectedNode);
            reverseDanglingConnections.removeIf(rdc -> rdc.target == selectedNode);

            // Remove node
            nodes.remove(selectedNode);
            selectedNode = null;
            selectedNodes.clear();
            modified = true;
        }

        // Remove selected connections
        if (!selectedConnections.isEmpty()) {
            connections.removeAll(selectedConnections);
            selectedConnections.clear();
            modified = true;
        }

        // Remove selected dangling connections
        if (!selectedDanglingConnections.isEmpty()) {
            danglingConnections.removeAll(selectedDanglingConnections);
            selectedDanglingConnections.clear();
            modified = true;
        }

        // Remove selected reverse dangling connections
        if (!selectedReverseDanglingConnections.isEmpty()) {
            reverseDanglingConnections.removeAll(selectedReverseDanglingConnections);
            selectedReverseDanglingConnections.clear();
            modified = true;
        }

        // Remove selected free connections
        if (!selectedFreeConnections.isEmpty()) {
            freeConnections.removeAll(selectedFreeConnections);
            selectedFreeConnections.clear();
            modified = true;
        }

        if (modified) {
            notifyModified();
            canvas.redraw();
        }
    }

    @Override
    protected void updatePreviewFromNode(PipelineNode node) {
        // Get a thread-safe clone of the output mat
        Mat outputMat = node.getOutputMatClone();
        if (outputMat != null) {
            try {
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
            } finally {
                outputMat.release();
            }
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
