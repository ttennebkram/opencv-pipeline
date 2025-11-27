package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.model.Connection;
import com.ttennebkram.pipeline.registry.NodeInfo;
import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.*;
import org.eclipse.swt.widgets.Display;
import org.eclipse.swt.widgets.Shell;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * Container Node - encapsulates a sub-pipeline of nodes.
 * Double-click to open a new editor window showing the internal pipeline.
 * Has single input and output (expandable later).
 *
 * The container itself has NO processing thread - it delegates to:
 * - ContainerInputNode thread (pulls from external input)
 * - Child node threads (process internally)
 * - ContainerOutputNode thread (pushes to external output)
 */
@NodeInfo(name = "Container", category = "Container", aliases = {"SubPipeline", "Group"})
public class ContainerNode extends ProcessingNode {

    // Internal pipeline components
    private List<PipelineNode> childNodes;
    private List<Connection> childConnections;

    // Boundary nodes (always present in every container)
    private ContainerInputNode boundaryInput;
    private ContainerOutputNode boundaryOutput;

    // Container dimensions (slightly larger than regular nodes)
    private static final int CONTAINER_WIDTH = 180;
    private static final int CONTAINER_HEIGHT = 120;

    // Container display name
    private String containerName = "Container";

    // Path to external JSON file for internal pipeline
    private String pipelineFilePath = null;

    // Callback to open container editor (set by PipelineEditor)
    private Runnable onEditSubDiagram;

    // Callback when properties change (set by PipelineEditor to trigger dirty flag and redraw)
    private Runnable onPropertiesChanged;

    public ContainerNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Container", x, y);
        this.width = CONTAINER_WIDTH;
        this.height = CONTAINER_HEIGHT;

        // Initialize internal lists
        this.childNodes = new ArrayList<>();
        this.childConnections = new ArrayList<>();

        // Create boundary nodes (positioned at default locations inside)
        this.boundaryInput = new ContainerInputNode(shell, display, 50, 100);
        this.boundaryInput.setParentContainer(this);

        this.boundaryOutput = new ContainerOutputNode(display, shell, 400, 100);
        this.boundaryOutput.setParentContainer(this);
    }

    public String getContainerName() {
        return containerName;
    }

    public void setContainerName(String name) {
        this.containerName = name;
        this.name = name; // Also update the ProcessingNode name
    }

    public String getPipelineFilePath() {
        return pipelineFilePath;
    }

    public void setPipelineFilePath(String path) {
        this.pipelineFilePath = path;
    }

    public void setOnEditSubDiagram(Runnable callback) {
        this.onEditSubDiagram = callback;
    }

    public void setOnPropertiesChanged(Runnable callback) {
        this.onPropertiesChanged = callback;
    }

    public List<PipelineNode> getChildNodes() {
        return childNodes;
    }

    public List<Connection> getChildConnections() {
        return childConnections;
    }

    public ContainerInputNode getBoundaryInput() {
        return boundaryInput;
    }

    public ContainerOutputNode getBoundaryOutput() {
        return boundaryOutput;
    }

    public void addChildNode(PipelineNode node) {
        childNodes.add(node);
    }

    public void removeChildNode(PipelineNode node) {
        childNodes.remove(node);
    }

    public void addChildConnection(Connection connection) {
        childConnections.add(connection);
    }

    public void removeChildConnection(Connection connection) {
        childConnections.remove(connection);
    }

    /**
     * Process - containers don't process directly.
     * The processing happens through internal pipeline when running.
     * This method handles the case when container is not running
     * (just passes through).
     */
    @Override
    public Mat process(Mat input) {
        // When not running the internal pipeline, just pass through
        return input != null ? input.clone() : null;
    }

    /**
     * Start processing the internal pipeline.
     * Wires up queues and starts all child nodes.
     */
    @Override
    public void startProcessing() {
        if (running.get()) {
            return;
        }

        running.set(true);
        workUnitsCompleted = 0;

        // Reset work counters on all internal nodes
        boundaryInput.resetWorkUnitsCompleted();
        for (PipelineNode child : childNodes) {
            child.resetWorkUnitsCompleted();
        }
        boundaryOutput.resetWorkUnitsCompleted();

        // Wire the container input queue to the boundary input node
        boundaryInput.setContainerInputQueue(inputQueue);

        // Activate all internal connections
        for (Connection conn : childConnections) {
            conn.activate();
        }

        // Set up input node references for backpressure/slowdown signaling
        for (Connection conn : childConnections) {
            if (conn.target != null && conn.source != null) {
                if (conn.inputIndex == 2) {
                    conn.target.setInputNode2(conn.source);
                } else {
                    conn.target.setInputNode(conn.source);
                }
            }
        }

        // Wire the boundary output to the container's output queue
        boundaryOutput.setContainerOutputQueue(outputQueue);

        // Start boundary input first (it's the source)
        boundaryInput.startProcessing();

        // Start all child nodes
        for (PipelineNode child : childNodes) {
            child.startProcessing();
        }

        // Start boundary output last
        boundaryOutput.startProcessing();
    }

    /**
     * Stop processing the internal pipeline.
     */
    @Override
    public void stopProcessing() {
        if (!running.get()) {
            return;
        }

        running.set(false);

        // Stop in reverse order
        boundaryOutput.stopProcessing();

        for (PipelineNode child : childNodes) {
            child.stopProcessing();
        }

        boundaryInput.stopProcessing();

        // Clear input node references
        for (PipelineNode child : childNodes) {
            child.setInputNode(null);
            child.setInputNode2(null);
        }
        boundaryOutput.setInputNode(null);

        // Deactivate connections (but keep queues for inspection)
        for (Connection conn : childConnections) {
            conn.deactivate();
        }
    }

    @Override
    public String getDescription() {
        int nodeCount = childNodes.size() + 2; // +2 for boundary input and output nodes
        return "Container with " + nodeCount + " internal node" + (nodeCount == 1 ? "" : "s") +
               ".\nDouble-click to open the internal editor.";
    }

    @Override
    public String getDisplayName() {
        return "Container";
    }

    @Override
    public String getCategory() {
        return "Container";
    }

    /**
     * Paint the container node with distinctive styling.
     */
    @Override
    public void paint(GC gc) {
        // Draw node background - light lavender for containers
        Color bgColor = new Color(230, 230, 250); // Lavender
        gc.setBackground(bgColor);
        gc.fillRoundRectangle(x, y, width, height, 10, 10);
        bgColor.dispose();

        // Draw thicker border - purple-gray for containers
        Color borderColor = new Color(120, 100, 140);
        gc.setForeground(borderColor);
        gc.setLineWidth(3);
        gc.drawRoundRectangle(x, y, width, height, 10, 10);
        borderColor.dispose();
        gc.setLineWidth(1);

        // Draw title with container name and node count
        gc.setForeground(display.getSystemColor(SWT.COLOR_BLACK));
        Font boldFont = new Font(display, "Arial", 10, SWT.BOLD);
        gc.setFont(boldFont);
        int nodeCount = childNodes.size() + 2; // +2 for boundary input and output nodes
        String titleText = containerName + " (" + nodeCount + " node" + (nodeCount == 1 ? "" : "s") + ")";
        gc.drawString(titleText, x + 10, y + 5, true);
        boldFont.dispose();

        // Draw container icon (nested rectangles) in corner
        drawContainerIcon(gc, x + width - 25, y + 5);

        // Draw thread priority label on second line
        Font smallFont = new Font(display, "Arial", 8, SWT.NORMAL);
        gc.setFont(smallFont);
        gc.setForeground(display.getSystemColor(SWT.COLOR_DARK_GRAY));
        gc.drawString(getThreadPriorityLabel(), x + 10, y + 22, true);
        smallFont.dispose();

        // Draw input read count on the left side (use boundary input's count since container delegates to it)
        Font statsFont = new Font(display, "Arial", 7, SWT.NORMAL);
        gc.setFont(statsFont);
        gc.setForeground(display.getSystemColor(SWT.COLOR_DARK_GRAY));
        int statsX = x + 5;
        long inputCount = boundaryInput.getInputReads1();
        if (inputCount >= 1000) {
            gc.drawString("In:", statsX, y + 38, true);
            gc.drawString(formatNumber(inputCount), statsX, y + 48, true);
        } else {
            gc.drawString("In:" + formatNumber(inputCount), statsX, y + 38, true);
        }
        statsFont.dispose();

        // Draw thumbnail if available (centered horizontally)
        Rectangle bounds = getThumbnailBounds();
        if (bounds != null) {
            int thumbX = x + (width - bounds.width) / 2;
            int thumbY = y + 38;
            drawThumbnail(gc, thumbX, thumbY);
        } else {
            // Draw placeholder text
            gc.setForeground(display.getSystemColor(SWT.COLOR_GRAY));
            Font tinyFont = new Font(display, "Arial", 8, SWT.ITALIC);
            gc.setFont(tinyFont);
            gc.drawString("(double-click to edit)", x + 20, y + 55, true);
            tinyFont.dispose();
        }

        // Draw connection points
        drawConnectionPoints(gc);
    }

    /**
     * Draw a container icon (nested rectangles).
     */
    private void drawContainerIcon(GC gc, int iconX, int iconY) {
        Color iconColor = new Color(100, 80, 120);
        gc.setForeground(iconColor);
        gc.setLineWidth(1);

        // Outer rectangle
        gc.drawRectangle(iconX, iconY, 16, 12);

        // Inner rectangle (offset)
        gc.drawRectangle(iconX + 3, iconY + 3, 10, 6);

        iconColor.dispose();
    }

    /**
     * Check if a point is within the container icon area.
     */
    public boolean isOnContainerIcon(Point p) {
        int iconX = x + width - 25;
        int iconY = y + 5;
        // Add a few pixels padding for easier clicking
        return p.x >= iconX - 2 && p.x <= iconX + 18 &&
               p.y >= iconY - 2 && p.y <= iconY + 14;
    }

    /**
     * Show properties dialog for container - allows setting name and pipeline file path.
     */
    @Override
    public void showPropertiesDialog() {
        org.eclipse.swt.widgets.Shell dialog = new org.eclipse.swt.widgets.Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Container Properties");
        dialog.setSize(450, 200);
        dialog.setLayout(new org.eclipse.swt.layout.GridLayout(3, false));

        // Container name
        org.eclipse.swt.widgets.Label nameLabel = new org.eclipse.swt.widgets.Label(dialog, SWT.NONE);
        nameLabel.setText("Name:");
        org.eclipse.swt.widgets.Text nameText = new org.eclipse.swt.widgets.Text(dialog, SWT.BORDER);
        nameText.setText(containerName);
        org.eclipse.swt.layout.GridData nameGd = new org.eclipse.swt.layout.GridData(SWT.FILL, SWT.CENTER, true, false);
        nameGd.horizontalSpan = 2;
        nameText.setLayoutData(nameGd);

        // Pipeline file path
        org.eclipse.swt.widgets.Label fileLabel = new org.eclipse.swt.widgets.Label(dialog, SWT.NONE);
        fileLabel.setText("Pipeline File:");
        org.eclipse.swt.widgets.Text fileText = new org.eclipse.swt.widgets.Text(dialog, SWT.BORDER);
        fileText.setText(pipelineFilePath != null ? pipelineFilePath : "");
        fileText.setLayoutData(new org.eclipse.swt.layout.GridData(SWT.FILL, SWT.CENTER, true, false));

        org.eclipse.swt.widgets.Button browseBtn = new org.eclipse.swt.widgets.Button(dialog, SWT.PUSH);
        browseBtn.setText("Browse...");
        browseBtn.addListener(SWT.Selection, e -> {
            org.eclipse.swt.widgets.FileDialog fd = new org.eclipse.swt.widgets.FileDialog(dialog, SWT.SAVE);
            fd.setText("Select Pipeline File");
            fd.setFilterExtensions(new String[]{"*.json", "*.*"});
            fd.setFilterNames(new String[]{"Pipeline Files (*.json)", "All Files (*.*)"});
            if (pipelineFilePath != null && !pipelineFilePath.isEmpty()) {
                java.io.File current = new java.io.File(pipelineFilePath);
                fd.setFilterPath(current.getParent());
                fd.setFileName(current.getName());
            }
            String selected = fd.open();
            if (selected != null) {
                fileText.setText(selected);
            }
        });

        // Edit Sub-Diagram button
        org.eclipse.swt.widgets.Button editBtn = new org.eclipse.swt.widgets.Button(dialog, SWT.PUSH);
        editBtn.setText("Edit Sub-Diagram...");
        org.eclipse.swt.layout.GridData editGd = new org.eclipse.swt.layout.GridData(SWT.LEFT, SWT.CENTER, false, false);
        editGd.horizontalSpan = 3;
        editBtn.setLayoutData(editGd);
        editBtn.addListener(SWT.Selection, e -> {
            if (onEditSubDiagram != null) {
                dialog.dispose();
                onEditSubDiagram.run();
            }
        });

        // Buttons
        org.eclipse.swt.widgets.Composite btnPanel = new org.eclipse.swt.widgets.Composite(dialog, SWT.NONE);
        org.eclipse.swt.layout.GridData btnGd = new org.eclipse.swt.layout.GridData(SWT.RIGHT, SWT.CENTER, true, false);
        btnGd.horizontalSpan = 3;
        btnPanel.setLayoutData(btnGd);
        btnPanel.setLayout(new org.eclipse.swt.layout.GridLayout(2, true));

        org.eclipse.swt.widgets.Button okBtn = new org.eclipse.swt.widgets.Button(btnPanel, SWT.PUSH);
        okBtn.setText("OK");
        okBtn.setLayoutData(new org.eclipse.swt.layout.GridData(80, SWT.DEFAULT));
        okBtn.addListener(SWT.Selection, e -> {
            setContainerName(nameText.getText().trim());
            String filePath = fileText.getText().trim();
            pipelineFilePath = filePath.isEmpty() ? null : filePath;
            dialog.dispose();
            // Notify that properties changed (triggers dirty flag and redraw)
            if (onPropertiesChanged != null) {
                onPropertiesChanged.run();
            }
        });

        org.eclipse.swt.widgets.Button cancelBtn = new org.eclipse.swt.widgets.Button(btnPanel, SWT.PUSH);
        cancelBtn.setText("Cancel");
        cancelBtn.setLayoutData(new org.eclipse.swt.layout.GridData(80, SWT.DEFAULT));
        cancelBtn.addListener(SWT.Selection, e -> dialog.dispose());

        dialog.setDefaultButton(okBtn);

        // Center on parent
        Rectangle parentBounds = shell.getBounds();
        Rectangle dialogBounds = dialog.getBounds();
        dialog.setLocation(
            parentBounds.x + (parentBounds.width - dialogBounds.width) / 2,
            parentBounds.y + (parentBounds.height - dialogBounds.height) / 2
        );

        dialog.open();
    }

    public void dispose() {
        // Dispose all child node thumbnails
        for (PipelineNode child : childNodes) {
            child.disposeThumbnail();
        }

        // Dispose boundary node thumbnails
        if (boundaryInput != null) {
            boundaryInput.disposeThumbnail();
        }
        if (boundaryOutput != null) {
            boundaryOutput.disposeThumbnail();
        }
    }

    /**
     * Called by ContainerOutputNode when it processes a frame.
     * Updates the container's thumbnail to show the output.
     */
    public void onOutputFrame(Mat frame) {
        if (frame != null && !frame.empty()) {
            // Clone and store for thumbnail
            setOutputMat(frame.clone());
            // Notify for preview (clone to avoid threading issues)
            notifyFrame(frame.clone());
        }
    }

    @Override
    public String getInputTooltip() {
        return "Container Input " + CONNECTION_DATA_TYPE;
    }

    @Override
    public String getOutputTooltip(int index) {
        return "Container Output " + CONNECTION_DATA_TYPE;
    }

    /**
     * Serialize the container - just save the reference to external pipeline file.
     */
    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("containerName", containerName);

        // Save reference to external pipeline file (if set)
        if (pipelineFilePath != null && !pipelineFilePath.isEmpty()) {
            json.addProperty("pipelineFile", pipelineFilePath);
        }

        // Also save boundary node positions
        json.addProperty("boundaryInputY", boundaryInput.getY());
        json.addProperty("boundaryOutputY", boundaryOutput.getY());
    }

    /**
     * Deserialize the container - load the reference to external pipeline file.
     * The actual internal pipeline loading is done separately via loadInternalPipeline().
     */
    @Override
    public void deserializeProperties(JsonObject json) {
        if (json.has("containerName")) {
            setContainerName(json.get("containerName").getAsString());
        }

        if (json.has("pipelineFile")) {
            pipelineFilePath = json.get("pipelineFile").getAsString();
        }

        // Restore boundary node positions
        if (json.has("boundaryInputY")) {
            boundaryInput.setY(json.get("boundaryInputY").getAsInt());
        }
        if (json.has("boundaryOutputY")) {
            boundaryOutput.setY(json.get("boundaryOutputY").getAsInt());
        }
    }

    /**
     * Get accumulated work units from all internal nodes.
     */
    @Override
    public long getWorkUnitsCompleted() {
        long total = 0;
        total += boundaryInput.getWorkUnitsCompleted();
        for (PipelineNode child : childNodes) {
            total += child.getWorkUnitsCompleted();
        }
        total += boundaryOutput.getWorkUnitsCompleted();
        return total;
    }

    /**
     * Save thumbnails for all internal nodes (boundary + children).
     * Uses a subdirectory per container to avoid index collisions.
     */
    @Override
    public void saveThumbnailToCache(String cacheDir, int nodeIndex) {
        // First, save our own thumbnail (container's output)
        super.saveThumbnailToCache(cacheDir, nodeIndex);

        // Create a subdirectory for this container's internal nodes
        String containerCacheDir = cacheDir + java.io.File.separator + "container_" + nodeIndex;
        java.io.File containerDir = new java.io.File(containerCacheDir);
        if (!containerDir.exists()) {
            containerDir.mkdirs();
        }

        // Save boundary input thumbnail (index 0)
        boundaryInput.saveThumbnailToCache(containerCacheDir, 0);

        // Save child node thumbnails (indices 1..N)
        for (int i = 0; i < childNodes.size(); i++) {
            childNodes.get(i).saveThumbnailToCache(containerCacheDir, i + 1);
        }

        // Save boundary output thumbnail (last index)
        boundaryOutput.saveThumbnailToCache(containerCacheDir, childNodes.size() + 1);
    }

    /**
     * Load thumbnails for all internal nodes (boundary + children).
     */
    @Override
    public boolean loadThumbnailFromCache(String cacheDir, int nodeIndex) {
        // First, load our own thumbnail
        boolean loaded = super.loadThumbnailFromCache(cacheDir, nodeIndex);

        // Load from subdirectory for this container's internal nodes
        String containerCacheDir = cacheDir + java.io.File.separator + "container_" + nodeIndex;
        java.io.File containerDir = new java.io.File(containerCacheDir);
        if (!containerDir.exists()) {
            return loaded;
        }

        // Load boundary input thumbnail (index 0)
        boundaryInput.loadThumbnailFromCache(containerCacheDir, 0);

        // Load child node thumbnails (indices 1..N)
        for (int i = 0; i < childNodes.size(); i++) {
            childNodes.get(i).loadThumbnailFromCache(containerCacheDir, i + 1);
        }

        // Load boundary output thumbnail (last index)
        boundaryOutput.loadThumbnailFromCache(containerCacheDir, childNodes.size() + 1);

        return loaded;
    }
}
