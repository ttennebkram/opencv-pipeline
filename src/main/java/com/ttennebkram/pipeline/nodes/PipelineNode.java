package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.registry.NodeInfo;
import com.ttennebkram.pipeline.serialization.NodeSerializable;
import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.*;
import org.eclipse.swt.widgets.Display;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Base class for all pipeline nodes.
 * Implements NodeSerializable to handle common property serialization.
 */
public abstract class PipelineNode implements NodeSerializable {
    // Connection point data type suffix for tooltips
    protected static final String CONNECTION_DATA_TYPE = "Images";

    // Selection highlight color (RGB) - used for nodes and connections
    public static final int SELECTION_COLOR_R = 0;
    public static final int SELECTION_COLOR_G = 120;
    public static final int SELECTION_COLOR_B = 215;

    // Node dimension constants
    public static final int PROCESSING_NODE_THUMB_WIDTH = 120;
    public static final int PROCESSING_NODE_THUMB_HEIGHT = 80;
    public static final int SOURCE_NODE_THUMB_WIDTH = 280;
    public static final int SOURCE_NODE_THUMB_HEIGHT = 90;

    public static final int NODE_WIDTH = PROCESSING_NODE_THUMB_WIDTH + 60;  // thumbnail + space for queue stats
    public static final int NODE_HEIGHT = PROCESSING_NODE_THUMB_HEIGHT + 40; // thumbnail + 25px title + 15px bottom
    public static final int SOURCE_NODE_HEIGHT = SOURCE_NODE_THUMB_HEIGHT + 32; // thumbnail + 22px title + 10px bottom

    protected Display display;
    public int x, y;
    public int width = NODE_WIDTH;
    public int height = NODE_HEIGHT;
    protected Image thumbnail;
    protected Mat outputMat;
    protected volatile boolean thumbnailUpdatePending = false;

    // Thread management for pipeline execution
    protected Thread processingThread;
    protected AtomicBoolean running = new AtomicBoolean(false);
    protected BlockingQueue<Mat> inputQueue;
    protected BlockingQueue<Mat> outputQueue;  // Primary output (index 0) - kept for backwards compatibility
    protected BlockingQueue<Mat>[] outputQueues;  // Array for multi-output nodes
    protected int outputCount = 1;  // Number of outputs this node has (default 1)
    protected volatile long frameDelayMs = 0; // Frame rate throttling (0 = no delay)
    protected int threadPriority = Thread.NORM_PRIORITY; // Thread priority (1-10, default 5)
    protected int originalPriority = Thread.NORM_PRIORITY; // Original priority before backpressure adjustments
    protected int lastRunningPriority = Thread.NORM_PRIORITY; // Last actual running priority (persists after stop)

    // Backpressure timing constants
    /** Cooldown between priority reductions (going down) - 1 second per step */
    protected static final long PRIORITY_LOWER_COOLDOWN_MS = 1000;
    /** Cooldown between priority increases (going up) - 10 seconds per step */
    protected static final long PRIORITY_RAISE_COOLDOWN_MS = 10000;
    /** How long after last slowdown signal before starting recovery */
    protected static final long SLOWDOWN_RECOVERY_MS = 10000;
    protected long lastPriorityAdjustmentTime = 0;
    protected PipelineNode inputNode = null; // Reference to upstream node
    protected PipelineNode inputNode2 = null; // Reference to second upstream node for dual-input nodes

    // Slowdown signaling (cascading backpressure)
    protected volatile long lastSlowdownReceivedTime = 0; // When we last received a slowdown signal
    protected volatile boolean inSlowdownMode = false; // Whether we're currently slowed due to downstream request
    protected int slowdownPriorityReduction = 0; // How much we've reduced priority due to slowdown signals

    // Work unit tracking
    protected long workUnitsCompleted = 0; // Count of work units completed (persists across runs)

    // Input read counters (persists across runs)
    protected long inputReads1 = 0; // Frames read from input queue 1
    protected long inputReads2 = 0; // Frames read from input queue 2 (for dual-input nodes)

    // Callback for frame updates (used for preview)
    protected java.util.function.Consumer<Mat> onFrameCallback;

    public abstract void paint(GC gc);

    public abstract String getNodeName();

    /**
     * Get the background color for this node.
     * Subclasses can override to customize appearance.
     */
    protected Color getBackgroundColor() {
        return new Color(230, 255, 230); // Default light green
    }

    /**
     * Get the border color for this node.
     * Subclasses can override to customize appearance.
     */
    protected Color getBorderColor() {
        return new Color(0, 100, 0); // Default dark green
    }

    public boolean containsPoint(Point p) {
        return p.x >= x && p.x <= x + width && p.y >= y && p.y <= y + height;
    }

    /**
     * Get the primary output point (index 0).
     * For single-output nodes, this is at the vertical center.
     * For multi-output nodes, outputs are distributed vertically.
     */
    public Point getOutputPoint() {
        return getOutputPoint(0);
    }

    /**
     * Get the output point for a specific output index.
     * Outputs are distributed vertically on the right side of the node.
     */
    public Point getOutputPoint(int index) {
        if (index < 0 || index >= outputCount) {
            return null;
        }
        if (outputCount == 1) {
            return new Point(x + width, y + height / 2);
        }
        // Distribute outputs vertically, evenly spaced
        int spacing = height / (outputCount + 1);
        int yPos = y + spacing * (index + 1);
        return new Point(x + width, yPos);
    }

    public Point getInputPoint() {
        return new Point(x, y + height / 2);
    }

    /**
     * Get the second input point for dual-input nodes.
     * Default implementation returns null (single input).
     * Override in DualInputNode subclasses.
     */
    public Point getInputPoint2() {
        return null;
    }

    /**
     * Check if this node has a second input.
     */
    public boolean hasDualInput() {
        return getInputPoint2() != null;
    }

    /**
     * Get the number of outputs this node has.
     */
    public int getOutputCount() {
        return outputCount;
    }

    /**
     * Check if this node has multiple outputs.
     */
    public boolean hasMultipleOutputs() {
        return outputCount > 1;
    }

    /**
     * Connection point hit radius for tooltip detection.
     */
    protected static final int CONNECTION_HIT_RADIUS = 10;

    /**
     * Check if a point is near the primary input connection point.
     * Returns true if within CONNECTION_HIT_RADIUS pixels.
     */
    public boolean isNearInputPoint(int px, int py) {
        Point input = getInputPoint();
        if (input == null) return false;
        double dist = Math.sqrt(Math.pow(px - input.x, 2) + Math.pow(py - input.y, 2));
        return dist <= CONNECTION_HIT_RADIUS;
    }

    /**
     * Check if a point is near the secondary input connection point (for dual-input nodes).
     * Returns true if within CONNECTION_HIT_RADIUS pixels.
     */
    public boolean isNearInputPoint2(int px, int py) {
        Point input2 = getInputPoint2();
        if (input2 == null) return false;
        double dist = Math.sqrt(Math.pow(px - input2.x, 2) + Math.pow(py - input2.y, 2));
        return dist <= CONNECTION_HIT_RADIUS;
    }

    /**
     * Check if a point is near the output connection point.
     * Returns the output index (0-based) if near, or -1 if not near any output.
     */
    public int getOutputIndexNear(int px, int py) {
        for (int i = 0; i < outputCount; i++) {
            Point output = getOutputPoint(i);
            if (output != null) {
                double dist = Math.sqrt(Math.pow(px - output.x, 2) + Math.pow(py - output.y, 2));
                if (dist <= CONNECTION_HIT_RADIUS) {
                    return i;
                }
            }
        }
        return -1;
    }

    /**
     * Get tooltip text for the primary input connection point.
     * Subclasses can override to provide custom tooltips.
     */
    public String getInputTooltip() {
        return "Input " + CONNECTION_DATA_TYPE;
    }

    /**
     * Get tooltip text for the secondary input connection point (dual-input nodes).
     * Subclasses can override to provide custom tooltips.
     */
    public String getInput2Tooltip() {
        return "Input 2 " + CONNECTION_DATA_TYPE;
    }

    /**
     * Get tooltip text for the output connection point at the given index.
     * Subclasses can override to provide custom tooltips.
     */
    public String getOutputTooltip(int index) {
        if (outputCount == 1) {
            return "Output " + CONNECTION_DATA_TYPE;
        }
        return "Output " + (index + 1) + " " + CONNECTION_DATA_TYPE;
    }

    public int getX() {
        return x;
    }

    public void setX(int x) {
        this.x = x;
    }

    public int getY() {
        return y;
    }

    public void setY(int y) {
        this.y = y;
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    // Draw connection points (circles) on the node - always visible
    protected void drawConnectionPoints(GC gc) {
        int radius = 6;  // Slightly larger for visibility

        // Draw input point on left side (blue tint for input)
        Point input = getInputPoint();
        gc.setBackground(new Color(200, 220, 255));  // Light blue fill
        gc.fillOval(input.x - radius, input.y - radius, radius * 2, radius * 2);
        gc.setForeground(new Color(70, 100, 180));   // Blue border
        gc.setLineWidth(2);
        gc.drawOval(input.x - radius, input.y - radius, radius * 2, radius * 2);

        // Draw output point on right side (orange tint for output)
        Point output = getOutputPoint();
        gc.setBackground(new Color(255, 230, 200)); // Light orange fill
        gc.fillOval(output.x - radius, output.y - radius, radius * 2, radius * 2);
        gc.setForeground(new Color(200, 120, 50));  // Orange border
        gc.drawOval(output.x - radius, output.y - radius, radius * 2, radius * 2);
        gc.setLineWidth(1);  // Reset line width
    }

    // Draw selection highlight around node
    public void drawSelectionHighlight(GC gc, boolean isSelected) {
        if (isSelected) {
            gc.setForeground(new Color(SELECTION_COLOR_R, SELECTION_COLOR_G, SELECTION_COLOR_B));
            gc.setLineWidth(3);
            gc.drawRoundRectangle(x - 3, y - 3, width + 6, height + 6, 13, 13);
        }
    }

    // Lock object for outputMat access (synchronizing on 'this' could cause deadlocks with other synchronized methods)
    private final Object outputMatLock = new Object();

    // Lock object for thumbnail access - separate from outputMat to avoid holding Mat lock during UI operations
    private final Object thumbnailLock = new Object();

    // Pending thumbnail data for UI thread to create Image from
    private volatile ImageData pendingThumbnailData = null;

    public void setOutputMat(Mat mat) {
        synchronized (outputMatLock) {
            // Release old outputMat if it exists
            if (this.outputMat != null && !this.outputMat.empty()) {
                this.outputMat.release();
            }
            this.outputMat = mat;
            prepareThumbnailData();
        }
    }

    public Mat getOutputMat() {
        synchronized (outputMatLock) {
            return outputMat;
        }
    }

    /**
     * Get a clone of the output Mat for safe external use.
     * Returns null if no output or if empty.
     */
    public Mat getOutputMatClone() {
        synchronized (outputMatLock) {
            if (outputMat != null && !outputMat.empty()) {
                return outputMat.clone();
            }
            return null;
        }
    }

    protected void updateThumbnail() {
        synchronized (outputMatLock) {
            prepareThumbnailData();
        }
    }

    /**
     * Prepare thumbnail ImageData from outputMat. This can be called from any thread.
     * The actual Image creation happens on the UI thread when drawThumbnail is called.
     * Must be called with outputMatLock held.
     */
    private void prepareThumbnailData() {
        if (outputMat == null || outputMat.empty()) {
            return;
        }

        // Create thumbnail data - clone the Mat first to avoid issues if outputMat changes
        Mat source = outputMat.clone();
        try {
            Mat resized = new Mat();
            double scale = Math.min((double) PROCESSING_NODE_THUMB_WIDTH / source.width(),
                                    (double) PROCESSING_NODE_THUMB_HEIGHT / source.height());
            Imgproc.resize(source, resized,
                new Size(source.width() * scale, source.height() * scale));

            // Convert to RGB
            Mat rgb = new Mat();
            if (resized.channels() == 3) {
                Imgproc.cvtColor(resized, rgb, Imgproc.COLOR_BGR2RGB);
            } else if (resized.channels() == 1) {
                Imgproc.cvtColor(resized, rgb, Imgproc.COLOR_GRAY2RGB);
            } else {
                rgb = resized;
            }

            int w = rgb.width();
            int h = rgb.height();
            byte[] data = new byte[w * h * 3];
            rgb.get(0, 0, data);

            // Create ImageData (this is just data, not an SWT resource - safe from any thread)
            PaletteData palette = new PaletteData(0xFF0000, 0x00FF00, 0x0000FF);
            ImageData imageData = new ImageData(w, h, 24, palette);

            // Copy data row by row to handle scanline padding
            int bytesPerLine = imageData.bytesPerLine;
            for (int row = 0; row < h; row++) {
                int srcOffset = row * w * 3;
                int dstOffset = row * bytesPerLine;
                for (int col = 0; col < w; col++) {
                    int srcIdx = srcOffset + col * 3;
                    int dstIdx = dstOffset + col * 3;
                    imageData.data[dstIdx] = data[srcIdx];         // R
                    imageData.data[dstIdx + 1] = data[srcIdx + 1]; // G
                    imageData.data[dstIdx + 2] = data[srcIdx + 2]; // B
                }
            }

            // Store pending data - UI thread will create Image from this
            synchronized (thumbnailLock) {
                pendingThumbnailData = imageData;
            }

            // Clean up intermediate Mats
            resized.release();
            if (rgb != resized) {
                rgb.release();
            }
        } finally {
            source.release();
        }
    }

    /**
     * Get the thumbnail bounds (width and height) if available.
     * Returns null if no thumbnail exists. Thread-safe.
     */
    protected Rectangle getThumbnailBounds() {
        synchronized (thumbnailLock) {
            if (thumbnail != null && !thumbnail.isDisposed()) {
                return thumbnail.getBounds();
            }
            // If we have pending data, return bounds based on that
            if (pendingThumbnailData != null) {
                return new Rectangle(0, 0, pendingThumbnailData.width, pendingThumbnailData.height);
            }
            return null;
        }
    }

    /**
     * Draw the thumbnail. This must be called from the UI thread.
     * If there's pending thumbnail data, creates the Image first.
     */
    protected void drawThumbnail(GC gc, int thumbX, int thumbY) {
        synchronized (thumbnailLock) {
            // Check if we have pending data to create a new thumbnail
            if (pendingThumbnailData != null) {
                // Dispose old thumbnail first
                if (thumbnail != null && !thumbnail.isDisposed()) {
                    thumbnail.dispose();
                }
                // Create new thumbnail on UI thread (this is the safe place to do it)
                try {
                    thumbnail = new Image(display, pendingThumbnailData);
                } catch (Exception e) {
                    // Display may be disposed during shutdown
                    thumbnail = null;
                }
                pendingThumbnailData = null;
            }

            // Draw the thumbnail
            if (thumbnail != null && !thumbnail.isDisposed()) {
                gc.drawImage(thumbnail, thumbX, thumbY);
            }
        }
    }

    public void disposeThumbnail() {
        synchronized (thumbnailLock) {
            if (thumbnail != null && !thumbnail.isDisposed()) {
                thumbnail.dispose();
            }
            pendingThumbnailData = null;
        }
    }

    /**
     * Set pending thumbnail data from subclasses (e.g., when loading from cache).
     * The actual Image will be created on the UI thread when drawThumbnail is called.
     */
    protected void setPendingThumbnailData(ImageData data) {
        synchronized (thumbnailLock) {
            pendingThumbnailData = data;
        }
    }

    /**
     * Save thumbnail to cache directory.
     * Default implementation does nothing - subclasses override as needed.
     */
    public void saveThumbnailToCache(String cacheDir, int nodeIndex) {
        // Default: no caching
    }

    /**
     * Load thumbnail from cache directory.
     * Default implementation returns false - subclasses override as needed.
     */
    public boolean loadThumbnailFromCache(String cacheDir, int nodeIndex) {
        return false;
    }

    // Queue management for pipeline connections
    public void setInputQueue(BlockingQueue<Mat> queue) {
        this.inputQueue = queue;
    }

    /**
     * Set the primary output queue (index 0).
     * For backwards compatibility with single-output nodes.
     */
    public void setOutputQueue(BlockingQueue<Mat> queue) {
        this.outputQueue = queue;
        // Also set in array if multi-output
        if (outputQueues != null && outputQueues.length > 0) {
            outputQueues[0] = queue;
        }
    }

    /**
     * Set an output queue by index.
     * Index 0 is the primary output.
     */
    @SuppressWarnings("unchecked")
    public void setOutputQueue(int index, BlockingQueue<Mat> queue) {
        if (index < 0 || index >= outputCount) {
            return;
        }
        // Initialize array if needed
        if (outputQueues == null) {
            outputQueues = (BlockingQueue<Mat>[]) new BlockingQueue[outputCount];
        }
        outputQueues[index] = queue;
        // Keep primary output in sync
        if (index == 0) {
            outputQueue = queue;
        }
    }

    public BlockingQueue<Mat> getInputQueue() {
        return inputQueue;
    }

    /**
     * Get the primary output queue (index 0).
     */
    public BlockingQueue<Mat> getOutputQueue() {
        return outputQueue;
    }

    /**
     * Get an output queue by index.
     */
    public BlockingQueue<Mat> getOutputQueue(int index) {
        if (outputQueues != null && index >= 0 && index < outputQueues.length) {
            return outputQueues[index];
        }
        // Fallback to primary for index 0
        if (index == 0) {
            return outputQueue;
        }
        return null;
    }

    /**
     * Initialize the output queues array for a multi-output node.
     * Call this when setting the output count.
     */
    @SuppressWarnings("unchecked")
    protected void initOutputQueues(int count) {
        this.outputCount = count;
        this.outputQueues = (BlockingQueue<Mat>[]) new BlockingQueue[count];
    }

    public void setFrameDelayMs(long delayMs) {
        this.frameDelayMs = delayMs;
    }

    public long getFrameDelayMs() {
        return frameDelayMs;
    }

    public void setThreadPriority(int priority) {
        this.threadPriority = Math.max(Thread.MIN_PRIORITY, Math.min(Thread.MAX_PRIORITY, priority));
        this.originalPriority = this.threadPriority; // Track original priority
        this.lastRunningPriority = this.threadPriority; // Initialize last running priority
    }

    public int getThreadPriority() {
        // Return actual running priority if thread is alive, otherwise return last running priority
        if (processingThread != null && processingThread.isAlive()) {
            int currentPriority = processingThread.getPriority();
            lastRunningPriority = currentPriority; // Track it
            return currentPriority;
        }
        return lastRunningPriority; // Return last known priority when stopped
    }

    public String getThreadPriorityLabel() {
        // Show raw priority number and work units completed
        // Use getWorkUnitsCompleted() to allow ContainerNode to return sum of child nodes
        return "Pri: " + getThreadPriority() + "   Work: " + formatNumber(getWorkUnitsCompleted());
    }

    /**
     * Format a number with commas for display (e.g., 1234567 -> "1,234,567").
     */
    protected static String formatNumber(long number) {
        return String.format("%,d", number);
    }

    public long getWorkUnitsCompleted() {
        return workUnitsCompleted;
    }

    public void setWorkUnitsCompleted(long count) {
        this.workUnitsCompleted = count;
    }

    public void resetWorkUnitsCompleted() {
        this.workUnitsCompleted = 0;
    }

    protected void incrementWorkUnits() {
        workUnitsCompleted++;
    }

    public long getInputReads1() {
        return inputReads1;
    }

    public void setInputReads1(long count) {
        this.inputReads1 = count;
    }

    protected void incrementInputReads1() {
        inputReads1++;
    }

    public long getInputReads2() {
        return inputReads2;
    }

    public void setInputReads2(long count) {
        this.inputReads2 = count;
    }

    protected void incrementInputReads2() {
        inputReads2++;
    }

    public void resetInputReads() {
        this.inputReads1 = 0;
        this.inputReads2 = 0;
    }

    public void setInputNode(PipelineNode node) {
        this.inputNode = node;
    }

    public PipelineNode getInputNode() {
        return inputNode;
    }

    public void setInputNode2(PipelineNode node) {
        this.inputNode2 = node;
    }

    public PipelineNode getInputNode2() {
        return inputNode2;
    }

    /**
     * Check output queue and apply progressive backpressure.
     * When this node's output queue backs up, progressively lower THIS node's priority.
     * - Queue size 5+: reduce by 1
     * - Queue size 10+: reduce by 2
     * - Queue size 15+: reduce by 3
     * - And so on, down to minimum priority of 1
     * This causes a cascading effect: upstream nodes will back up and lower their own priorities.
     *
     * If already at minimum priority and queue still backed up after cooldown, send slowdown
     * signal to upstream nodes.
     *
     * Important: Priority boosts are capped at the originalPriority, so a node configured
     * as low-priority will stay low-priority even when its queue drains.
     */
    protected void checkBackpressure() {
        // First, check if we should recover from slowdown mode
        checkSlowdownRecovery();

        // Calculate total queue size across all outputs
        int queueSize = 0;

        // Check primary output queue
        if (outputQueue != null) {
            queueSize += outputQueue.size();
        }

        // Check multi-output queues
        if (outputQueues != null) {
            for (BlockingQueue<Mat> q : outputQueues) {
                if (q != null && q != outputQueue) { // Avoid double-counting if same queue
                    queueSize += q.size();
                }
            }
        }

        // If no queues at all, nothing to do
        if (outputQueue == null && (outputQueues == null || outputQueues.length == 0)) {
            return;
        }

        if (processingThread != null && processingThread.isAlive()) {
            int currentPriority = processingThread.getPriority();
            long now = System.currentTimeMillis();

            // If in slowdown mode, don't raise priority - let checkSlowdownRecovery() handle it
            if (inSlowdownMode) {
                if (queueSize >= 5) {
                    if (currentPriority > Thread.MIN_PRIORITY) {
                        // Lower by 1 if cooldown has passed
                        if (now - lastPriorityAdjustmentTime >= PRIORITY_LOWER_COOLDOWN_MS) {
                            int newPriority = currentPriority - 1;
                            System.out.println("[" + timestamp() + "] [" + getClass().getSimpleName() + " " + getNodeName() + "] LOWERING priority: " +
                                currentPriority + " -> " + newPriority + " (queueSize=" + queueSize + ", inSlowdownMode)");
                            processingThread.setPriority(newPriority);
                            lastPriorityAdjustmentTime = now;
                            lastRunningPriority = newPriority;
                        }
                    } else {
                        // At min priority, signal upstream
                        if (now - lastPriorityAdjustmentTime >= PRIORITY_LOWER_COOLDOWN_MS) {
                            signalUpstreamSlowdown();
                            lastPriorityAdjustmentTime = now;
                        }
                    }
                }
                return; // Don't raise priority - let slowdown recovery handle it
            }

            // Normal backpressure logic (not in slowdown mode)
            if (queueSize >= 5) {
                // Queue backed up - lower priority by 1 if cooldown has passed
                if (currentPriority > Thread.MIN_PRIORITY) {
                    if (now - lastPriorityAdjustmentTime >= PRIORITY_LOWER_COOLDOWN_MS) {
                        int newPriority = currentPriority - 1;
                        System.out.println("[" + timestamp() + "] [" + getClass().getSimpleName() + " " + getNodeName() + "] LOWERING priority: " +
                            currentPriority + " -> " + newPriority + " (queueSize=" + queueSize + ")");
                        processingThread.setPriority(newPriority);
                        lastPriorityAdjustmentTime = now;
                        lastRunningPriority = newPriority;
                    }
                } else {
                    // At min priority and still backed up - signal upstream
                    if (now - lastPriorityAdjustmentTime >= PRIORITY_LOWER_COOLDOWN_MS) {
                        signalUpstreamSlowdown();
                        lastPriorityAdjustmentTime = now;
                    }
                }
            } else if (queueSize == 0) {
                // Queue empty - raise priority by 1 toward original if cooldown has passed
                if (currentPriority < originalPriority) {
                    if (now - lastPriorityAdjustmentTime >= PRIORITY_RAISE_COOLDOWN_MS) {
                        int newPriority = currentPriority + 1;
                        System.out.println("[" + timestamp() + "] [" + getClass().getSimpleName() + " " + getNodeName() + "] RAISING priority: " +
                            currentPriority + " -> " + newPriority + " (queueSize=" + queueSize + ")");
                        processingThread.setPriority(newPriority);
                        lastPriorityAdjustmentTime = now;
                        lastRunningPriority = newPriority;
                    }
                }
            }
            // If queue is 1-4, do nothing - hold current priority
        }
    }

    /**
     * Signal upstream nodes to slow down because this node is overwhelmed.
     * Called when this node is at minimum priority and still has backlog.
     */
    protected void signalUpstreamSlowdown() {
        System.out.println("[" + timestamp() + "] [" + getClass().getSimpleName() + " " + getNodeName() + "] SIGNALING SLOWDOWN to upstream nodes");
        if (inputNode != null) {
            inputNode.receiveSlowdownSignal();
        }
        if (inputNode2 != null) {
            inputNode2.receiveSlowdownSignal();
        }
    }

    /**
     * Receive a slowdown signal from a downstream node.
     * Reduces priority and enters slowdown mode for at least 10 seconds.
     */
    public synchronized void receiveSlowdownSignal() {
        long now = System.currentTimeMillis();
        lastSlowdownReceivedTime = now;
        inSlowdownMode = true;

        Thread pt = processingThread; // Local copy for thread safety
        if (pt != null && pt.isAlive()) {
            int currentPriority = pt.getPriority();

            if (currentPriority > Thread.MIN_PRIORITY) {
                // Reduce priority by 1
                int newPriority = currentPriority - 1;
                System.out.println("[" + timestamp() + "] [" + getClass().getSimpleName() + " " + getNodeName() + "] RECEIVED SLOWDOWN, " +
                    "lowering priority: " + currentPriority + " -> " + newPriority);
                pt.setPriority(newPriority);
                lastRunningPriority = newPriority;
                slowdownPriorityReduction++;
            } else {
                // Already at minimum priority, cascade the slowdown upstream
                System.out.println("[" + timestamp() + "] [" + getClass().getSimpleName() + " " + getNodeName() + "] RECEIVED SLOWDOWN at min priority, cascading upstream");
                signalUpstreamSlowdown();
            }
        }
    }

    /**
     * Check if we should recover from slowdown mode.
     * After 10 seconds without a slowdown signal, increase priority by 1.
     * Continue recovering every 10 seconds until back to original priority.
     */
    protected synchronized void checkSlowdownRecovery() {
        if (!inSlowdownMode || slowdownPriorityReduction <= 0) {
            return;
        }

        long now = System.currentTimeMillis();
        long timeSinceSlowdown = now - lastSlowdownReceivedTime;

        if (timeSinceSlowdown >= SLOWDOWN_RECOVERY_MS) {
            if (processingThread != null && processingThread.isAlive()) {
                int currentPriority = processingThread.getPriority();
                int maxAllowedPriority = originalPriority;

                if (currentPriority < maxAllowedPriority) {
                    // Recover one priority level
                    int newPriority = currentPriority + 1;
                    System.out.println("[" + timestamp() + "] [" + getClass().getSimpleName() + " " + getNodeName() + "] SLOWDOWN RECOVERY, " +
                        "raising priority: " + currentPriority + " -> " + newPriority);
                    processingThread.setPriority(newPriority);
                    lastRunningPriority = newPriority;
                    slowdownPriorityReduction--;
                    lastSlowdownReceivedTime = now; // Reset timer for next recovery step
                }

                if (slowdownPriorityReduction <= 0) {
                    inSlowdownMode = false;
                    System.out.println("[" + timestamp() + "] [" + getClass().getSimpleName() + " " + getNodeName() + "] EXITED SLOWDOWN MODE");
                }
            }
        }
    }

    public boolean isRunning() {
        return running.get();
    }

    /**
     * Check if the processing thread is alive.
     * Used for counting active threads in the status bar.
     */
    public boolean hasActiveThread() {
        return processingThread != null && processingThread.isAlive();
    }

    public void setOnFrameCallback(java.util.function.Consumer<Mat> callback) {
        this.onFrameCallback = callback;
    }

    protected void notifyFrame(Mat frame) {
        if (onFrameCallback != null && frame != null) {
            onFrameCallback.accept(frame);
        }
    }

    /**
     * Get a timestamp string for logging.
     */
    private static final SimpleDateFormat LOG_TIME_FORMAT = new SimpleDateFormat("HH:mm:ss.SSS");
    public static String timestamp() {
        return LOG_TIME_FORMAT.format(new Date());
    }

    /**
     * Start this node's processing thread.
     * Source nodes generate frames, processing nodes consume from input and produce to output.
     */
    public abstract void startProcessing();

    /**
     * Stop this node's processing thread.
     * Queues are NOT cleared - they retain their data.
     */
    public void stopProcessing() {
        running.set(false);
        if (processingThread != null) {
            processingThread.interrupt();
            try {
                processingThread.join(1000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            processingThread = null;
        }
        // Do NOT clear queues - they keep their data
    }

    // ==================== Serialization Support ====================

    /**
     * Serialize common properties shared by all nodes.
     * Called by PipelineSerializer before serializeProperties().
     *
     * @param json the JSON object to write to
     */
    public void serializeCommon(JsonObject json) {
        json.addProperty("type", getSerializationType());
        json.addProperty("x", x);
        json.addProperty("y", y);
        json.addProperty("threadPriority", threadPriority);
        json.addProperty("workUnitsCompleted", workUnitsCompleted);
        json.addProperty("inputReads1", inputReads1);
        json.addProperty("inputReads2", inputReads2);
    }

    /**
     * Deserialize common properties shared by all nodes.
     * Called by PipelineSerializer before deserializeProperties().
     *
     * @param json the JSON object to read from
     */
    public void deserializeCommon(JsonObject json) {
        if (json.has("x")) x = json.get("x").getAsInt();
        if (json.has("y")) y = json.get("y").getAsInt();
        if (json.has("threadPriority")) {
            threadPriority = json.get("threadPriority").getAsInt();
            originalPriority = threadPriority;
            lastRunningPriority = threadPriority;
        }
        if (json.has("workUnitsCompleted")) {
            workUnitsCompleted = json.get("workUnitsCompleted").getAsLong();
        }
        if (json.has("inputReads1")) {
            inputReads1 = json.get("inputReads1").getAsLong();
        }
        if (json.has("inputReads2")) {
            inputReads2 = json.get("inputReads2").getAsLong();
        }
    }

    /**
     * Default implementation - subclasses override to serialize their specific properties.
     */
    @Override
    public void serializeProperties(JsonObject json) {
        // No additional properties by default
    }

    /**
     * Default implementation - subclasses override to deserialize their specific properties.
     */
    @Override
    public void deserializeProperties(JsonObject json) {
        // No additional properties by default
    }
}
