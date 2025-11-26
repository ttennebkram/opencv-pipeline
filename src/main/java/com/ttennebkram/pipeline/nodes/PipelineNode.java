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

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Base class for all pipeline nodes.
 * Implements NodeSerializable to handle common property serialization.
 */
public abstract class PipelineNode implements NodeSerializable {
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
    protected long frameDelayMs = 0; // Frame rate throttling (0 = no delay)
    protected int threadPriority = Thread.NORM_PRIORITY; // Thread priority (1-10, default 5)
    protected int originalPriority = Thread.NORM_PRIORITY; // Original priority before backpressure adjustments
    protected int lastRunningPriority = Thread.NORM_PRIORITY; // Last actual running priority (persists after stop)

    // Backpressure management
    protected static final int QUEUE_HIGH_WATERMARK = 10; // Reduce upstream priority when queue exceeds this
    protected static final int QUEUE_LOW_WATERMARK = 3;   // Restore upstream priority when queue drops below this
    protected static final long PRIORITY_ADJUSTMENT_TIMEOUT_MS = 10000; // 10 seconds between priority adjustments
    protected long lastPriorityAdjustmentTime = 0; // Timestamp of last priority change
    protected PipelineNode inputNode = null; // Reference to upstream node for backpressure signaling
    protected PipelineNode inputNode2 = null; // Reference to second upstream node for dual-input nodes

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
            gc.setForeground(new Color(0, 120, 215));  // Blue selection color
            gc.setLineWidth(3);
            gc.drawRoundRectangle(x - 3, y - 3, width + 6, height + 6, 13, 13);
        }
    }

    public void setOutputMat(Mat mat) {
        // Release old outputMat if it exists
        if (this.outputMat != null && !this.outputMat.empty()) {
            this.outputMat.release();
        }
        this.outputMat = mat;
        updateThumbnail();
    }

    public Mat getOutputMat() {
        return outputMat;
    }

    protected void updateThumbnail() {
        if (outputMat == null || outputMat.empty()) {
            return;
        }

        // Dispose old thumbnail
        if (thumbnail != null && !thumbnail.isDisposed()) {
            thumbnail.dispose();
        }

        // Create thumbnail
        Mat resized = new Mat();
        double scale = Math.min((double) PROCESSING_NODE_THUMB_WIDTH / outputMat.width(),
                                (double) PROCESSING_NODE_THUMB_HEIGHT / outputMat.height());
        Imgproc.resize(outputMat, resized,
            new Size(outputMat.width() * scale, outputMat.height() * scale));

        // Convert to SWT Image
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

        // Create ImageData with direct data copy (much faster than setPixel loop)
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
                // Direct copy - data is already RGB from cvtColor
                imageData.data[dstIdx] = data[srcIdx];         // R
                imageData.data[dstIdx + 1] = data[srcIdx + 1]; // G
                imageData.data[dstIdx + 2] = data[srcIdx + 2]; // B
            }
        }

        thumbnail = new Image(display, imageData);
    }

    protected void drawThumbnail(GC gc, int thumbX, int thumbY) {
        if (thumbnail != null && !thumbnail.isDisposed()) {
            gc.drawImage(thumbnail, thumbX, thumbY);
        }
    }

    public void disposeThumbnail() {
        if (thumbnail != null && !thumbnail.isDisposed()) {
            thumbnail.dispose();
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
        return "Priority: " + getThreadPriority() + " | Work: " + formatNumber(workUnitsCompleted);
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
     */
    protected void checkBackpressure() {
        if (outputQueue == null) {
            return;
        }

        int queueSize = outputQueue.size();
        int targetPriority;

        if (processingThread != null && processingThread.isAlive()) {
            int currentPriority = processingThread.getPriority();

            // Progressive backpressure/boost based on queue size
            if (queueSize == 0) {
                // Queue empty: increase priority by 1 (up to max of 5)
                targetPriority = Math.min(Thread.NORM_PRIORITY, currentPriority + 1);
            } else if (queueSize < 5) {
                // Queue has 1-4 items: increase priority by 1 (up to max of 5)
                targetPriority = Math.min(Thread.NORM_PRIORITY, currentPriority + 1);
            } else {
                // Queue >= 5: apply backpressure, reduce by 1 for every 5 items
                int reductionAmount = queueSize / 5;
                targetPriority = Math.max(Thread.MIN_PRIORITY, originalPriority - reductionAmount);
            }

            if (currentPriority != targetPriority) {
                // Check if enough time has passed since last adjustment
                long currentTime = System.currentTimeMillis();
                long timeSinceLastAdjustment = currentTime - lastPriorityAdjustmentTime;

                if (timeSinceLastAdjustment >= PRIORITY_ADJUSTMENT_TIMEOUT_MS) {
                    processingThread.setPriority(targetPriority);
                    lastPriorityAdjustmentTime = currentTime;
                    lastRunningPriority = targetPriority;
                }
            } else {
                // Always update cached priority to reflect current state
                lastRunningPriority = targetPriority;
            }
        }
    }

    public boolean isRunning() {
        return running.get();
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
