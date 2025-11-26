package com.ttennebkram.pipeline.model;

import com.ttennebkram.pipeline.nodes.PipelineNode;
import com.ttennebkram.pipeline.nodes.DualInputNode;
import java.util.concurrent.BlockingQueue;
import org.eclipse.swt.graphics.Point;
import org.opencv.core.Mat;

/**
 * Connection between two nodes (or with free endpoints) with an associated queue for pipeline execution.
 *
 * A Connection can be in one of four states:
 * - Complete: both source and target are connected to nodes
 * - Source-dangling: source is connected, target is a free point
 * - Target-dangling: target is connected, source is a free point
 * - Free: both ends are free points (not connected to any node)
 */
public class Connection {
    // Node references (null if that end is disconnected)
    public PipelineNode source;
    public PipelineNode target;

    // Input/output indices for when connected to nodes
    public int inputIndex; // 1 for primary input, 2 for secondary input (dual-input nodes)
    public int outputIndex; // 0 for primary output, 1+ for additional outputs (multi-output nodes)

    // Free endpoint positions (used when corresponding node is null)
    private Point freeSourcePoint;
    private Point freeTargetPoint;

    // Queue for pipeline execution
    private CountingBlockingQueue<Mat> queue;
    private int queueCapacity = 0; // 0 = unlimited, >0 = fixed capacity
    private int lastQueueSize = 0; // Track queue size for persistence
    private long pendingTotalFrames = 0; // Store for deferred application when queue doesn't exist yet

    /**
     * Create a complete connection between two nodes.
     */
    public Connection(PipelineNode source, PipelineNode target) {
        this(source, target, 1, 0);
    }

    /**
     * Create a complete connection with specified input index.
     */
    public Connection(PipelineNode source, PipelineNode target, int inputIndex) {
        this(source, target, inputIndex, 0);
    }

    /**
     * Create a complete connection with specified input and output indices.
     */
    public Connection(PipelineNode source, PipelineNode target, int inputIndex, int outputIndex) {
        this.source = source;
        this.target = target;
        this.inputIndex = inputIndex;
        this.outputIndex = outputIndex;
        this.queue = null; // Queue is created when pipeline starts
    }

    /**
     * Create a source-dangling connection (source connected, target free).
     */
    public static Connection createSourceDangling(PipelineNode source, int outputIndex, Point freeEnd) {
        Connection conn = new Connection(source, null, 1, outputIndex);
        conn.freeTargetPoint = new Point(freeEnd.x, freeEnd.y);
        return conn;
    }

    /**
     * Create a target-dangling connection (target connected, source free).
     */
    public static Connection createTargetDangling(PipelineNode target, int inputIndex, Point freeEnd) {
        Connection conn = new Connection(null, target, inputIndex, 0);
        conn.freeSourcePoint = new Point(freeEnd.x, freeEnd.y);
        return conn;
    }

    /**
     * Create a free connection (both ends free).
     */
    public static Connection createFree(Point sourcePoint, Point targetPoint) {
        Connection conn = new Connection(null, null, 1, 0);
        conn.freeSourcePoint = new Point(sourcePoint.x, sourcePoint.y);
        conn.freeTargetPoint = new Point(targetPoint.x, targetPoint.y);
        return conn;
    }

    // ========== Connection state queries ==========

    /**
     * Check if this is a complete connection (both ends connected to nodes).
     */
    public boolean isComplete() {
        return source != null && target != null;
    }

    /**
     * Check if this connection has at least one disconnected end.
     */
    public boolean isDangling() {
        return source == null || target == null;
    }

    /**
     * Check if this is a source-dangling connection (source connected, target free).
     */
    public boolean isSourceDangling() {
        return source != null && target == null;
    }

    /**
     * Check if this is a target-dangling connection (target connected, source free).
     */
    public boolean isTargetDangling() {
        return source == null && target != null;
    }

    /**
     * Check if this is a free connection (both ends free).
     */
    public boolean isFree() {
        return source == null && target == null;
    }

    // ========== Endpoint access ==========

    /**
     * Get the start point of this connection (source output or free point).
     */
    public Point getStartPoint() {
        if (source != null) {
            return source.getOutputPoint(outputIndex);
        }
        return freeSourcePoint;
    }

    /**
     * Get the end point of this connection (target input or free point).
     */
    public Point getEndPoint() {
        if (target != null) {
            if (inputIndex == 2 && target.hasDualInput()) {
                return target.getInputPoint2();
            }
            return target.getInputPoint();
        }
        return freeTargetPoint;
    }

    /**
     * Get the free source point (null if source is connected to a node).
     */
    public Point getFreeSourcePoint() {
        return freeSourcePoint;
    }

    /**
     * Set the free source point.
     */
    public void setFreeSourcePoint(Point point) {
        this.freeSourcePoint = point != null ? new Point(point.x, point.y) : null;
    }

    /**
     * Set the free source point by coordinates.
     */
    public void setFreeSourcePoint(int x, int y) {
        this.freeSourcePoint = new Point(x, y);
    }

    /**
     * Get the free target point (null if target is connected to a node).
     */
    public Point getFreeTargetPoint() {
        return freeTargetPoint;
    }

    /**
     * Set the free target point.
     */
    public void setFreeTargetPoint(Point point) {
        this.freeTargetPoint = point != null ? new Point(point.x, point.y) : null;
    }

    /**
     * Set the free target point by coordinates.
     */
    public void setFreeTargetPoint(int x, int y) {
        this.freeTargetPoint = new Point(x, y);
    }

    // ========== Connection modification ==========

    /**
     * Connect the source end to a node.
     */
    public void connectSource(PipelineNode node, int outputIdx) {
        this.source = node;
        this.outputIndex = outputIdx;
        this.freeSourcePoint = null;
    }

    /**
     * Disconnect the source end, saving its current position.
     */
    public void disconnectSource() {
        if (source != null) {
            this.freeSourcePoint = source.getOutputPoint(outputIndex);
            this.source = null;
        }
    }

    /**
     * Connect the target end to a node.
     */
    public void connectTarget(PipelineNode node, int inputIdx) {
        this.target = node;
        this.inputIndex = inputIdx;
        this.freeTargetPoint = null;
    }

    /**
     * Disconnect the target end, saving its current position.
     */
    public void disconnectTarget() {
        if (target != null) {
            if (inputIndex == 2 && target.hasDualInput()) {
                this.freeTargetPoint = target.getInputPoint2();
            } else {
                this.freeTargetPoint = target.getInputPoint();
            }
            this.target = null;
        }
    }

    // ========== Queue management ==========

    public int getConfiguredCapacity() {
        return queueCapacity;
    }

    public void setConfiguredCapacity(int capacity) {
        this.queueCapacity = capacity;
    }

    /**
     * Create and initialize the queue for this connection.
     * Uses the configured capacity (0 = unlimited).
     */
    public void createQueue() {
        if (queueCapacity <= 0) {
            this.queue = new CountingBlockingQueue<>();
        } else {
            this.queue = new CountingBlockingQueue<>(queueCapacity);
        }
    }

    /**
     * Create and initialize the queue with a specific capacity.
     */
    public void createQueue(int capacity) {
        if (capacity <= 0) {
            this.queue = new CountingBlockingQueue<>();
        } else {
            this.queue = new CountingBlockingQueue<>(capacity);
        }
    }

    /**
     * Clear the queue when pipeline stops.
     * Captures the queue size before clearing so it persists.
     */
    public void clearQueue() {
        if (queue != null) {
            lastQueueSize = queue.size(); // Save size before clearing
            queue.clear();
            queue = null;
        }
    }

    /**
     * Get the queue for this connection.
     */
    public BlockingQueue<Mat> getQueue() {
        return queue;
    }

    /**
     * Get the current number of items in the queue.
     * Returns last known size if queue is not active.
     */
    public int getQueueSize() {
        if (queue != null) {
            lastQueueSize = queue.size();
            return lastQueueSize;
        }
        return lastQueueSize;
    }

    /**
     * Set the last known queue size (used when loading from JSON).
     */
    public void setLastQueueSize(int size) {
        this.lastQueueSize = size;
    }

    /**
     * Get total frames sent through this connection.
     */
    public long getTotalFramesSent() {
        return queue != null ? queue.getTotalAdded() : pendingTotalFrames;
    }

    /**
     * Set total frames sent (used when loading from JSON, applied when queue is created).
     */
    public void setTotalFramesSent(long count) {
        if (queue != null) {
            queue.setTotalAdded(count);
        }
    }

    /**
     * Set pending total frames (for loading before queue exists).
     */
    public void setPendingTotalFrames(long count) {
        this.pendingTotalFrames = count;
    }

    /**
     * Get pending total frames.
     */
    public long getPendingTotalFrames() {
        return pendingTotalFrames;
    }

    /**
     * Get the queue capacity.
     */
    public int getQueueCapacity() {
        return queue != null ? queue.remainingCapacity() + queue.size() : Integer.MAX_VALUE;
    }

    /**
     * Check if the queue is active (pipeline running).
     */
    public boolean isActive() {
        return queue != null;
    }

    /**
     * Activate this connection: create queue (if needed) and wire it between source and target nodes.
     * For dangling/free connections, only wires to the connected end(s).
     */
    public void activate() {
        // Only create queue if it doesn't exist yet
        if (queue == null) {
            createQueue();
        }

        // Wire the source end (if connected)
        if (source != null) {
            if (source.hasMultipleOutputs()) {
                source.setOutputQueue(outputIndex, queue);
            } else {
                source.setOutputQueue(queue);
            }
        }

        // Wire the target end (if connected)
        if (target != null) {
            if (inputIndex == 2) {
                if (target instanceof DualInputNode) {
                    ((DualInputNode) target).setInputQueue2(queue);
                } else {
                    target.setInputQueue(queue);
                }
            } else {
                target.setInputQueue(queue);
            }
        }
    }

    /**
     * Deactivate this connection: do nothing, keep queue and all connections intact.
     */
    public void deactivate() {
        // Do nothing - queue stays connected and retains all data
    }
}
