package com.ttennebkram.pipeline.model;

import com.ttennebkram.pipeline.nodes.PipelineNode;
import com.ttennebkram.pipeline.nodes.DualInputNode;
import java.util.concurrent.BlockingQueue;
import org.opencv.core.Mat;

/**
 * Connection between two nodes with an associated queue for pipeline execution.
 */
public class Connection {
    public PipelineNode source;
    public PipelineNode target;
    public int inputIndex; // 1 for primary input, 2 for secondary input (dual-input nodes)
    public int outputIndex; // 0 for primary output, 1+ for additional outputs (multi-output nodes)
    private CountingBlockingQueue<Mat> queue;
    private int queueCapacity = 0; // 0 = unlimited, >0 = fixed capacity
    private int lastQueueSize = 0; // Track queue size for persistence

    public Connection(PipelineNode source, PipelineNode target) {
        this(source, target, 1, 0);
    }

    public Connection(PipelineNode source, PipelineNode target, int inputIndex) {
        this(source, target, inputIndex, 0);
    }

    public Connection(PipelineNode source, PipelineNode target, int inputIndex, int outputIndex) {
        this.source = source;
        this.target = target;
        this.inputIndex = inputIndex;
        this.outputIndex = outputIndex;
        this.queue = null; // Queue is created when pipeline starts
    }

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

    // Store for deferred application when queue doesn't exist yet
    private long pendingTotalFrames = 0;

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
     */
    public void activate() {
        // Only create queue if it doesn't exist yet
        if (queue == null) {
            createQueue();
        }

        // Wire the queue: source's output (by index) -> this queue -> target's input
        if (source.hasMultipleOutputs()) {
            source.setOutputQueue(outputIndex, queue);
        } else {
            source.setOutputQueue(queue);
        }

        // For dual-input nodes, use the appropriate input queue
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

    /**
     * Deactivate this connection: do nothing, keep queue and all connections intact.
     */
    public void deactivate() {
        // Do nothing - queue stays connected and retains all data
    }
}
