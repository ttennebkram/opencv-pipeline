package com.ttennebkram.pipeline.model;

import com.ttennebkram.pipeline.nodes.PipelineNode;
import com.ttennebkram.pipeline.nodes.DualInputNode;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import org.opencv.core.Mat;

/**
 * Connection between two nodes with an associated queue for pipeline execution.
 */
public class Connection {
    public PipelineNode source;
    public PipelineNode target;
    public int inputIndex; // 1 for primary input, 2 for secondary input (dual-input nodes)
    private BlockingQueue<Mat> queue;
    private int queueCapacity = 0; // 0 = unlimited, >0 = fixed capacity
    private int lastQueueSize = 0; // Track queue size for persistence

    public Connection(PipelineNode source, PipelineNode target) {
        this(source, target, 1);
    }

    public Connection(PipelineNode source, PipelineNode target, int inputIndex) {
        this.source = source;
        this.target = target;
        this.inputIndex = inputIndex;
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
            this.queue = new LinkedBlockingQueue<>();
        } else {
            this.queue = new LinkedBlockingQueue<>(queueCapacity);
        }
    }

    /**
     * Create and initialize the queue with a specific capacity.
     */
    public void createQueue(int capacity) {
        if (capacity <= 0) {
            this.queue = new LinkedBlockingQueue<>();
        } else {
            this.queue = new LinkedBlockingQueue<>(capacity);
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
        // Wire the queue: source's output -> this queue -> target's input
        source.setOutputQueue(queue);

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
