package com.example.pipeline.model;

import com.example.pipeline.nodes.PipelineNode;
import org.eclipse.swt.graphics.Point;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import org.opencv.core.Mat;

/**
 * Reverse dangling connection (target fixed, source free).
 * Maintains a queue for potential frames if it were connected.
 */
public class ReverseDanglingConnection {
    public PipelineNode target;
    public Point freeEnd;
    private BlockingQueue<Mat> queue;
    private int queueCapacity = 0; // 0 = unlimited, >0 = fixed capacity
    private int lastQueueSize = 0; // Track queue size for persistence

    public ReverseDanglingConnection(PipelineNode target, Point freeEnd) {
        this.target = target;
        this.freeEnd = new Point(freeEnd.x, freeEnd.y);
        this.queue = null; // Queue is created when pipeline starts
    }

    public int getConfiguredCapacity() {
        return queueCapacity;
    }

    public void setConfiguredCapacity(int capacity) {
        this.queueCapacity = capacity;
    }

    /**
     * Create and initialize the queue for this reverse dangling connection.
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
     * Get the queue for this reverse dangling connection.
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
     * Check if the queue is active (pipeline running).
     */
    public boolean isActive() {
        return queue != null;
    }

    /**
     * Activate this reverse dangling connection: create queue (if needed) and wire it to target node.
     */
    public void activate() {
        // Only create queue if it doesn't exist yet
        if (queue == null) {
            createQueue();
        }
        // Wire the queue: this queue -> target's input (no source)
        target.setInputQueue(queue);
    }

    /**
     * Deactivate this reverse dangling connection: do nothing, keep queue intact.
     */
    public void deactivate() {
        // Do nothing - queue stays connected and retains all data
    }
}
