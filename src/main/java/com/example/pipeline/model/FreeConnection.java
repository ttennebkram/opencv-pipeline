package com.example.pipeline.model;

import org.eclipse.swt.graphics.Point;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import org.opencv.core.Mat;

/**
 * Free connection (both ends free, no nodes attached).
 * Maintains a queue that retains data from when it was connected.
 */
public class FreeConnection {
    public Point startEnd;  // Non-arrow end
    public Point arrowEnd;  // Arrow end
    private BlockingQueue<Mat> queue;
    private int queueCapacity = 0; // 0 = unlimited, >0 = fixed capacity
    private int lastQueueSize = 0; // Track queue size for persistence

    public FreeConnection(Point startEnd, Point arrowEnd) {
        this.startEnd = new Point(startEnd.x, startEnd.y);
        this.arrowEnd = new Point(arrowEnd.x, arrowEnd.y);
        this.queue = null; // Queue is created when pipeline starts
    }

    public int getConfiguredCapacity() {
        return queueCapacity;
    }

    public void setConfiguredCapacity(int capacity) {
        this.queueCapacity = capacity;
    }

    /**
     * Create and initialize the queue for this free connection.
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
     * Get the queue for this free connection.
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
     * Activate this free connection: create queue (if needed, not connected to any node).
     */
    public void activate() {
        // Only create queue if it doesn't exist yet
        if (queue == null) {
            createQueue();
        }
        // No nodes to wire to - this queue just exists
    }

    /**
     * Deactivate this free connection: do nothing, keep queue intact.
     */
    public void deactivate() {
        // Do nothing - queue retains all data
    }
}
