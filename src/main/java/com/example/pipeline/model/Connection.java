package com.example.pipeline.model;

import com.example.pipeline.nodes.PipelineNode;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import org.opencv.core.Mat;

/**
 * Connection between two nodes with an associated queue for pipeline execution.
 */
public class Connection {
    public PipelineNode source;
    public PipelineNode target;
    private BlockingQueue<Mat> queue;
    private static final int DEFAULT_QUEUE_CAPACITY = 3;

    public Connection(PipelineNode source, PipelineNode target) {
        this.source = source;
        this.target = target;
        this.queue = null; // Queue is created when pipeline starts
    }

    /**
     * Create and initialize the queue for this connection.
     */
    public void createQueue() {
        this.queue = new LinkedBlockingQueue<>(DEFAULT_QUEUE_CAPACITY);
    }

    /**
     * Create and initialize the queue with a specific capacity.
     */
    public void createQueue(int capacity) {
        this.queue = new LinkedBlockingQueue<>(capacity);
    }

    /**
     * Clear the queue when pipeline stops.
     */
    public void clearQueue() {
        if (queue != null) {
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
     */
    public int getQueueSize() {
        return queue != null ? queue.size() : 0;
    }

    /**
     * Get the queue capacity.
     */
    public int getQueueCapacity() {
        return queue != null ? queue.remainingCapacity() + queue.size() : DEFAULT_QUEUE_CAPACITY;
    }

    /**
     * Check if the queue is active (pipeline running).
     */
    public boolean isActive() {
        return queue != null;
    }
}
