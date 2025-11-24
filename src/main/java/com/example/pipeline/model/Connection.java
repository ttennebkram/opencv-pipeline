package com.example.pipeline.model;

import com.example.pipeline.nodes.PipelineNode;
import com.example.pipeline.nodes.AddClampNode;
import com.example.pipeline.nodes.SubtractClampNode;
import com.example.pipeline.nodes.BitwiseAndNode;
import com.example.pipeline.nodes.BitwiseOrNode;
import com.example.pipeline.nodes.BitwiseXorNode;
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
    private static final int DEFAULT_QUEUE_CAPACITY = 3;

    public Connection(PipelineNode source, PipelineNode target) {
        this(source, target, 1);
    }

    public Connection(PipelineNode source, PipelineNode target, int inputIndex) {
        this.source = source;
        this.target = target;
        this.inputIndex = inputIndex;
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

    /**
     * Activate this connection: create queue and wire it between source and target nodes.
     */
    public void activate() {
        createQueue();
        // Wire the queue: source's output -> this queue -> target's input
        source.setOutputQueue(queue);

        // For dual-input nodes, use the appropriate input queue
        if (inputIndex == 2) {
            if (target instanceof AddClampNode) {
                ((AddClampNode) target).setInputQueue2(queue);
            } else if (target instanceof SubtractClampNode) {
                ((SubtractClampNode) target).setInputQueue2(queue);
            } else if (target instanceof BitwiseAndNode) {
                ((BitwiseAndNode) target).setInputQueue2(queue);
            } else if (target instanceof BitwiseOrNode) {
                ((BitwiseOrNode) target).setInputQueue2(queue);
            } else if (target instanceof BitwiseXorNode) {
                ((BitwiseXorNode) target).setInputQueue2(queue);
            } else {
                target.setInputQueue(queue);
            }
        } else {
            target.setInputQueue(queue);
        }
    }

    /**
     * Deactivate this connection: clear queue and disconnect from nodes.
     */
    public void deactivate() {
        clearQueue();
        source.setOutputQueue(null);

        // For dual-input nodes, clear the appropriate input queue
        if (inputIndex == 2) {
            if (target instanceof AddClampNode) {
                ((AddClampNode) target).setInputQueue2(null);
            } else if (target instanceof SubtractClampNode) {
                ((SubtractClampNode) target).setInputQueue2(null);
            } else if (target instanceof BitwiseAndNode) {
                ((BitwiseAndNode) target).setInputQueue2(null);
            } else if (target instanceof BitwiseOrNode) {
                ((BitwiseOrNode) target).setInputQueue2(null);
            } else if (target instanceof BitwiseXorNode) {
                ((BitwiseXorNode) target).setInputQueue2(null);
            } else {
                target.setInputQueue(null);
            }
        } else {
            target.setInputQueue(null);
        }
    }
}
