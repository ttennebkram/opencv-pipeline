package com.ttennebkram.pipeline.processing;

import org.opencv.core.Mat;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * Specialized processor for Container nodes.
 * Handles shuttling frames between the container's external queues
 * and the internal boundary nodes (ContainerInput/ContainerOutput).
 */
public class ContainerProcessor extends ThreadedProcessor {

    private BlockingQueue<Mat> containerInputQueue;  // Queue to ContainerInput boundary node
    private BlockingQueue<Mat> containerOutputQueue; // Queue from ContainerOutput boundary node

    public ContainerProcessor(String name) {
        // Pass a no-op processor - we override the processing loop entirely
        super(name, input -> null);
    }

    /**
     * Set the container input queue.
     * Frames from upstream are forwarded to this queue,
     * which feeds the ContainerInput boundary node.
     */
    public void setContainerInputQueue(BlockingQueue<Mat> queue) {
        this.containerInputQueue = queue;
    }

    /**
     * Set the container output queue.
     * This queue receives frames from the ContainerOutput boundary node,
     * which are then forwarded to downstream.
     */
    public void setContainerOutputQueue(BlockingQueue<Mat> queue) {
        this.containerOutputQueue = queue;
    }

    /**
     * Override the processing loop to handle container-specific behavior.
     * Shuttles frames in both directions:
     * 1. upstream inputQueue -> containerInputQueue (to ContainerInput)
     * 2. containerOutputQueue (from ContainerOutput) -> downstream outputQueue
     */
    @Override
    protected void processingLoop() {
        System.out.println("[ContainerProcessor] " + getName() + " starting processingLoop");
        System.out.println("[ContainerProcessor] " + getName() + " inputQueue=" + (getInputQueue() != null ? "set" : "NULL") +
                           ", containerInputQueue=" + (containerInputQueue != null ? "set" : "NULL") +
                           ", containerOutputQueue=" + (containerOutputQueue != null ? "set" : "NULL") +
                           ", outputQueue=" + (getOutputQueue() != null ? "set" : "NULL"));

        long lastDebugTime = System.currentTimeMillis();
        long inputCount = 0;
        long outputCount = 0;

        while (isRunning()) {
            try {
                boolean didWork = false;

                // Check for slowdown recovery
                checkSlowdownRecovery();

                // === INPUT SIDE: Forward frames from upstream to ContainerInput ===
                BlockingQueue<Mat> inputQueue = getInputQueue();
                if (inputQueue != null) {
                    Mat input = inputQueue.poll(10, TimeUnit.MILLISECONDS);
                    if (input != null) {
                        inputCount++;
                        incrementInputReads1();

                        // Bypass mode: if disabled OR if no internal pipeline loaded
                        boolean bypass = !isEnabled() || containerInputQueue == null;
                        if (bypass) {
                            // Skip internal pipeline, put directly on output queue
                            BlockingQueue<Mat> outputQueue = getOutputQueue();
                            if (outputQueue != null) {
                                outputQueue.put(input.clone());
                                incrementOutputWrites1();
                            }
                            notifyCallback(input.clone());
                            incrementWorkUnitsCompleted();
                            input.release();
                        } else {
                            // Normal mode: forward to ContainerInput
                            containerInputQueue.put(input);
                        }
                        didWork = true;
                    }
                }

                // === OUTPUT SIDE: Forward frames from ContainerOutput to downstream ===
                if (isEnabled() && containerOutputQueue != null) {
                    Mat output = containerOutputQueue.poll(10, TimeUnit.MILLISECONDS);
                    if (output != null) {
                        outputCount++;
                        // Work is completed when we dequeue from boundary output
                        incrementWorkUnitsCompleted();

                        // Notify callback with a clone (for thumbnail update)
                        notifyCallback(output.clone());

                        // Forward to downstream output queue if connected
                        BlockingQueue<Mat> outputQueue = getOutputQueue();
                        if (outputQueue != null) {
                            outputQueue.put(output.clone());
                            incrementOutputWrites1();
                        }

                        // Check backpressure after producing output
                        checkBackpressure();

                        output.release();
                        didWork = true;
                    }
                }

                // If no work done, sleep briefly to avoid busy-spinning
                if (!didWork) {
                    Thread.sleep(5);
                }

                // Periodic debug output
                long now = System.currentTimeMillis();
                if (now - lastDebugTime > 2000) {
                    System.out.println("[ContainerProcessor] " + getName() + " stats: in=" + inputCount + " out=" + outputCount +
                                       " inputQueueSize=" + (inputQueue != null ? inputQueue.size() : -1) +
                                       " containerInQueueSize=" + (containerInputQueue != null ? containerInputQueue.size() : -1) +
                                       " containerOutQueueSize=" + (containerOutputQueue != null ? containerOutputQueue.size() : -1));
                    lastDebugTime = now;
                }

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
        System.out.println("[ContainerProcessor] " + getName() + " exiting processingLoop");
    }
}
