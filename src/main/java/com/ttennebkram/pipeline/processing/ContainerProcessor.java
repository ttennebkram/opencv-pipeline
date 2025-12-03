package com.ttennebkram.pipeline.processing;

import com.ttennebkram.pipeline.fx.FXNode;
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

    // Reference to internal ContainerOutput processor for backpressure signaling
    private ThreadedProcessor containerOutputProcessor;

    public ContainerProcessor(String name) {
        // Pass a no-op processor - we override the processing loop entirely
        super(name, input -> null);
    }

    /**
     * Set the internal ContainerOutput processor reference.
     * Used to forward slowdown signals into the container's internal pipeline.
     */
    public void setContainerOutputProcessor(ThreadedProcessor outputProcessor) {
        this.containerOutputProcessor = outputProcessor;
    }

    /**
     * Override receiveSlowdownSignal to forward to internal ContainerOutput when at min priority.
     * This allows backpressure from downstream to propagate into the container only after
     * this container has exhausted its own priority reduction.
     */
    @Override
    public synchronized void receiveSlowdownSignal() {
        // Check if we're already at minimum priority before handling
        int currentPriority = getThreadPriority();
        boolean wasAtMinPriority = (currentPriority <= Thread.MIN_PRIORITY);

        // Handle our own slowdown first
        super.receiveSlowdownSignal();

        // Only forward to internal ContainerOutput if we were already at min priority
        // (meaning we can't slow down any further ourselves)
        if (wasAtMinPriority && containerOutputProcessor != null) {
            containerOutputProcessor.receiveSlowdownSignal();
        }
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
                        incrementInputReads1();

                        // Bypass mode: if disabled OR if no internal pipeline loaded
                        boolean bypass = !isEnabled() || containerInputQueue == null;
                        if (bypass) {
                            // Skip internal pipeline, put directly on output queue
                            BlockingQueue<Mat> outputQueue = getOutputQueue();
                            if (outputQueue != null) {
                                outputQueue.put(input.clone());
                                incrementOutputWrites1();
                                // Check backpressure in bypass mode too
                                checkBackpressure();
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

                // Update FXNode stats periodically (priority display) even when no work done
                FXNode node = getFXNode();
                if (node != null) {
                    node.threadPriority = getThreadPriority();
                    node.workUnitsCompleted = getWorkUnitsCompleted();
                }

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
    }
}
