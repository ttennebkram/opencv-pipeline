package com.ttennebkram.pipeline.processing;

import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.util.MatTracker;
import org.opencv.core.Mat;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * Processor for dual-input nodes (AddClamp, SubtractClamp, BitwiseAnd, etc.).
 * Reads from both inputQueue and inputQueue2, processes with a DualImageProcessor,
 * and outputs to outputQueue.
 */
public class DualInputProcessor extends ThreadedProcessor {

    private final DualImageProcessor dualProcessor;

    // Cached frames from each input
    private Mat lastInput1 = null;
    private Mat lastInput2 = null;

    public DualInputProcessor(String name, DualImageProcessor processor) {
        // Pass null to parent - we override processingLoop entirely
        super(name, null);
        this.dualProcessor = processor;
        setIsDualInput(true);
    }

    @Override
    protected void processingLoop() {
        BlockingQueue<Mat> inputQueue = getInputQueue();
        BlockingQueue<Mat> inputQueue2 = getInputQueue2();

        while (isRunning()) {
            try {
                // Check for slowdown recovery
                checkSlowdownRecovery();

                // Re-fetch queues in case they were set after start
                inputQueue = getInputQueue();
                inputQueue2 = getInputQueue2();

                // Check queuesInSync mode from FXNode
                FXNode node = getFXNode();
                boolean syncMode = node != null && node.queuesInSync;

                boolean gotInput1 = false;
                boolean gotInput2 = false;

                if (syncMode) {
                    // Synchronized mode: wait until BOTH queues have data, then take from both
                    // This ensures we always process matching pairs

                    // Wait until both queues have at least one item
                    while (isRunning()) {
                        int q1Size = inputQueue != null ? inputQueue.size() : -1;
                        int q2Size = inputQueue2 != null ? inputQueue2.size() : -1;

                        if (q1Size > 0 && q2Size > 0) {
                            break; // Both have data, proceed
                        }

                        // Sleep briefly to avoid busy-waiting
                        Thread.sleep(5);
                    }

                    if (!isRunning()) break;

                    // Now take one from each (should not block since we checked size)
                    Mat input1 = inputQueue != null ? inputQueue.poll() : null;
                    Mat input2 = inputQueue2 != null ? inputQueue2.poll() : null;

                    // Track reads
                    if (input1 != null) {
                        if (lastInput1 != null) MatTracker.release(lastInput1);
                        lastInput1 = input1;
                        MatTracker.track(lastInput1);
                        gotInput1 = true;
                        incrementInputReads1();
                    }
                    if (input2 != null) {
                        if (lastInput2 != null) MatTracker.release(lastInput2);
                        lastInput2 = input2;
                        MatTracker.track(lastInput2);
                        gotInput2 = true;
                        incrementInputReads2();
                    }
                } else {
                    // Non-sync mode: poll both queues, process when either has new data
                    if (inputQueue != null) {
                        Mat newInput1 = inputQueue.poll(50, TimeUnit.MILLISECONDS);
                        if (newInput1 != null) {
                            if (lastInput1 != null) MatTracker.release(lastInput1);
                            lastInput1 = newInput1;
                            MatTracker.track(lastInput1);
                            gotInput1 = true;
                            incrementInputReads1();
                        }
                    }

                    if (inputQueue2 != null) {
                        Mat newInput2 = inputQueue2.poll(50, TimeUnit.MILLISECONDS);
                        if (newInput2 != null) {
                            if (lastInput2 != null) MatTracker.release(lastInput2);
                            lastInput2 = newInput2;
                            MatTracker.track(lastInput2);
                            gotInput2 = true;
                            incrementInputReads2();
                        }
                    }

                    // Skip processing if no new data arrived
                    if (!gotInput1 && !gotInput2) {
                        continue;
                    }
                }

                // Process if we have at least one valid input
                if (lastInput1 != null || lastInput2 != null) {
                    Mat output;

                    if (!isEnabled()) {
                        // Bypass mode: pass through input1 unchanged
                        output = lastInput1 != null ? lastInput1.clone() : null;
                        if (output != null) MatTracker.track(output);
                    } else {
                        // Process the frames
                        output = dualProcessor.process(lastInput1, lastInput2);
                        if (output != null) MatTracker.track(output);
                    }

                    incrementWorkUnitsCompleted();

                    if (output != null) {
                        // Notify callback with a clone
                        Mat callbackCopy = output.clone();
                        MatTracker.track(callbackCopy);
                        notifyCallback(callbackCopy);

                        // Put on output queue
                        BlockingQueue<Mat> outputQueue = getOutputQueue();
                        if (outputQueue != null) {
                            Mat queueCopy = output.clone();
                            MatTracker.track(queueCopy);
                            outputQueue.put(queueCopy);
                            incrementOutputWrites1();
                        }

                        // Check backpressure AFTER putting on output queue
                        checkBackpressure();

                        MatTracker.release(output);
                    }
                }

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }

        // Cleanup cached frames
        if (lastInput1 != null) {
            MatTracker.release(lastInput1);
            lastInput1 = null;
        }
        if (lastInput2 != null) {
            MatTracker.release(lastInput2);
            lastInput2 = null;
        }
    }

}
