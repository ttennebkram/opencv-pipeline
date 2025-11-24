package com.example.pipeline.nodes;

import org.eclipse.swt.widgets.Display;
import org.eclipse.swt.widgets.Shell;
import org.opencv.core.Mat;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * Base class for nodes that accept two inputs.
 * Handles polling from both input queues and synchronization logic.
 */
public abstract class DualInputNode extends ProcessingNode {
    // Second input queue
    protected BlockingQueue<Mat> inputQueue2;

    // Cached frames from each input
    protected Mat lastInput1 = null;
    protected Mat lastInput2 = null;

    // Whether to only process when both queues receive new frames
    protected boolean queuesInSync = false;

    public DualInputNode(Display display, Shell shell, String name, int x, int y) {
        super(display, shell, name, x, y);
    }

    // Dual input queue management
    public void setInputQueue2(BlockingQueue<Mat> queue) {
        this.inputQueue2 = queue;
    }

    public BlockingQueue<Mat> getInputQueue2() {
        return inputQueue2;
    }

    public boolean isQueuesInSync() {
        return queuesInSync;
    }

    public void setQueuesInSync(boolean sync) {
        this.queuesInSync = sync;
    }

    /**
     * Abstract method for processing two inputs.
     * Subclasses implement their specific dual-input operation.
     */
    public abstract Mat processDual(Mat input1, Mat input2);

    @Override
    public void startProcessing() {
        if (running.get()) {
            return;
        }

        running.set(true);
        workUnitsCompleted = 0; // Reset counter on start

        processingThread = new Thread(() -> {
            while (running.get()) {
                try {
                    boolean gotInput1 = false;
                    boolean gotInput2 = false;

                    // Poll both queues for new frames (non-blocking)
                    if (inputQueue != null) {
                        Mat newInput1 = inputQueue.poll(10, TimeUnit.MILLISECONDS);
                        if (newInput1 != null) {
                            if (lastInput1 != null) lastInput1.release();
                            lastInput1 = newInput1;
                            gotInput1 = true;
                        }
                    }

                    if (inputQueue2 != null) {
                        Mat newInput2 = inputQueue2.poll(10, TimeUnit.MILLISECONDS);
                        if (newInput2 != null) {
                            if (lastInput2 != null) lastInput2.release();
                            lastInput2 = newInput2;
                            gotInput2 = true;
                        }
                    }

                    // Determine if we should process based on sync mode
                    boolean shouldProcess;
                    if (queuesInSync) {
                        // Only process if both queues got new data
                        shouldProcess = gotInput1 && gotInput2;
                    } else {
                        // Process if either queue got new data
                        shouldProcess = gotInput1 || gotInput2;
                    }

                    // Process if we have at least one valid input and should process
                    if (shouldProcess && (lastInput1 != null || lastInput2 != null)) {
                        Mat output = processDual(lastInput1, lastInput2);

                        // Increment work units regardless of output (even if null)
                        incrementWorkUnits();

                        if (output != null) {
                            // Update thumbnail
                            setOutputMat(output);
                            notifyFrame(output);

                            // Send to output queue if available
                            if (outputQueue != null) {
                                try {
                                    outputQueue.put(output.clone());
                                } catch (InterruptedException e) {
                                    Thread.currentThread().interrupt();
                                    break;
                                }
                            }

                            output.release();
                        }
                    }

                    // Check for backpressure
                    checkBackpressure();

                    // Frame delay if configured
                    if (frameDelayMs > 0) {
                        Thread.sleep(frameDelayMs);
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }, "Processing-" + name + "-Thread");
        processingThread.setPriority(threadPriority);
        processingThread.start();
    }

    @Override
    public void stopProcessing() {
        super.stopProcessing();
        // Clear cached frames
        if (lastInput1 != null) {
            lastInput1.release();
            lastInput1 = null;
        }
        if (lastInput2 != null) {
            lastInput2.release();
            lastInput2 = null;
        }
        // Do NOT clear inputQueue2 - it keeps its data
    }
}
