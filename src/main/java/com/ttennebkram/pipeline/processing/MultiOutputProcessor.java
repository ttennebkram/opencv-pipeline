package com.ttennebkram.pipeline.processing;

import com.ttennebkram.pipeline.fx.FXNode;
import org.opencv.core.Mat;

import java.util.concurrent.BlockingQueue;

/**
 * Processor for nodes that produce multiple outputs (e.g., FFTHighPass4, FFTLowPass4).
 * Instead of a single ImageProcessor, uses a MultiOutputImageProcessor that returns Mat[].
 */
public class MultiOutputProcessor extends ThreadedProcessor {

    private final MultiOutputImageProcessor multiProcessor;
    private final int outputCount;

    /**
     * Functional interface for multi-output image processing.
     */
    @FunctionalInterface
    public interface MultiOutputImageProcessor {
        /**
         * Process an input image and return multiple output images.
         * @param input The input image
         * @return Array of output images (must match expected output count)
         */
        Mat[] process(Mat input);
    }

    public MultiOutputProcessor(String name, MultiOutputImageProcessor processor, int outputCount) {
        super(name, null); // Pass null to parent - we override processing loop
        this.multiProcessor = processor;
        this.outputCount = outputCount;
    }

    @Override
    protected void processingLoop() {
        while (isRunning()) {
            try {
                BlockingQueue<Mat> inputQueue = getInputQueue();
                if (inputQueue == null) {
                    Thread.sleep(100);
                    continue;
                }

                // Check if enabled
                if (!isEnabled()) {
                    Mat input = inputQueue.poll(100, java.util.concurrent.TimeUnit.MILLISECONDS);
                    if (input != null) {
                        incrementInputReads1();
                        // Pass through to all output queues unchanged when disabled
                        for (int i = 0; i < outputCount; i++) {
                            BlockingQueue<Mat> outQueue = getOutputQueue(i);
                            if (outQueue != null) {
                                outQueue.put(input.clone());
                            }
                        }
                        input.release();
                    }
                    continue;
                }

                // Take input from queue (blocking)
                Mat input = inputQueue.take();
                incrementInputReads1();

                if (input == null || input.empty()) {
                    continue;
                }

                // Process and get multiple outputs
                Mat[] outputs = null;
                try {
                    outputs = multiProcessor.process(input);
                } catch (Exception e) {
                    System.err.println("MultiOutputProcessor error: " + e.getMessage());
                    input.release();
                    continue;
                }

                if (outputs == null) {
                    input.release();
                    continue;
                }

                incrementWorkUnitsCompleted();

                // Send each output to its respective queue
                for (int i = 0; i < Math.min(outputs.length, outputCount); i++) {
                    Mat output = outputs[i];
                    if (output != null && !output.empty()) {
                        BlockingQueue<Mat> outQueue = getOutputQueue(i);
                        if (outQueue != null) {
                            outQueue.put(output.clone());
                            incrementOutputWrites(i);
                        }

                        // Send thumbnail for first output (for node preview)
                        if (i == 0) {
                            notifyCallback(output.clone());
                        }
                    }
                }

                // Update FXNode counters
                FXNode fxNode = getFXNode();
                if (fxNode != null) {
                    for (int i = 0; i < Math.min(outputs.length, 4); i++) {
                        if (outputs[i] != null) {
                            switch (i) {
                                case 0: fxNode.outputCount1++; break;
                                case 1: fxNode.outputCount2++; break;
                                case 2: fxNode.outputCount3++; break;
                                case 3: fxNode.outputCount4++; break;
                            }
                        }
                    }
                }

                // Release all outputs
                for (Mat output : outputs) {
                    if (output != null) {
                        output.release();
                    }
                }

                // Check backpressure
                checkBackpressure();

                input.release();

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
    }

    /**
     * Track output writes per queue.
     */
    private void incrementOutputWrites(int index) {
        // Output tracking is done via FXNode counters directly
    }
}
