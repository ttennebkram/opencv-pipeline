package com.ttennebkram.pipeline.nodes;

import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Color;
import org.eclipse.swt.graphics.Font;
import org.eclipse.swt.graphics.GC;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.widgets.Display;
import org.eclipse.swt.widgets.Shell;
import org.opencv.core.Mat;

import java.util.concurrent.BlockingQueue;

/**
 * Base class for nodes that have multiple outputs.
 * Handles output queue array management and connection point drawing.
 */
public abstract class MultiOutputNode extends ProcessingNode {
    // Output queues array
    protected BlockingQueue<Mat>[] multiOutputQueues;
    protected int numOutputs = 2;  // Default to 2 outputs

    // Connection point colors (matching DualInputNode style)
    private static final int INPUT_BG_R = 200;
    private static final int INPUT_BG_G = 220;
    private static final int INPUT_BG_B = 255;
    private static final int INPUT_FG_R = 70;
    private static final int INPUT_FG_G = 100;
    private static final int INPUT_FG_B = 180;
    private static final int OUTPUT_BG_R = 255;
    private static final int OUTPUT_BG_G = 230;
    private static final int OUTPUT_BG_B = 200;
    private static final int OUTPUT_FG_R = 200;
    private static final int OUTPUT_FG_G = 120;
    private static final int OUTPUT_FG_B = 50;

    public MultiOutputNode(Display display, Shell shell, String name, int x, int y) {
        super(display, shell, name, x, y);
        initMultiOutputQueues(2);
    }

    /**
     * Initialize the output queues array.
     */
    @SuppressWarnings("unchecked")
    protected void initMultiOutputQueues(int count) {
        this.numOutputs = count;
        this.multiOutputQueues = (BlockingQueue<Mat>[]) new BlockingQueue[count];
    }

    /**
     * Get the number of outputs this node has.
     */
    @Override
    public int getOutputCount() {
        return numOutputs;
    }

    /**
     * Check if this node has multiple outputs.
     */
    @Override
    public boolean hasMultipleOutputs() {
        return true;
    }

    /**
     * Set an output queue by index.
     */
    public void setMultiOutputQueue(int index, BlockingQueue<Mat> queue) {
        if (index >= 0 && index < numOutputs && multiOutputQueues != null) {
            multiOutputQueues[index] = queue;
            // Keep primary output in sync for backwards compatibility
            if (index == 0) {
                outputQueue = queue;
            }
        }
    }

    /**
     * Get an output queue by index.
     */
    public BlockingQueue<Mat> getMultiOutputQueue(int index) {
        if (multiOutputQueues != null && index >= 0 && index < multiOutputQueues.length) {
            return multiOutputQueues[index];
        }
        return null;
    }

    /**
     * Override setOutputQueue to also set in array.
     */
    @Override
    public void setOutputQueue(BlockingQueue<Mat> queue) {
        super.setOutputQueue(queue);
        if (multiOutputQueues != null && multiOutputQueues.length > 0) {
            multiOutputQueues[0] = queue;
        }
    }

    /**
     * Set output queue by index (called by Connection.activate).
     */
    @Override
    public void setOutputQueue(int index, BlockingQueue<Mat> queue) {
        setMultiOutputQueue(index, queue);
    }

    /**
     * Get output queue by index.
     */
    @Override
    public BlockingQueue<Mat> getOutputQueue(int index) {
        return getMultiOutputQueue(index);
    }

    /**
     * Get the output point for a specific output index.
     * Outputs are distributed vertically on the right side of the node.
     */
    @Override
    public Point getOutputPoint(int index) {
        if (index < 0 || index >= numOutputs) {
            return null;
        }
        // Distribute outputs vertically, evenly spaced
        int spacing = height / (numOutputs + 1);
        int yPos = y + spacing * (index + 1);
        return new Point(x + width, yPos);
    }

    /**
     * Get the primary output point (index 0).
     */
    @Override
    public Point getOutputPoint() {
        return getOutputPoint(0);
    }

    /**
     * Send a Mat to all output queues (cloning for each).
     * Returns true if at least one queue received the frame.
     */
    protected boolean sendToAllOutputs(Mat frame) throws InterruptedException {
        if (frame == null) return false;

        boolean sentAny = false;
        for (int i = 0; i < numOutputs; i++) {
            BlockingQueue<Mat> queue = getMultiOutputQueue(i);
            if (queue != null) {
                queue.put(frame.clone());
                sentAny = true;
            }
        }
        return sentAny;
    }

    @Override
    public void startProcessing() {
        if (running.get()) {
            return;
        }

        running.set(true);
        workUnitsCompleted = 0;

        processingThread = new Thread(() -> {
            while (running.get()) {
                try {
                    if (inputQueue == null) {
                        Thread.sleep(100);
                        continue;
                    }

                    Mat input = inputQueue.take();
                    incrementInputReads1();
                    if (input == null) {
                        continue;
                    }

                    // Process the frame (subclasses implement this)
                    Mat output = process(input);

                    incrementWorkUnits();

                    if (output != null) {
                        // Clone for persistent storage
                        setOutputMat(output.clone());

                        // Clone for preview callback
                        Mat previewClone = output.clone();
                        notifyFrame(previewClone);

                        // Send clone to each output queue
                        sendToAllOutputs(output);

                        output.release();
                    }

                    // Check for backpressure
                    checkBackpressure();

                    // Release input
                    input.release();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }, "MultiOutput-" + name + "-Thread");
        processingThread.setPriority(threadPriority);
        processingThread.start();
    }

    /**
     * Draw connection points for multi-output nodes.
     */
    protected void drawMultiOutputConnectionPoints(GC gc) {
        int radius = 6;

        // Draw input point on left side
        Point input = getInputPoint();
        Color inputBgColor = new Color(INPUT_BG_R, INPUT_BG_G, INPUT_BG_B);
        gc.setBackground(inputBgColor);
        gc.fillOval(input.x - radius, input.y - radius, radius * 2, radius * 2);
        inputBgColor.dispose();
        Color inputFgColor = new Color(INPUT_FG_R, INPUT_FG_G, INPUT_FG_B);
        gc.setForeground(inputFgColor);
        gc.setLineWidth(2);
        gc.drawOval(input.x - radius, input.y - radius, radius * 2, radius * 2);
        inputFgColor.dispose();

        // Draw output points on right side
        for (int i = 0; i < numOutputs; i++) {
            Point output = getOutputPoint(i);
            if (output != null) {
                Color outputBgColor = new Color(OUTPUT_BG_R, OUTPUT_BG_G, OUTPUT_BG_B);
                gc.setBackground(outputBgColor);
                gc.fillOval(output.x - radius, output.y - radius, radius * 2, radius * 2);
                outputBgColor.dispose();
                Color outputFgColor = new Color(OUTPUT_FG_R, OUTPUT_FG_G, OUTPUT_FG_B);
                gc.setForeground(outputFgColor);
                gc.drawOval(output.x - radius, output.y - radius, radius * 2, radius * 2);
                outputFgColor.dispose();

                // Draw output index label
                gc.setForeground(display.getSystemColor(SWT.COLOR_DARK_GRAY));
                Font tinyFont = new Font(display, "Arial", 7, SWT.NORMAL);
                gc.setFont(tinyFont);
                gc.drawString(String.valueOf(i + 1), output.x - 12, output.y - 4, true);
                tinyFont.dispose();
            }
        }
        gc.setLineWidth(1);
    }
}
