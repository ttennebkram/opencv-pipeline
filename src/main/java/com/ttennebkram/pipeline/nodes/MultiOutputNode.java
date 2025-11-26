package com.ttennebkram.pipeline.nodes;

import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Color;
import org.eclipse.swt.graphics.Font;
import org.eclipse.swt.graphics.GC;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.graphics.Rectangle;
import org.eclipse.swt.widgets.Display;
import org.eclipse.swt.widgets.Shell;
import org.opencv.core.Mat;

import java.util.concurrent.BlockingQueue;

/**
 * Base class for nodes that have multiple outputs.
 * Handles output queue array management and connection point drawing.
 */
public abstract class MultiOutputNode extends ProcessingNode {
    // Output labels for tooltips (subclasses can override getOutputLabels)
    protected String[] outputLabels = null;
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
        this.outputCount = count; // Sync with base class for getOutputIndexNear()
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
     * Get output labels for tooltips. Subclasses can override this.
     * Default returns numbered labels: "Output 1", "Output 2", etc.
     */
    public String[] getOutputLabels() {
        if (outputLabels != null) {
            return outputLabels;
        }
        // Default labels
        String[] labels = new String[numOutputs];
        for (int i = 0; i < numOutputs; i++) {
            labels[i] = "Output " + (i + 1);
        }
        return labels;
    }

    /**
     * Set custom output labels.
     */
    protected void setOutputLabels(String... labels) {
        this.outputLabels = labels;
    }

    /**
     * Get the tooltip text for a specific output index.
     */
    @Override
    public String getOutputTooltip(int index) {
        String[] labels = getOutputLabels();
        if (index >= 0 && index < labels.length) {
            return labels[index] + " " + CONNECTION_DATA_TYPE;
        }
        return "Output " + (index + 1) + " " + CONNECTION_DATA_TYPE;
    }

    /**
     * Check if a point is near an output connection point and return the index.
     * Returns -1 if not near any output.
     */
    public int getOutputIndexAt(int px, int py) {
        int hitRadius = 12;
        for (int i = 0; i < numOutputs; i++) {
            Point output = getOutputPoint(i);
            if (output != null) {
                double dist = Math.sqrt((px - output.x) * (px - output.x) + (py - output.y) * (py - output.y));
                if (dist <= hitRadius) {
                    return i;
                }
            }
        }
        return -1;
    }

    /**
     * Paint the node. Overrides ProcessingNode.paint() to use multi-output connection points.
     */
    @Override
    public void paint(GC gc) {
        // Draw node background
        Color bgColor = getBackgroundColor();
        gc.setBackground(bgColor);
        gc.fillRoundRectangle(x, y, width, height, 10, 10);
        bgColor.dispose();

        // Draw border
        Color borderColor = getBorderColor();
        gc.setForeground(borderColor);
        gc.setLineWidth(2);
        gc.drawRoundRectangle(x, y, width, height, 10, 10);
        borderColor.dispose();

        // Draw title
        gc.setForeground(display.getSystemColor(SWT.COLOR_BLACK));
        Font boldFont = new Font(display, "Arial", 10, SWT.BOLD);
        gc.setFont(boldFont);
        gc.drawString(name, x + 10, y + 5, true);
        boldFont.dispose();

        // Draw thread priority label
        Font smallFont = new Font(display, "Arial", 8, SWT.NORMAL);
        gc.setFont(smallFont);
        int currentPriority = getThreadPriority();
        if (currentPriority < 5) {
            gc.setForeground(new Color(200, 0, 0));
        } else {
            gc.setForeground(display.getSystemColor(SWT.COLOR_DARK_GRAY));
        }
        gc.drawString(getThreadPriorityLabel(), x + 10, y + 20, true);
        smallFont.dispose();

        // Draw input stats (frame counts)
        drawInputStats(gc);

        // Draw thumbnail if available
        if (thumbnail != null && !thumbnail.isDisposed()) {
            Rectangle bounds = thumbnail.getBounds();
            int thumbX = x + 40;
            int thumbY = y + 35;
            gc.drawImage(thumbnail, thumbX, thumbY);
        } else {
            gc.setForeground(display.getSystemColor(SWT.COLOR_GRAY));
            gc.drawString("(no output)", x + 45, y + 50, true);
        }

        // Draw multi-output connection points (instead of single output)
        drawMultiOutputConnectionPoints(gc);
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

                        // Check for backpressure BEFORE trying to send (so we can lower priority while blocked)
                        checkBackpressure();

                        // Send clone to each output queue
                        sendToAllOutputs(output);

                        output.release();
                    }

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
     * Override checkBackpressure to use multiOutputQueues instead of outputQueues.
     * Triggers backpressure if ANY output queue has >= 5 items.
     */
    @Override
    protected void checkBackpressure() {
        // First, check if we should recover from slowdown mode
        checkSlowdownRecovery();

        if (multiOutputQueues == null) {
            return;
        }

        // Find the maximum queue size across all outputs
        int maxQueueSize = 0;
        for (int i = 0; i < multiOutputQueues.length; i++) {
            BlockingQueue<Mat> q = multiOutputQueues[i];
            if (q != null) {
                maxQueueSize = Math.max(maxQueueSize, q.size());
            }
        }

        if (processingThread != null && processingThread.isAlive()) {
            int currentPriority = processingThread.getPriority();
            long now = System.currentTimeMillis();

            // If in slowdown mode, don't raise priority - let checkSlowdownRecovery() handle it
            if (inSlowdownMode) {
                if (maxQueueSize >= 5) {
                    if (currentPriority > Thread.MIN_PRIORITY) {
                        // Lower by 1 if cooldown has passed
                        if (now - lastPriorityAdjustmentTime >= PRIORITY_LOWER_COOLDOWN_MS) {
                            int newPriority = currentPriority - 1;
                            System.out.println("[" + getClass().getSimpleName() + " " + name + "] LOWERING priority: " +
                                currentPriority + " -> " + newPriority + " (maxQueueSize=" + maxQueueSize + ", inSlowdownMode)");
                            processingThread.setPriority(newPriority);
                            lastPriorityAdjustmentTime = now;
                            lastRunningPriority = newPriority;
                        }
                    } else {
                        // At min priority, signal upstream
                        if (now - lastPriorityAdjustmentTime >= PRIORITY_LOWER_COOLDOWN_MS) {
                            signalUpstreamSlowdown();
                            lastPriorityAdjustmentTime = now;
                        }
                    }
                }
                return; // Don't raise priority - let slowdown recovery handle it
            }

            // Normal backpressure logic (not in slowdown mode)
            if (maxQueueSize >= 5) {
                // Queue backed up - lower priority by 1 if cooldown has passed
                if (currentPriority > Thread.MIN_PRIORITY) {
                    if (now - lastPriorityAdjustmentTime >= PRIORITY_LOWER_COOLDOWN_MS) {
                        int newPriority = currentPriority - 1;
                        System.out.println("[" + getClass().getSimpleName() + " " + name + "] LOWERING priority: " +
                            currentPriority + " -> " + newPriority + " (maxQueueSize=" + maxQueueSize + ")");
                        processingThread.setPriority(newPriority);
                        lastPriorityAdjustmentTime = now;
                        lastRunningPriority = newPriority;
                    }
                } else {
                    // At min priority and still backed up - signal upstream
                    if (now - lastPriorityAdjustmentTime >= PRIORITY_LOWER_COOLDOWN_MS) {
                        signalUpstreamSlowdown();
                        lastPriorityAdjustmentTime = now;
                    }
                }
            } else if (maxQueueSize == 0) {
                // Queue empty - raise priority by 1 toward original if cooldown has passed
                if (currentPriority < originalPriority) {
                    if (now - lastPriorityAdjustmentTime >= PRIORITY_RAISE_COOLDOWN_MS) {
                        int newPriority = currentPriority + 1;
                        System.out.println("[" + getClass().getSimpleName() + " " + name + "] RAISING priority: " +
                            currentPriority + " -> " + newPriority + " (maxQueueSize=" + maxQueueSize + ")");
                        processingThread.setPriority(newPriority);
                        lastPriorityAdjustmentTime = now;
                        lastRunningPriority = newPriority;
                    }
                }
            }
            // If queue is 1-4, do nothing - hold current priority
        }
    }

    /**
     * Draw input stats (frame counts) on the left side of the node.
     * Call this from paint() in subclasses that override paint().
     */
    protected void drawInputStats(GC gc) {
        Font tinyFont = new Font(display, "Arial", 7, SWT.NORMAL);
        gc.setFont(tinyFont);
        gc.setForeground(display.getSystemColor(SWT.COLOR_DARK_GRAY));
        int statsX = x + 5;
        // Display input counts, wrapping to next line if 4+ digits
        if (inputReads1 >= 1000) {
            gc.drawString("In1:", statsX, y + 40, true);
            gc.drawString(formatNumber(inputReads1), statsX, y + 50, true);
        } else {
            gc.drawString("In1:" + formatNumber(inputReads1), statsX, y + 40, true);
        }
        if (hasDualInput()) {
            if (inputReads2 >= 1000) {
                gc.drawString("In2:", statsX, y + 70, true);
                gc.drawString(formatNumber(inputReads2), statsX, y + 80, true);
            } else {
                gc.drawString("In2:" + formatNumber(inputReads2), statsX, y + 70, true);
            }
        }
        tinyFont.dispose();
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
