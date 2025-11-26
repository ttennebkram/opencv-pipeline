package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.registry.NodeInfo;
import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.*;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Mat;

import java.util.concurrent.BlockingQueue;

/**
 * Container Input Node - boundary node that receives frames from the parent container's input.
 * Acts as a SourceNode inside the container, but pulls frames from the external input queue.
 * Renders as a clear/hollow circle to indicate it's a boundary point.
 */
@NodeInfo(name = "ContainerInput", category = "Container", aliases = {"Container Input"})
public class ContainerInputNode extends SourceNode {

    // Reference to the parent container's input queue
    private BlockingQueue<Mat> containerInputQueue;

    // Reference to parent container (for context)
    private ContainerNode parentContainer;

    // Fixed width for boundary nodes - wider to accommodate bolt circles and content
    private static final int BOUNDARY_NODE_WIDTH = 210;
    private static final int BOUNDARY_NODE_HEIGHT = 130;

    public ContainerInputNode(Shell shell, Display display, int x, int y) {
        this.shell = shell;
        this.display = display;
        this.x = x;
        this.y = y;
        this.width = BOUNDARY_NODE_WIDTH;
        this.height = BOUNDARY_NODE_HEIGHT;
        // Support multiple outputs so we can connect to multiple nodes
        this.outputCount = 4; // Allow up to 4 connections from container input
        this.outputQueues = new java.util.concurrent.BlockingQueue[outputCount];
    }

    /**
     * Set the parent container (for queue wiring).
     */
    public void setParentContainer(ContainerNode parent) {
        this.parentContainer = parent;
    }

    /**
     * Set the container's input queue that this node reads from.
     */
    public void setContainerInputQueue(BlockingQueue<Mat> queue) {
        this.containerInputQueue = queue;
    }

    /**
     * Get the next frame by reading from the container's external input queue.
     */
    @Override
    public Mat getNextFrame() {
        if (containerInputQueue == null) {
            return null;
        }
        try {
            Mat frame = containerInputQueue.take();
            incrementInputReads1();
            return frame;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return null;
        }
    }

    /**
     * No frame rate throttling - process frames as fast as they arrive.
     */
    @Override
    public double getFps() {
        return 0; // No throttling
    }

    /**
     * Receive a slowdown signal from a downstream node inside the container.
     * Since ContainerInputNode has no FPS to reduce, it cascades the slowdown
     * to the parent container's upstream node.
     */
    @Override
    public synchronized void receiveSlowdownSignal() {
        long now = System.currentTimeMillis();
        lastSlowdownReceivedTime = now;
        inSlowdownMode = true;

        Thread pt = processingThread; // Local copy for thread safety
        if (pt != null && pt.isAlive()) {
            int currentPriority = pt.getPriority();

            if (currentPriority > Thread.MIN_PRIORITY) {
                // First, reduce priority like other nodes
                int newPriority = currentPriority - 1;
                System.out.println("[" + timestamp() + "] [" + getClass().getSimpleName() + "] RECEIVED SLOWDOWN, " +
                    "lowering priority: " + currentPriority + " -> " + newPriority);
                pt.setPriority(newPriority);
                lastRunningPriority = newPriority;
                slowdownPriorityReduction++;
            } else {
                // Already at minimum priority, cascade to parent container's upstream
                System.out.println("[" + timestamp() + "] [" + getClass().getSimpleName() + "] RECEIVED SLOWDOWN at min priority, cascading to parent container");
                if (parentContainer != null) {
                    // Signal the container's upstream node directly
                    PipelineNode containerUpstream = parentContainer.getInputNode();
                    if (containerUpstream != null) {
                        containerUpstream.receiveSlowdownSignal();
                    }
                }
            }
        }
    }

    /**
     * Override startProcessing to not apply frame rate delay.
     */
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
                    // Check for slowdown recovery
                    checkSlowdownRecovery();

                    Mat frame = getNextFrame();

                    if (frame != null) {
                        incrementWorkUnits();

                        // Clone for persistent storage
                        setOutputMat(frame.clone());

                        // Clone for preview callback
                        Mat previewClone = frame.clone();
                        notifyFrame(previewClone);

                        // Put frame on all output queues (for multi-output support)
                        if (outputQueues != null) {
                            for (int i = 0; i < outputQueues.length; i++) {
                                if (outputQueues[i] != null) {
                                    outputQueues[i].put(frame.clone());
                                }
                            }
                        } else if (outputQueue != null) {
                            outputQueue.put(frame.clone());
                        }

                        // Release the original frame
                        frame.release();

                        // Check for backpressure
                        checkBackpressure();
                    }

                    // No frame delay - process as fast as input arrives

                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }, "ContainerInput-Thread");
        processingThread.setPriority(threadPriority);
        processingThread.start();
    }

    /**
     * Paint the container input node as a bracket bolted to the left edge.
     */
    @Override
    public void paint(GC gc) {
        // Draw as a bracket/tab shape attached to left edge
        // Left side is flat (against edge), right side is rounded

        // Main bracket background - metallic gray-blue
        Color bgColor = new Color(180, 195, 210);
        gc.setBackground(bgColor);

        // Draw bracket shape: flat left, rounded right
        int[] bracketPoints = {
            x, y,                           // Top-left (flat edge)
            x + width - 15, y,              // Top before curve
            x + width, y + 15,              // Top-right curve point
            x + width, y + height - 15,     // Bottom-right curve point
            x + width - 15, y + height,     // Bottom before curve
            x, y + height                   // Bottom-left (flat edge)
        };
        gc.fillPolygon(bracketPoints);
        bgColor.dispose();

        // Draw bracket border
        Color borderColor = new Color(100, 120, 140);
        gc.setForeground(borderColor);
        gc.setLineWidth(2);
        gc.drawPolygon(bracketPoints);
        borderColor.dispose();

        // Draw "bolt" circles on left edge (3 bolts)
        Color boltColor = new Color(80, 90, 100);
        gc.setBackground(boltColor);
        int boltRadius = 4;
        int boltX = x + 8;
        gc.fillOval(boltX - boltRadius, y + 15 - boltRadius, boltRadius * 2, boltRadius * 2);
        gc.fillOval(boltX - boltRadius, y + height/2 - boltRadius, boltRadius * 2, boltRadius * 2);
        gc.fillOval(boltX - boltRadius, y + height - 15 - boltRadius, boltRadius * 2, boltRadius * 2);
        boltColor.dispose();

        // Draw title "Input Images" on same line
        gc.setForeground(display.getSystemColor(SWT.COLOR_BLACK));
        Font boldFont = new Font(display, "Arial", 9, SWT.BOLD);
        gc.setFont(boldFont);
        gc.drawString("Input Images", x + 25, y + 8, true);
        boldFont.dispose();

        // Draw thread priority label (red if priority < 5)
        Font smallFont = new Font(display, "Arial", 8, SWT.NORMAL);
        gc.setFont(smallFont);
        int currentPriority = getThreadPriority();
        if (currentPriority < 5) {
            Color redColor = new Color(200, 0, 0);
            gc.setForeground(redColor);
            redColor.dispose();
        } else {
            gc.setForeground(display.getSystemColor(SWT.COLOR_DARK_GRAY));
        }
        gc.drawString(getThreadPriorityLabel(), x + 25, y + 22, true);
        smallFont.dispose();

        // Draw input read count on the left side (below priority)
        Font statsFont = new Font(display, "Arial", 7, SWT.NORMAL);
        gc.setFont(statsFont);
        gc.setForeground(display.getSystemColor(SWT.COLOR_DARK_GRAY));
        int statsX = x + 25;
        if (inputReads1 >= 1000) {
            gc.drawString("In:", statsX, y + 38, true);
            gc.drawString(formatNumber(inputReads1), statsX, y + 48, true);
        } else {
            gc.drawString("In:" + formatNumber(inputReads1), statsX, y + 38, true);
        }
        statsFont.dispose();

        // Draw thumbnail if available (positioned to not overlap stats)
        if (thumbnail != null && !thumbnail.isDisposed()) {
            Rectangle bounds = thumbnail.getBounds();
            int thumbX = x + 55;  // Move right to not overlap stats
            int thumbY = y + 38;
            gc.drawImage(thumbnail, thumbX, thumbY);
        }

        // Draw hollow output connection point
        drawConnectionPoints(gc);
    }

    /**
     * Draw hollow/clear circle for output (boundary style).
     */
    @Override
    protected void drawConnectionPoints(GC gc) {
        int radius = 10; // Larger than normal for boundary
        Point output = getOutputPoint();

        // Hollow circle - white fill with green border
        gc.setBackground(display.getSystemColor(SWT.COLOR_WHITE));
        gc.fillOval(output.x - radius, output.y - radius, radius * 2, radius * 2);

        Color borderColor = new Color(80, 160, 100); // Green tint for input boundary
        gc.setForeground(borderColor);
        gc.setLineWidth(3);
        gc.drawOval(output.x - radius, output.y - radius, radius * 2, radius * 2);

        // Draw arrow pointing right inside the circle
        gc.setLineWidth(2);
        int arrowSize = 5;
        gc.drawLine(output.x - arrowSize, output.y, output.x + arrowSize, output.y);
        gc.drawLine(output.x + arrowSize, output.y, output.x + 2, output.y - 4);
        gc.drawLine(output.x + arrowSize, output.y, output.x + 2, output.y + 4);

        borderColor.dispose();
        gc.setLineWidth(1);
    }

    /**
     * Container boundary nodes don't show properties dialog.
     */
    @Override
    public void showPropertiesDialog() {
        // No properties to edit for boundary nodes
        MessageBox box = new MessageBox(shell, SWT.ICON_INFORMATION | SWT.OK);
        box.setText("Container Input");
        box.setMessage("This is the container's input boundary.\n\n" +
                       "Frames from the parent pipeline enter the container here.\n" +
                       "Connect this output to processing nodes inside the container.");
        box.open();
    }

    public void dispose() {
        // No special resources to dispose
    }

    @Override
    public String getInputTooltip() {
        return null; // No input
    }

    @Override
    public String getOutputTooltip(int index) {
        return "Container Input " + CONNECTION_DATA_TYPE;
    }

    @Override
    public void serializeProperties(JsonObject json) {
        // No custom properties - position is handled by serializeCommon
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        // No custom properties
    }
}
