package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.registry.NodeInfo;
import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.*;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Mat;

import java.util.concurrent.BlockingQueue;

/**
 * Container Output Node - boundary node that sends frames to the parent container's output.
 * Acts as a sink inside the container, forwarding processed frames to the external output queue.
 * Renders with a hollow/clear circle to indicate it's a boundary point.
 */
@NodeInfo(name = "ContainerOutput", category = "Container", aliases = {"Container Output"})
public class ContainerOutputNode extends ProcessingNode {

    // Reference to the parent container's output queue
    private BlockingQueue<Mat> containerOutputQueue;

    // Reference to parent container (for context)
    private ContainerNode parentContainer;

    // Fixed dimensions for boundary nodes - wider to accommodate bolt circles and content
    private static final int BOUNDARY_NODE_WIDTH = 210;
    private static final int BOUNDARY_NODE_HEIGHT = 130;

    public ContainerOutputNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Container Output", x, y);
        this.width = BOUNDARY_NODE_WIDTH;
        this.height = BOUNDARY_NODE_HEIGHT;
    }

    /**
     * Set the parent container (for queue wiring).
     */
    public void setParentContainer(ContainerNode parent) {
        this.parentContainer = parent;
    }

    /**
     * Set the container's output queue that this node writes to.
     */
    public void setContainerOutputQueue(BlockingQueue<Mat> queue) {
        this.containerOutputQueue = queue;
    }

    /**
     * Process incoming frame by forwarding it to the container's external output queue.
     */
    @Override
    public Mat process(Mat input) {
        if (input == null) {
            return null;
        }

        // Forward to external output queue
        if (containerOutputQueue != null) {
            try {
                containerOutputQueue.put(input.clone());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        // Notify parent container of output frame for thumbnail update
        if (parentContainer != null) {
            parentContainer.onOutputFrame(input);
        }

        // Return the input unchanged (for preview/thumbnail)
        return input;
    }

    @Override
    public String getDescription() {
        return "Container output boundary.\nForwards processed frames to the parent pipeline.";
    }

    @Override
    public String getDisplayName() {
        return "Container Output";
    }

    @Override
    public String getCategory() {
        return "Container";
    }

    /**
     * Paint the container output node as a bracket bolted to the right edge.
     */
    @Override
    public void paint(GC gc) {
        // Draw as a bracket/tab shape attached to right edge
        // Right side is flat (against edge), left side is rounded

        // Main bracket background - metallic gray-orange
        Color bgColor = new Color(210, 195, 180);
        gc.setBackground(bgColor);

        // Draw bracket shape: rounded left, flat right
        int[] bracketPoints = {
            x, y + 15,                      // Top-left curve point
            x + 15, y,                      // Top after curve
            x + width, y,                   // Top-right (flat edge)
            x + width, y + height,          // Bottom-right (flat edge)
            x + 15, y + height,             // Bottom after curve
            x, y + height - 15              // Bottom-left curve point
        };
        gc.fillPolygon(bracketPoints);
        bgColor.dispose();

        // Draw bracket border
        Color borderColor = new Color(140, 120, 100);
        gc.setForeground(borderColor);
        gc.setLineWidth(2);
        gc.drawPolygon(bracketPoints);
        borderColor.dispose();

        // Draw "bolt" circles on right edge (3 bolts)
        Color boltColor = new Color(100, 90, 80);
        gc.setBackground(boltColor);
        int boltRadius = 4;
        int boltX = x + width - 8;
        gc.fillOval(boltX - boltRadius, y + 15 - boltRadius, boltRadius * 2, boltRadius * 2);
        gc.fillOval(boltX - boltRadius, y + height/2 - boltRadius, boltRadius * 2, boltRadius * 2);
        gc.fillOval(boltX - boltRadius, y + height - 15 - boltRadius, boltRadius * 2, boltRadius * 2);
        boltColor.dispose();

        // Draw title "Output Images" on same line
        gc.setForeground(display.getSystemColor(SWT.COLOR_BLACK));
        Font boldFont = new Font(display, "Arial", 9, SWT.BOLD);
        gc.setFont(boldFont);
        gc.drawString("Output Images", x + 20, y + 8, true);
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
        gc.drawString(getThreadPriorityLabel(), x + 20, y + 22, true);
        smallFont.dispose();

        // Draw input read count on the left side
        Font statsFont = new Font(display, "Arial", 7, SWT.NORMAL);
        gc.setFont(statsFont);
        gc.setForeground(display.getSystemColor(SWT.COLOR_DARK_GRAY));
        int statsX = x + 20;
        if (inputReads1 >= 1000) {
            gc.drawString("In:", statsX, y + 38, true);
            gc.drawString(formatNumber(inputReads1), statsX, y + 48, true);
        } else {
            gc.drawString("In:" + formatNumber(inputReads1), statsX, y + 38, true);
        }
        statsFont.dispose();

        // Draw thumbnail if available (positioned to not overlap stats)
        Rectangle bounds = getThumbnailBounds();
        if (bounds != null) {
            int thumbX = x + 50;  // Move right to not overlap stats
            int thumbY = y + 38;
            drawThumbnail(gc, thumbX, thumbY);
        }

        // Draw hollow input connection point
        drawConnectionPoints(gc);
    }

    /**
     * Draw hollow/clear circle for input (boundary style).
     */
    @Override
    protected void drawConnectionPoints(GC gc) {
        int radius = 10; // Larger than normal for boundary
        Point input = getInputPoint();

        // Hollow circle - white fill with orange border
        gc.setBackground(display.getSystemColor(SWT.COLOR_WHITE));
        gc.fillOval(input.x - radius, input.y - radius, radius * 2, radius * 2);

        Color borderColor = new Color(200, 140, 80); // Orange tint for output boundary
        gc.setForeground(borderColor);
        gc.setLineWidth(3);
        gc.drawOval(input.x - radius, input.y - radius, radius * 2, radius * 2);

        // Draw arrow pointing right inside the circle (showing flow direction)
        gc.setLineWidth(2);
        int arrowSize = 5;
        gc.drawLine(input.x - arrowSize, input.y, input.x + arrowSize, input.y);
        gc.drawLine(input.x + arrowSize, input.y, input.x + 2, input.y - 4);
        gc.drawLine(input.x + arrowSize, input.y, input.x + 2, input.y + 4);

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
        box.setText("Container Output");
        box.setMessage("This is the container's output boundary.\n\n" +
                       "Frames processed inside the container exit here.\n" +
                       "Connect processing nodes to this input.");
        box.open();
    }

    public void dispose() {
        // No special resources to dispose
    }

    @Override
    public String getInputTooltip() {
        return "Container Output " + CONNECTION_DATA_TYPE;
    }

    @Override
    public String getOutputTooltip(int index) {
        return null; // No output connection point shown inside container
    }

    @Override
    public void serializeProperties(JsonObject json) {
        super.serializeProperties(json);
        // No custom properties - position is handled by serializeCommon
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        super.deserializeProperties(json);
        // No custom properties
    }

    /**
     * Indicate this node has no output connection point
     * (the output goes to the external queue, not to another internal node).
     */
    public boolean hasOutput() {
        return false;
    }
}
