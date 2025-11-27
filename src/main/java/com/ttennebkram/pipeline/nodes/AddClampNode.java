package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.registry.NodeInfo;
import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Color;
import org.eclipse.swt.graphics.Font;
import org.eclipse.swt.graphics.GC;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.graphics.Rectangle;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;

/**
 * Add with Clamping node - adds two images together with clamping to 0-255.
 * Has two input connection points for receiving images from two sources.
 */
@NodeInfo(name = "AddClamp", category = "Dual Input", aliases = {"Add Clamp"})
public class AddClampNode extends DualInputNode {

    public AddClampNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Add w/Clamp", x, y);
    }

    // Get second input connection point (below the first)
    public Point getInputPoint2() {
        return new Point(x, y + height * 3 / 4);
    }

    @Override
    public Point getInputPoint() {
        // First input point is at 1/4 height
        return new Point(x, y + height / 4);
    }

    @Override
    public Mat process(Mat input) {
        // This is called with single input; for dual input we use processDual
        return input;
    }

    @Override
    public Mat processDual(Mat input1, Mat input2) {
        if (input1 == null || input2 == null) {
            return input1 != null ? input1.clone() : (input2 != null ? input2.clone() : null);
        }

        Mat output = new Mat();

        // Resize input2 to match input1 if sizes differ
        Mat resized2 = input2;
        if (input1.width() != input2.width() || input1.height() != input2.height()) {
            resized2 = new Mat();
            org.opencv.imgproc.Imgproc.resize(input2, resized2, new Size(input1.width(), input1.height()));
        }

        // Convert to same type if needed
        Mat converted2 = resized2;
        if (input1.type() != resized2.type()) {
            converted2 = new Mat();
            resized2.convertTo(converted2, input1.type());
        }

        Core.add(input1, converted2, output);

        // Clean up temporary mats
        if (resized2 != input2) resized2.release();
        if (converted2 != resized2) converted2.release();

        return output;
    }

    @Override
    public void paint(GC gc) {
        // Draw node background
        Color bgColor = new Color(255, 230, 200); // Light orange for arithmetic
        gc.setBackground(bgColor);
        gc.fillRoundRectangle(x, y, width, height, 10, 10);
        bgColor.dispose();

        // Draw border
        Color borderColor = new Color(200, 100, 0);
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
        gc.setForeground(display.getSystemColor(SWT.COLOR_DARK_GRAY));
        gc.drawString(getThreadPriorityLabel(), x + 10, y + 20, true);
        smallFont.dispose();

        // Draw input read counts on the left side
        Font tinyFont = new Font(display, "Arial", 7, SWT.NORMAL);
        gc.setFont(tinyFont);
        gc.setForeground(display.getSystemColor(SWT.COLOR_DARK_GRAY));
        int statsX = x + 5;
        // Display input counts
        if (inputReads1 >= 1000) {
            gc.drawString("In1:", statsX, y + 40, true);
            gc.drawString(formatNumber(inputReads1), statsX, y + 50, true);
        } else {
            gc.drawString("In1:" + formatNumber(inputReads1), statsX, y + 40, true);
        }
        if (inputReads2 >= 1000) {
            gc.drawString("In2:", statsX, y + 70, true);
            gc.drawString(formatNumber(inputReads2), statsX, y + 80, true);
        } else {
            gc.drawString("In2:" + formatNumber(inputReads2), statsX, y + 70, true);
        }
        tinyFont.dispose();

        // Draw thumbnail if available (offset to right to not overlap stats)
        Rectangle bounds = getThumbnailBounds();
        if (bounds != null) {
            int thumbX = x + 40;
            int thumbY = y + 35;
            drawThumbnail(gc, thumbX, thumbY);
        } else {
            gc.setForeground(display.getSystemColor(SWT.COLOR_GRAY));
            gc.drawString("(no output)", x + 45, y + 50, true);
        }

        // Draw connection points (custom for dual input)
        drawDualInputConnectionPoints(gc);
    }

    protected void drawDualInputConnectionPoints(GC gc) {
        int radius = 6;

        // Draw first input point (top left)
        Point input1 = getInputPoint();
        gc.setBackground(new Color(200, 220, 255));
        gc.fillOval(input1.x - radius, input1.y - radius, radius * 2, radius * 2);
        gc.setForeground(new Color(70, 100, 180));
        gc.setLineWidth(2);
        gc.drawOval(input1.x - radius, input1.y - radius, radius * 2, radius * 2);

        // Draw second input point (bottom left)
        Point input2 = getInputPoint2();
        gc.setBackground(new Color(200, 220, 255));
        gc.fillOval(input2.x - radius, input2.y - radius, radius * 2, radius * 2);
        gc.setForeground(new Color(70, 100, 180));
        gc.drawOval(input2.x - radius, input2.y - radius, radius * 2, radius * 2);

        // Draw output point on right side
        Point output = getOutputPoint();
        gc.setBackground(new Color(255, 230, 200));
        gc.fillOval(output.x - radius, output.y - radius, radius * 2, radius * 2);
        gc.setForeground(new Color(200, 120, 50));
        gc.drawOval(output.x - radius, output.y - radius, radius * 2, radius * 2);
        gc.setLineWidth(1);
    }

    @Override
    public String getDescription() {
        return "Add Images with Clamping\ncv2.add(img1, img2)";
    }

    @Override
    public String getDisplayName() {
        return "Add w/Clamp";
    }

    @Override
    public String getCategory() {
        return "Dual Input Nodes";
    }

    @Override
    protected Runnable addPropertiesContent(Shell dialog, int columns) {
        // Description
        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText(getDescription() + "\n\nConnect two image sources to the input points.");
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sigGd.horizontalSpan = columns;
        sigLabel.setLayoutData(sigGd);

        // Queues In Sync checkbox - use Label + Checkbox pattern for consistent styling
        new Label(dialog, SWT.NONE).setText("Queues In Sync:");
        Button syncCheckbox = new Button(dialog, SWT.CHECK);
        syncCheckbox.setSelection(queuesInSync);
        syncCheckbox.setToolTipText("When checked, only process when both inputs receive new frames");

        return () -> {
            queuesInSync = syncCheckbox.getSelection();
        };
    }

    @Override
    public void serializeProperties(JsonObject json) {
        // No properties to serialize
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        // No properties to deserialize
    }
}
