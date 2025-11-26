package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.registry.NodeInfo;
import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.*;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;

/**
 * Bitwise XOR node - performs bitwise XOR on two images.
 * Has two input connection points for receiving images from two sources.
 */
@NodeInfo(name = "BitwiseXor", category = "Dual Input", aliases = {"Bitwise Xor"})
public class BitwiseXorNode extends DualInputNode {
    public BitwiseXorNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Bitwise XOR", x, y);
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

        Core.bitwise_xor(input1, converted2, output);

        // Clean up temporary mats
        if (resized2 != input2) resized2.release();
        if (converted2 != resized2) converted2.release();

        return output;
    }

    @Override
    public void paint(GC gc) {
        // Draw node background
        gc.setBackground(new Color(200, 230, 255)); // Light blue for bitwise
        gc.fillRoundRectangle(x, y, width, height, 10, 10);

        // Draw border
        gc.setForeground(new Color(50, 100, 200));
        gc.setLineWidth(2);
        gc.drawRoundRectangle(x, y, width, height, 10, 10);

        // Draw title
        gc.setForeground(display.getSystemColor(SWT.COLOR_BLACK));
        Font boldFont = new Font(display, "Arial", 10, SWT.BOLD);
        gc.setFont(boldFont);
        gc.drawString(name, x + 10, y + 5, true);
        boldFont.dispose();

        // Draw thumbnail if available
        if (thumbnail != null && !thumbnail.isDisposed()) {
            Rectangle bounds = thumbnail.getBounds();
            int thumbX = x + (width - bounds.width) / 2;
            int thumbY = y + 25;
            gc.drawImage(thumbnail, thumbX, thumbY);
        } else {
            gc.setForeground(display.getSystemColor(SWT.COLOR_GRAY));
            gc.drawString("(no output)", x + 10, y + 40, true);
        }

        // Draw connection points (custom for dual input)
        drawDualInputConnectionPoints(gc);
    }

    protected void drawDualInputConnectionPoints(GC gc) {
        int radius = 6;

        // Draw first input point (top left)
        Point input1 = getInputPoint();
        Color input1BgColor = new Color(200, 220, 255);
        gc.setBackground(input1BgColor);
        gc.fillOval(input1.x - radius, input1.y - radius, radius * 2, radius * 2);
        input1BgColor.dispose();
        Color input1FgColor = new Color(70, 100, 180);
        gc.setForeground(input1FgColor);
        gc.setLineWidth(2);
        gc.drawOval(input1.x - radius, input1.y - radius, radius * 2, radius * 2);
        input1FgColor.dispose();

        // Draw second input point (bottom left)
        Point input2 = getInputPoint2();
        Color input2BgColor = new Color(200, 220, 255);
        gc.setBackground(input2BgColor);
        gc.fillOval(input2.x - radius, input2.y - radius, radius * 2, radius * 2);
        input2BgColor.dispose();
        Color input2FgColor = new Color(70, 100, 180);
        gc.setForeground(input2FgColor);
        gc.drawOval(input2.x - radius, input2.y - radius, radius * 2, radius * 2);
        input2FgColor.dispose();

        // Draw output point on right side
        Point output = getOutputPoint();
        Color outputBgColor = new Color(255, 230, 200);
        gc.setBackground(outputBgColor);
        gc.fillOval(output.x - radius, output.y - radius, radius * 2, radius * 2);
        outputBgColor.dispose();
        Color outputFgColor = new Color(200, 120, 50);
        gc.setForeground(outputFgColor);
        gc.drawOval(output.x - radius, output.y - radius, radius * 2, radius * 2);
        outputFgColor.dispose();
        gc.setLineWidth(1);
    }

    @Override
    public String getDescription() {
        return "Bitwise XOR\ncv2.bitwise_xor(img1, img2)";
    }

    @Override
    public String getDisplayName() {
        return "Bitwise XOR";
    }

    @Override
    public String getCategory() {
        return "Dual Input Nodes";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Bitwise XOR Properties");
        dialog.setLayout(new GridLayout(2, false));

        // Description
        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText(getDescription() + "\n\nConnect two image sources to the input points.");
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sigGd.horizontalSpan = 2;
        sigLabel.setLayoutData(sigGd);

        // Queues In Sync checkbox
        Button syncCheckbox = new Button(dialog, SWT.CHECK);
        syncCheckbox.setText("Queues In Sync");
        syncCheckbox.setSelection(queuesInSync);
        syncCheckbox.setToolTipText("When checked, only process when both inputs receive new frames");
        GridData syncGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        syncGd.horizontalSpan = 2;
        syncCheckbox.setLayoutData(syncGd);

        // Buttons
        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(1, true));
        GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
        gd.horizontalSpan = 2;
        buttonComp.setLayoutData(gd);

        Button okBtn = new Button(buttonComp, SWT.PUSH);
        okBtn.setText("OK");
        dialog.setDefaultButton(okBtn);
        okBtn.addListener(SWT.Selection, e -> {
            queuesInSync = syncCheckbox.getSelection();
            dialog.dispose();
        });

        dialog.pack();
        org.eclipse.swt.graphics.Point cursorLoc = shell.getDisplay().getCursorLocation();
        dialog.setLocation(cursorLoc.x, cursorLoc.y);
        dialog.open();
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
