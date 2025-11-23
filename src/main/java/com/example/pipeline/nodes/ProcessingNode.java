package com.example.pipeline.nodes;

import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.*;
import org.eclipse.swt.widgets.Display;
import org.eclipse.swt.widgets.Shell;
import org.opencv.core.Mat;

/**
 * Base class for processing nodes with properties dialog support.
 */
public abstract class ProcessingNode extends PipelineNode {
    protected String name;
    protected Shell shell;
    protected boolean enabled = true;
    protected Runnable onChanged;  // Callback when properties change

    public ProcessingNode(Display display, Shell shell, String name, int x, int y) {
        this.display = display;
        this.shell = shell;
        this.name = name;
        this.x = x;
        this.y = y;
    }

    public void setOnChanged(Runnable onChanged) {
        this.onChanged = onChanged;
    }

    protected void notifyChanged() {
        if (onChanged != null) {
            onChanged.run();
        }
    }

    // Process input Mat and return output Mat
    public abstract Mat process(Mat input);

    // Show properties dialog
    public abstract void showPropertiesDialog();

    // Get description for tooltip
    public abstract String getDescription();

    @Override
    public void paint(GC gc) {
        // Draw node background
        gc.setBackground(new Color(230, 255, 230));
        gc.fillRoundRectangle(x, y, width, height, 10, 10);

        // Draw border
        gc.setForeground(new Color(0, 100, 0));
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
            // Draw placeholder
            gc.setForeground(display.getSystemColor(SWT.COLOR_GRAY));
            gc.drawString("(no output)", x + 10, y + 40, true);
        }

        // Draw connection points
        drawConnectionPoints(gc);
    }

    public String getName() {
        return name;
    }

    public boolean isEnabled() {
        return enabled;
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }

    public Shell getShell() {
        return shell;
    }

    public Display getDisplay() {
        return display;
    }
}
