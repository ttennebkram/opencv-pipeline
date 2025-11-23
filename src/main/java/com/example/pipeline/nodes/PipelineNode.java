package com.example.pipeline.nodes;

import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.*;
import org.eclipse.swt.widgets.Display;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Base class for all pipeline nodes.
 */
public abstract class PipelineNode {
    // Node dimension constants
    public static final int PROCESSING_NODE_THUMB_WIDTH = 120;
    public static final int PROCESSING_NODE_THUMB_HEIGHT = 80;
    public static final int SOURCE_NODE_THUMB_WIDTH = 280;
    public static final int SOURCE_NODE_THUMB_HEIGHT = 90;

    public static final int NODE_WIDTH = PROCESSING_NODE_THUMB_WIDTH + 20;  // thumbnail + 10px padding each side
    public static final int NODE_HEIGHT = PROCESSING_NODE_THUMB_HEIGHT + 40; // thumbnail + 25px title + 15px bottom
    public static final int SOURCE_NODE_HEIGHT = SOURCE_NODE_THUMB_HEIGHT + 32; // thumbnail + 22px title + 10px bottom

    protected Display display;
    public int x, y;
    public int width = NODE_WIDTH;
    public int height = NODE_HEIGHT;
    protected Image thumbnail;
    protected Mat outputMat;

    public abstract void paint(GC gc);

    public boolean containsPoint(Point p) {
        return p.x >= x && p.x <= x + width && p.y >= y && p.y <= y + height;
    }

    public Point getOutputPoint() {
        return new Point(x + width, y + height / 2);
    }

    public Point getInputPoint() {
        return new Point(x, y + height / 2);
    }

    public int getX() {
        return x;
    }

    public void setX(int x) {
        this.x = x;
    }

    public int getY() {
        return y;
    }

    public void setY(int y) {
        this.y = y;
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    // Draw connection points (circles) on the node - always visible
    protected void drawConnectionPoints(GC gc) {
        int radius = 6;  // Slightly larger for visibility

        // Draw input point on left side (blue tint for input)
        Point input = getInputPoint();
        gc.setBackground(new Color(200, 220, 255));  // Light blue fill
        gc.fillOval(input.x - radius, input.y - radius, radius * 2, radius * 2);
        gc.setForeground(new Color(70, 100, 180));   // Blue border
        gc.setLineWidth(2);
        gc.drawOval(input.x - radius, input.y - radius, radius * 2, radius * 2);

        // Draw output point on right side (orange tint for output)
        Point output = getOutputPoint();
        gc.setBackground(new Color(255, 230, 200)); // Light orange fill
        gc.fillOval(output.x - radius, output.y - radius, radius * 2, radius * 2);
        gc.setForeground(new Color(200, 120, 50));  // Orange border
        gc.drawOval(output.x - radius, output.y - radius, radius * 2, radius * 2);
        gc.setLineWidth(1);  // Reset line width
    }

    // Draw selection highlight around node
    public void drawSelectionHighlight(GC gc, boolean isSelected) {
        if (isSelected) {
            gc.setForeground(new Color(0, 120, 215));  // Blue selection color
            gc.setLineWidth(3);
            gc.drawRoundRectangle(x - 3, y - 3, width + 6, height + 6, 13, 13);
        }
    }

    public void setOutputMat(Mat mat) {
        this.outputMat = mat;
        updateThumbnail();
    }

    public Mat getOutputMat() {
        return outputMat;
    }

    protected void updateThumbnail() {
        if (outputMat == null || outputMat.empty()) {
            return;
        }

        // Dispose old thumbnail
        if (thumbnail != null && !thumbnail.isDisposed()) {
            thumbnail.dispose();
        }

        // Create thumbnail
        Mat resized = new Mat();
        double scale = Math.min((double) PROCESSING_NODE_THUMB_WIDTH / outputMat.width(),
                                (double) PROCESSING_NODE_THUMB_HEIGHT / outputMat.height());
        Imgproc.resize(outputMat, resized,
            new Size(outputMat.width() * scale, outputMat.height() * scale));

        // Convert to SWT Image
        Mat rgb = new Mat();
        if (resized.channels() == 3) {
            Imgproc.cvtColor(resized, rgb, Imgproc.COLOR_BGR2RGB);
        } else if (resized.channels() == 1) {
            Imgproc.cvtColor(resized, rgb, Imgproc.COLOR_GRAY2RGB);
        } else {
            rgb = resized;
        }

        int w = rgb.width();
        int h = rgb.height();
        byte[] data = new byte[w * h * 3];
        rgb.get(0, 0, data);

        // Create ImageData with proper scanline padding
        PaletteData palette = new PaletteData(0xFF0000, 0x00FF00, 0x0000FF);
        ImageData imageData = new ImageData(w, h, 24, palette);

        // Copy data row by row to handle scanline padding
        for (int row = 0; row < h; row++) {
            for (int col = 0; col < w; col++) {
                int srcIdx = (row * w + col) * 3;
                int r = data[srcIdx] & 0xFF;
                int g = data[srcIdx + 1] & 0xFF;
                int b = data[srcIdx + 2] & 0xFF;
                imageData.setPixel(col, row, (r << 16) | (g << 8) | b);
            }
        }

        thumbnail = new Image(display, imageData);
    }

    protected void drawThumbnail(GC gc, int thumbX, int thumbY) {
        if (thumbnail != null && !thumbnail.isDisposed()) {
            gc.drawImage(thumbnail, thumbX, thumbY);
        }
    }

    public void disposeThumbnail() {
        if (thumbnail != null && !thumbnail.isDisposed()) {
            thumbnail.dispose();
        }
    }
}
