package com.ttennebkram.pipeline.ui;

import org.eclipse.swt.SWT;
import org.eclipse.swt.custom.ScrolledComposite;
import org.eclipse.swt.events.*;
import org.eclipse.swt.graphics.*;
import org.eclipse.swt.widgets.*;

import com.ttennebkram.pipeline.model.Connection;
import com.ttennebkram.pipeline.nodes.*;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Abstract base class for pipeline canvas editors.
 * Contains shared logic for:
 * - Connection drawing (bezier curves, arrows, queue stats)
 * - Mouse handling (selection, dragging, connection creation)
 * - Zoom support
 * - Selection box (marquee selection)
 * - Preview panel updates
 *
 * Subclasses provide the specific node/connection lists and UI chrome.
 */
public abstract class PipelineCanvasBase {

    // SWT components
    protected Shell shell;
    protected Display display;
    protected Canvas canvas;
    protected ScrolledComposite scrolledCanvas;
    protected Canvas previewCanvas;
    protected Image previewImage;
    protected Label nodeCountLabel;
    protected Combo zoomCombo;

    // Zoom support
    protected double zoomLevel = 1.0;
    protected static final int[] ZOOM_LEVELS = {25, 50, 75, 100, 125, 150, 200, 300, 400};

    // Selection state
    protected PipelineNode selectedNode = null;
    protected Set<PipelineNode> selectedNodes = new HashSet<>();
    protected Set<Connection> selectedConnections = new HashSet<>();

    // Dragging state
    protected Point dragOffset = null;
    protected boolean isDragging = false;

    // Connection drawing state
    protected PipelineNode connectionSource = null;
    protected int connectionSourceOutputIndex = 0;
    protected Point connectionEndPoint = null;

    // Yanked connection original target (to restore if dropped in empty space)
    protected PipelineNode yankedOriginalTarget = null;
    protected int yankedOriginalInputIndex = 0;

    // Selection box (marquee selection) state
    protected Point selectionBoxStart = null;
    protected Point selectionBoxEnd = null;
    protected boolean isSelectionBoxDragging = false;

    // ========== Abstract methods - subclasses must implement ==========

    /** Get the list of nodes to display/edit */
    protected abstract List<PipelineNode> getNodes();

    /** Get the list of connections to display/edit */
    protected abstract List<Connection> getConnections();

    /** Called when the canvas needs to be redrawn */
    protected abstract void redrawCanvas();

    /** Called when the diagram has been modified */
    protected abstract void notifyModified();

    /** Add a node of the given type at the specified position */
    protected abstract void addNodeAt(String nodeType, int x, int y);

    /** Delete the currently selected nodes and connections */
    protected abstract void deleteSelected();

    // ========== Coordinate conversion (zoom-aware) ==========

    protected int toCanvasX(int screenX) {
        return (int) (screenX / zoomLevel);
    }

    protected int toCanvasY(int screenY) {
        return (int) (screenY / zoomLevel);
    }

    protected Point toCanvasPoint(int screenX, int screenY) {
        return new Point(toCanvasX(screenX), toCanvasY(screenY));
    }

    // ========== Connection drawing ==========

    /**
     * Calculate bezier control points for a smooth S-curve from start to end.
     * Returns [cx1, cy1, cx2, cy2].
     */
    protected int[] calculateBezierControlPoints(Point start, Point end, PipelineNode sourceNode, PipelineNode targetNode) {
        int cx1, cy1, cx2, cy2;
        int dx = end.x - start.x;
        int dy = end.y - start.y;

        // If nearly horizontal (small Y difference), draw straight
        if (Math.abs(dy) < 10) {
            cx1 = start.x + dx / 3;
            cy1 = start.y;
            cx2 = end.x - dx / 3;
            cy2 = end.y;
        } else if (dx >= 0) {
            // Normal left-to-right: smooth S-curve
            int controlDist = Math.max(50, Math.abs(dx) / 2);
            cx1 = start.x + controlDist;
            cy1 = start.y;
            cx2 = end.x - controlDist;
            cy2 = end.y;
        } else {
            // Right-to-left (backwards): loop around
            int loopOut = Math.max(80, Math.abs(dx) / 2 + 40);
            cx1 = start.x + loopOut;
            cy1 = start.y;
            cx2 = end.x - loopOut;
            cy2 = end.y;
        }

        return new int[] { cx1, cy1, cx2, cy2 };
    }

    /**
     * Simplified bezier control points when we don't have node references.
     */
    protected int[] calculateBezierControlPoints(Point start, Point end) {
        return calculateBezierControlPoints(start, end, null, null);
    }

    /**
     * Calculate the midpoint of a cubic bezier curve at t=0.5.
     */
    protected Point getBezierMidpoint(Point start, Point end, int[] cp) {
        int cx1 = cp[0], cy1 = cp[1], cx2 = cp[2], cy2 = cp[3];

        double t = 0.5;
        double mt = 1 - t;
        double mt2 = mt * mt;
        double mt3 = mt2 * mt;
        double t2 = t * t;
        double t3 = t2 * t;

        int midX = (int) (mt3 * start.x + 3 * mt2 * t * cx1 + 3 * mt * t2 * cx2 + t3 * end.x);
        int midY = (int) (mt3 * start.y + 3 * mt2 * t * cy1 + 3 * mt * t2 * cy2 + t3 * end.y);

        return new Point(midX, midY);
    }

    /**
     * Draw an arrow head at the end point, pointing from the 'from' direction.
     */
    protected void drawArrow(GC gc, Point from, Point to) {
        double angle = Math.atan2(to.y - from.y, to.x - from.x);
        int arrowSize = 14;

        int x1 = (int) (to.x - arrowSize * Math.cos(angle - Math.PI / 6));
        int y1 = (int) (to.y - arrowSize * Math.sin(angle - Math.PI / 6));
        int x2 = (int) (to.x - arrowSize * Math.cos(angle + Math.PI / 6));
        int y2 = (int) (to.y - arrowSize * Math.sin(angle + Math.PI / 6));

        gc.drawLine(to.x, to.y, x1, y1);
        gc.drawLine(to.x, to.y, x2, y2);
    }

    /**
     * Draw a connection with bezier curve, arrow, and queue stats.
     */
    protected void drawConnection(GC gc, Connection conn, boolean isSelected) {
        Point start = conn.source.getOutputPoint(conn.outputIndex);
        Point end = getConnectionTargetPoint(conn);

        if (isSelected) {
            gc.setLineWidth(3);
            gc.setForeground(display.getSystemColor(SWT.COLOR_CYAN));
        } else {
            gc.setLineWidth(2);
            gc.setForeground(display.getSystemColor(SWT.COLOR_DARK_GRAY));
        }

        // Calculate bezier control points
        int[] cp = calculateBezierControlPoints(start, end, conn.source, conn.target);

        // Draw bezier curve
        Path path = new Path(display);
        path.moveTo(start.x, start.y);
        path.cubicTo(cp[0], cp[1], cp[2], cp[3], end.x, end.y);
        gc.drawPath(path);
        path.dispose();

        // Draw arrow from second control point direction
        drawArrow(gc, new Point(cp[2], cp[3]), end);

        // Draw queue size and total frames (blue box on connection)
        drawQueueStats(gc, conn, start, end, cp);
    }

    /**
     * Draw the queue statistics box on a connection.
     */
    protected void drawQueueStats(GC gc, Connection conn, Point start, Point end, int[] cp) {
        int queueSize = conn.getQueueSize();
        long totalFrames = conn.getTotalFramesSent();
        Point midPoint = getBezierMidpoint(start, end, cp);
        String sizeText = String.format("%,d / %,d", queueSize, totalFrames);

        gc.setForeground(display.getSystemColor(SWT.COLOR_WHITE));
        // Change background color based on queue backlog: blue for normal, dark red for backpressure
        if (queueSize >= 5) {
            Color backpressureColor = new Color(139, 0, 0); // Dark red
            gc.setBackground(backpressureColor);
            Point textExtent = gc.textExtent(sizeText);
            gc.fillRoundRectangle(midPoint.x - textExtent.x/2 - 3, midPoint.y - textExtent.y/2 - 2,
                textExtent.x + 6, textExtent.y + 4, 6, 6);
            gc.drawString(sizeText, midPoint.x - textExtent.x/2, midPoint.y - textExtent.y/2, true);
            backpressureColor.dispose();
        } else {
            gc.setBackground(display.getSystemColor(SWT.COLOR_DARK_BLUE));
            Point textExtent = gc.textExtent(sizeText);
            gc.fillRoundRectangle(midPoint.x - textExtent.x/2 - 3, midPoint.y - textExtent.y/2 - 2,
                textExtent.x + 6, textExtent.y + 4, 6, 6);
            gc.drawString(sizeText, midPoint.x - textExtent.x/2, midPoint.y - textExtent.y/2, true);
        }
    }

    /**
     * Get the correct input point for a connection based on its inputIndex.
     */
    protected Point getConnectionTargetPoint(Connection conn) {
        if (conn.inputIndex == 2 && conn.target.hasDualInput()) {
            return conn.target.getInputPoint2();
        }
        return conn.target.getInputPoint();
    }

    /**
     * Check if a point is near a connection line (for clicking on connections).
     */
    protected boolean isNearConnectionLine(Connection conn, Point click) {
        Point start = conn.source.getOutputPoint(conn.outputIndex);
        Point end = getConnectionTargetPoint(conn);
        int[] cp = calculateBezierControlPoints(start, end, conn.source, conn.target);

        // Sample points along the bezier curve and check distance
        for (double t = 0; t <= 1.0; t += 0.05) {
            double mt = 1 - t;
            double mt2 = mt * mt;
            double mt3 = mt2 * mt;
            double t2 = t * t;
            double t3 = t2 * t;

            int px = (int) (mt3 * start.x + 3 * mt2 * t * cp[0] + 3 * mt * t2 * cp[2] + t3 * end.x);
            int py = (int) (mt3 * start.y + 3 * mt2 * t * cp[1] + 3 * mt * t2 * cp[3] + t3 * end.y);

            double dist = Math.sqrt((click.x - px) * (click.x - px) + (click.y - py) * (click.y - py));
            if (dist < 8) {
                return true;
            }
        }
        return false;
    }

    // ========== Selection box drawing ==========

    /**
     * Draw the selection box (marquee) if currently dragging.
     */
    protected void drawSelectionBox(GC gc) {
        if (isSelectionBoxDragging && selectionBoxStart != null && selectionBoxEnd != null) {
            int boxX = Math.min(selectionBoxStart.x, selectionBoxEnd.x);
            int boxY = Math.min(selectionBoxStart.y, selectionBoxEnd.y);
            int boxWidth = Math.abs(selectionBoxEnd.x - selectionBoxStart.x);
            int boxHeight = Math.abs(selectionBoxEnd.y - selectionBoxStart.y);

            // Draw selection box with semi-transparent fill
            gc.setBackground(new Color(0, 120, 215));
            gc.setAlpha(30);
            gc.fillRectangle(boxX, boxY, boxWidth, boxHeight);
            gc.setAlpha(255);

            // Draw selection box border
            gc.setForeground(new Color(0, 120, 215));
            gc.setLineWidth(1);
            gc.drawRectangle(boxX, boxY, boxWidth, boxHeight);
        }
    }

    /**
     * Complete selection box and select nodes/connections within it.
     */
    protected void completeSelectionBox() {
        if (!isSelectionBoxDragging || selectionBoxStart == null || selectionBoxEnd == null) {
            return;
        }

        // Calculate box bounds
        int boxX = Math.min(selectionBoxStart.x, selectionBoxEnd.x);
        int boxY = Math.min(selectionBoxStart.y, selectionBoxEnd.y);
        int boxWidth = Math.abs(selectionBoxEnd.x - selectionBoxStart.x);
        int boxHeight = Math.abs(selectionBoxEnd.y - selectionBoxStart.y);

        // Select nodes that are completely inside the selection box
        for (PipelineNode node : getNodes()) {
            if (node.getX() >= boxX && node.getY() >= boxY &&
                node.getX() + node.getWidth() <= boxX + boxWidth &&
                node.getY() + node.getHeight() <= boxY + boxHeight) {
                selectedNodes.add(node);
            }
        }

        // Select connections that are completely inside selection box
        for (Connection conn : getConnections()) {
            Point start = conn.source.getOutputPoint(conn.outputIndex);
            Point end = getConnectionTargetPoint(conn);
            if (start.x >= boxX && start.x <= boxX + boxWidth &&
                start.y >= boxY && start.y <= boxY + boxHeight &&
                end.x >= boxX && end.x <= boxX + boxWidth &&
                end.y >= boxY && end.y <= boxY + boxHeight) {
                selectedConnections.add(conn);
            }
        }

        // Clear selection box state
        selectionBoxStart = null;
        selectionBoxEnd = null;
        isSelectionBoxDragging = false;
    }

    // ========== Node movement ==========

    /**
     * Move all selected nodes by arrow key direction.
     * Moves 1 canvas pixel (adjusted for zoom level).
     */
    protected void moveSelectedNodes(int keyCode) {
        if (selectedNodes.isEmpty()) {
            return;
        }

        int deltaX = 0;
        int deltaY = 0;
        int moveAmount = 1; // 1 canvas pixel per key press

        switch (keyCode) {
            case SWT.ARROW_UP:
                deltaY = -moveAmount;
                break;
            case SWT.ARROW_DOWN:
                deltaY = moveAmount;
                break;
            case SWT.ARROW_LEFT:
                deltaX = -moveAmount;
                break;
            case SWT.ARROW_RIGHT:
                deltaX = moveAmount;
                break;
        }

        for (PipelineNode node : selectedNodes) {
            node.setX(node.getX() + deltaX);
            node.setY(node.getY() + deltaY);
        }

        notifyModified();
        redrawCanvas();
    }

    // ========== Preview panel ==========

    /**
     * Update the preview panel with the given Mat image.
     */
    protected void updatePreview(Mat mat) {
        if (previewCanvas == null || previewCanvas.isDisposed()) {
            return;
        }

        display.asyncExec(() -> {
            if (previewCanvas.isDisposed()) return;

            // Dispose old preview image
            if (previewImage != null && !previewImage.isDisposed()) {
                previewImage.dispose();
            }

            if (mat == null || mat.empty()) {
                previewImage = null;
                previewCanvas.redraw();
                return;
            }

            try {
                // Scale to fit preview canvas
                Rectangle bounds = previewCanvas.getBounds();
                int maxWidth = bounds.width > 0 ? bounds.width : 200;
                int maxHeight = bounds.height > 0 ? bounds.height : 150;

                double scale = Math.min((double) maxWidth / mat.width(),
                                        (double) maxHeight / mat.height());
                Mat resized = new Mat();
                Imgproc.resize(mat, resized, new Size(mat.width() * scale, mat.height() * scale));

                // Convert to RGB if needed
                Mat rgb = new Mat();
                if (resized.channels() == 1) {
                    Imgproc.cvtColor(resized, rgb, Imgproc.COLOR_GRAY2RGB);
                } else if (resized.channels() == 3) {
                    Imgproc.cvtColor(resized, rgb, Imgproc.COLOR_BGR2RGB);
                } else {
                    rgb = resized.clone();
                }

                int w = rgb.width();
                int h = rgb.height();
                byte[] data = new byte[w * h * 3];
                rgb.get(0, 0, data);

                PaletteData palette = new PaletteData(0xFF0000, 0x00FF00, 0x0000FF);
                ImageData imageData = new ImageData(w, h, 24, palette);

                int bytesPerLine = imageData.bytesPerLine;
                for (int row = 0; row < h; row++) {
                    int srcOffset = row * w * 3;
                    int dstOffset = row * bytesPerLine;
                    for (int col = 0; col < w; col++) {
                        int srcIdx = srcOffset + col * 3;
                        int dstIdx = dstOffset + col * 3;
                        imageData.data[dstIdx] = data[srcIdx];
                        imageData.data[dstIdx + 1] = data[srcIdx + 1];
                        imageData.data[dstIdx + 2] = data[srcIdx + 2];
                    }
                }

                previewImage = new Image(display, imageData);
                previewCanvas.redraw();

                resized.release();
                rgb.release();
            } catch (Exception e) {
                System.err.println("Error updating preview: " + e.getMessage());
            }
        });
    }

    /**
     * Update preview from the currently selected node's output.
     */
    protected void updatePreviewFromNode(PipelineNode node) {
        if (node == null) {
            updatePreview(null);
            return;
        }
        Mat output = node.getOutputMat();
        if (output != null && !output.empty()) {
            updatePreview(output.clone());
        }
    }

    // ========== Zoom support ==========

    /**
     * Set the zoom level and update the canvas.
     */
    protected void setZoomLevel(double newZoom) {
        this.zoomLevel = newZoom;
        updateCanvasSize();
        redrawCanvas();
    }

    /**
     * Change zoom level while keeping the viewport centered on the same canvas point.
     * This calculates the canvas point at the center of the viewport before zoom,
     * then adjusts scroll position after zoom so that same point stays centered.
     *
     * @param newZoom The new zoom level (e.g., 1.0 for 100%, 2.0 for 200%)
     */
    protected void setZoomLevelCentered(double newZoom) {
        if (scrolledCanvas == null || canvas == null) {
            setZoomLevel(newZoom);
            return;
        }

        double oldZoom = zoomLevel;
        if (Math.abs(oldZoom - newZoom) < 0.001) {
            return; // No change
        }

        // Get current scroll position and viewport size
        org.eclipse.swt.graphics.Point origin = scrolledCanvas.getOrigin();
        org.eclipse.swt.graphics.Rectangle viewport = scrolledCanvas.getClientArea();

        // Calculate the canvas coordinate at the center of the viewport (before zoom)
        double centerCanvasX = (origin.x + viewport.width / 2.0) / oldZoom;
        double centerCanvasY = (origin.y + viewport.height / 2.0) / oldZoom;

        // Apply the new zoom level
        zoomLevel = newZoom;
        updateCanvasSize();

        // Calculate new scroll position to keep the same canvas point centered
        int newOriginX = (int) (centerCanvasX * newZoom - viewport.width / 2.0);
        int newOriginY = (int) (centerCanvasY * newZoom - viewport.height / 2.0);

        // Clamp to valid scroll range
        org.eclipse.swt.graphics.Point canvasSize = canvas.getSize();
        newOriginX = Math.max(0, Math.min(newOriginX, canvasSize.x - viewport.width));
        newOriginY = Math.max(0, Math.min(newOriginY, canvasSize.y - viewport.height));

        // Set the new scroll position
        scrolledCanvas.setOrigin(newOriginX, newOriginY);

        redrawCanvas();
    }

    /**
     * Update the canvas size based on zoom level and content bounds.
     */
    protected void updateCanvasSize() {
        int baseWidth = 1200;
        int baseHeight = 800;

        // Find the bounds of all nodes
        int maxX = baseWidth;
        int maxY = baseHeight;
        for (PipelineNode node : getNodes()) {
            maxX = Math.max(maxX, node.getX() + node.getWidth() + 100);
            maxY = Math.max(maxY, node.getY() + node.getHeight() + 100);
        }

        // Apply zoom
        int scaledWidth = (int) (maxX * zoomLevel);
        int scaledHeight = (int) (maxY * zoomLevel);

        canvas.setSize(scaledWidth, scaledHeight);
        if (scrolledCanvas != null) {
            scrolledCanvas.setMinSize(scaledWidth, scaledHeight);
        }
    }

    /**
     * Update the node count label.
     */
    protected void updateNodeCount() {
        if (nodeCountLabel != null && !nodeCountLabel.isDisposed()) {
            int nodeCount = getNodes().size();
            int connectionCount = getConnections().size();
            nodeCountLabel.setText(String.format("Nodes: %d  Connections: %d", nodeCount, connectionCount));
        }
    }

    // ========== Connection creation helpers ==========

    /**
     * Find a connection targeting a specific node and input index.
     */
    protected Connection findConnectionToTarget(PipelineNode target, int inputIndex) {
        for (Connection conn : getConnections()) {
            if (conn.target == target && conn.inputIndex == inputIndex) {
                return conn;
            }
        }
        return null;
    }

    /**
     * Check if a target input is already connected.
     */
    protected boolean isInputConnected(PipelineNode target, int inputIndex) {
        return findConnectionToTarget(target, inputIndex) != null;
    }

    /**
     * Try to complete a connection to the given target node at the specified input.
     * Returns true if connection was made, false otherwise.
     */
    protected boolean tryCompleteConnection(PipelineNode target, int inputIndex) {
        if (connectionSource == null || target == null) {
            return false;
        }

        // Don't connect to self
        if (connectionSource == target) {
            return false;
        }

        // Check if target input is already connected
        if (isInputConnected(target, inputIndex)) {
            return false;
        }

        // Create the connection
        Connection conn = new Connection(connectionSource, target, inputIndex, connectionSourceOutputIndex);
        getConnections().add(conn);

        // Clear connection state
        connectionSource = null;
        connectionSourceOutputIndex = 0;
        connectionEndPoint = null;
        yankedOriginalTarget = null;
        yankedOriginalInputIndex = 0;

        notifyModified();
        return true;
    }

    /**
     * Cancel the current connection being drawn.
     * If it was a yanked connection, restore it.
     */
    protected void cancelConnection() {
        if (yankedOriginalTarget != null && connectionSource != null) {
            // Restore the yanked connection
            Connection restored = new Connection(connectionSource, yankedOriginalTarget,
                                                  yankedOriginalInputIndex, connectionSourceOutputIndex);
            getConnections().add(restored);
        }

        connectionSource = null;
        connectionSourceOutputIndex = 0;
        connectionEndPoint = null;
        yankedOriginalTarget = null;
        yankedOriginalInputIndex = 0;
    }

    // ========== Grid drawing ==========

    /**
     * Draw a background grid that scales with zoom.
     */
    protected void drawGrid(GC gc, int width, int height) {
        gc.setForeground(new Color(240, 240, 240));
        gc.setLineWidth(1);

        int gridSize = 20; // Base grid size in canvas coordinates

        // Draw vertical lines
        for (int x = 0; x < width; x += gridSize) {
            gc.drawLine(x, 0, x, height);
        }

        // Draw horizontal lines
        for (int y = 0; y < height; y += gridSize) {
            gc.drawLine(0, y, width, y);
        }
    }
}
