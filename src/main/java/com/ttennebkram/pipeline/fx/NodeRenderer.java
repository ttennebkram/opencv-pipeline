package com.ttennebkram.pipeline.fx;

import javafx.scene.canvas.GraphicsContext;
import javafx.scene.image.Image;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;

/**
 * Utility class for rendering pipeline nodes on a JavaFX Canvas.
 * This provides a clean separation between node data and rendering,
 * making the SWT to JavaFX migration easier.
 */
public class NodeRenderer {

    // Node dimension constants (matching SWT version)
    public static final int PROCESSING_NODE_THUMB_WIDTH = 120;
    public static final int PROCESSING_NODE_THUMB_HEIGHT = 80;
    public static final int SOURCE_NODE_THUMB_WIDTH = 280;
    public static final int SOURCE_NODE_THUMB_HEIGHT = 90;

    public static final int NODE_WIDTH = PROCESSING_NODE_THUMB_WIDTH + 60;
    public static final int NODE_HEIGHT = PROCESSING_NODE_THUMB_HEIGHT + 40;
    public static final int SOURCE_NODE_HEIGHT = SOURCE_NODE_THUMB_HEIGHT + 32;

    // Checkbox constants
    private static final int CHECKBOX_SIZE = 12;
    private static final int CHECKBOX_MARGIN = 5;

    // Help icon constants
    private static final int HELP_ICON_SIZE = 14;
    private static final int HELP_ICON_MARGIN = 5;

    // Connection point constants
    public static final int CONNECTION_RADIUS = 6;

    // Colors
    private static final Color COLOR_NODE_BG = Color.rgb(200, 220, 255);
    private static final Color COLOR_NODE_BORDER = Color.rgb(100, 100, 150);
    private static final Color COLOR_NODE_SELECTED = Color.rgb(0, 120, 215);
    private static final Color COLOR_NODE_DISABLED_BG = Color.rgb(220, 220, 220);
    private static final Color COLOR_INPUT_POINT = Color.rgb(100, 150, 100);
    private static final Color COLOR_OUTPUT_POINT = Color.rgb(150, 100, 100);
    private static final Color COLOR_HELP_ICON = Color.rgb(50, 100, 180);
    private static final Color COLOR_CHECKBOX_CHECK = Color.rgb(0, 150, 0);

    /**
     * Render a processing node at the given position.
     *
     * @param gc           Graphics context to draw on
     * @param x            X position
     * @param y            Y position
     * @param width        Node width
     * @param height       Node height
     * @param label        Node display label
     * @param selected     Whether the node is selected
     * @param enabled      Whether the node is enabled
     * @param bgColor      Background color (or null for default)
     * @param hasInput     Whether to draw input connection point
     * @param hasDualInput Whether to draw second input connection point
     * @param outputCount  Number of output connection points
     */
    public static void renderNode(GraphicsContext gc, double x, double y,
                                   double width, double height, String label,
                                   boolean selected, boolean enabled,
                                   Color bgColor, boolean hasInput,
                                   boolean hasDualInput, int outputCount) {
        renderNode(gc, x, y, width, height, label, selected, enabled,
                   bgColor, hasInput, hasDualInput, outputCount, null);
    }

    /**
     * Render a processing node with a thumbnail image.
     */
    public static void renderNode(GraphicsContext gc, double x, double y,
                                   double width, double height, String label,
                                   boolean selected, boolean enabled,
                                   Color bgColor, boolean hasInput,
                                   boolean hasDualInput, int outputCount,
                                   Image thumbnail) {

        // Draw background
        Color bg = enabled ? (bgColor != null ? bgColor : COLOR_NODE_BG) : COLOR_NODE_DISABLED_BG;
        gc.setFill(bg);
        gc.fillRoundRect(x, y, width, height, 10, 10);

        // Draw border (thicker if selected)
        gc.setStroke(selected ? COLOR_NODE_SELECTED : COLOR_NODE_BORDER);
        gc.setLineWidth(selected ? 3 : 2);
        gc.strokeRoundRect(x, y, width, height, 10, 10);

        // Draw enabled checkbox
        drawCheckbox(gc, x + CHECKBOX_MARGIN, y + CHECKBOX_MARGIN, enabled);

        // Draw title
        gc.setFill(Color.BLACK);
        gc.setFont(Font.font("Arial", FontWeight.BOLD, 10));
        gc.fillText(label, x + CHECKBOX_MARGIN + CHECKBOX_SIZE + 5, y + 15);

        // Draw help icon
        drawHelpIcon(gc, x + width - HELP_ICON_SIZE - HELP_ICON_MARGIN,
                     y + HELP_ICON_MARGIN);

        // Draw thumbnail or placeholder
        double thumbX = x + 40;
        double thumbY = y + 35;

        if (thumbnail != null) {
            // Draw the actual thumbnail image
            double thumbW = Math.min(thumbnail.getWidth(), PROCESSING_NODE_THUMB_WIDTH);
            double thumbH = Math.min(thumbnail.getHeight(), PROCESSING_NODE_THUMB_HEIGHT);
            // Center the thumbnail if smaller than max size
            double drawX = thumbX + (PROCESSING_NODE_THUMB_WIDTH - thumbW) / 2;
            double drawY = thumbY + (PROCESSING_NODE_THUMB_HEIGHT - thumbH) / 2;
            gc.drawImage(thumbnail, drawX, drawY, thumbW, thumbH);
        } else {
            // Draw placeholder
            gc.setStroke(Color.LIGHTGRAY);
            gc.setLineWidth(1);
            gc.strokeRect(thumbX, thumbY, PROCESSING_NODE_THUMB_WIDTH, PROCESSING_NODE_THUMB_HEIGHT);
            gc.setFill(Color.GRAY);
            gc.setFont(Font.font("Arial", 9));
            gc.fillText("(no output)", thumbX + 30, thumbY + 45);
        }

        // Draw input connection point(s)
        if (hasInput) {
            drawConnectionPoint(gc, x, y + height / 2, true);
            if (hasDualInput) {
                drawConnectionPoint(gc, x, y + height / 2 + 20, true);
            }
        }

        // Draw output connection point(s)
        double outputSpacing = height / (outputCount + 1);
        for (int i = 0; i < outputCount; i++) {
            drawConnectionPoint(gc, x + width, y + outputSpacing * (i + 1), false);
        }
    }

    /**
     * Render a source node (wider, no input point).
     */
    public static void renderSourceNode(GraphicsContext gc, double x, double y,
                                         double width, double height, String label,
                                         boolean selected, boolean enabled,
                                         Color bgColor) {
        // Source nodes are similar but wider and have no input
        renderNode(gc, x, y, width, height, label, selected, enabled, bgColor,
                   false, false, 1);
    }

    /**
     * Draw an enabled checkbox.
     */
    private static void drawCheckbox(GraphicsContext gc, double x, double y, boolean checked) {
        // Draw checkbox border
        gc.setStroke(Color.DARKGRAY);
        gc.setLineWidth(1);
        gc.strokeRect(x, y, CHECKBOX_SIZE, CHECKBOX_SIZE);

        // Fill if checked
        if (checked) {
            gc.setFill(COLOR_CHECKBOX_CHECK);
            // Draw checkmark
            gc.setStroke(Color.WHITE);
            gc.setLineWidth(2);
            gc.strokeLine(x + 2, y + CHECKBOX_SIZE / 2,
                         x + CHECKBOX_SIZE / 2 - 1, y + CHECKBOX_SIZE - 3);
            gc.strokeLine(x + CHECKBOX_SIZE / 2 - 1, y + CHECKBOX_SIZE - 3,
                         x + CHECKBOX_SIZE - 2, y + 2);
        }
    }

    /**
     * Draw a help icon (question mark in circle).
     */
    private static void drawHelpIcon(GraphicsContext gc, double x, double y) {
        gc.setFill(COLOR_HELP_ICON);
        gc.fillOval(x, y, HELP_ICON_SIZE, HELP_ICON_SIZE);
        gc.setFill(Color.WHITE);
        gc.setFont(Font.font("Arial", FontWeight.BOLD, 10));
        gc.fillText("?", x + 4, y + 11);
    }

    /**
     * Draw a connection point (input or output).
     */
    private static void drawConnectionPoint(GraphicsContext gc, double x, double y, boolean isInput) {
        gc.setFill(isInput ? COLOR_INPUT_POINT : COLOR_OUTPUT_POINT);
        gc.fillOval(x - CONNECTION_RADIUS, y - CONNECTION_RADIUS,
                    CONNECTION_RADIUS * 2, CONNECTION_RADIUS * 2);
        gc.setStroke(Color.BLACK);
        gc.setLineWidth(1);
        gc.strokeOval(x - CONNECTION_RADIUS, y - CONNECTION_RADIUS,
                      CONNECTION_RADIUS * 2, CONNECTION_RADIUS * 2);
    }

    /**
     * Render a connection line between two points.
     */
    public static void renderConnection(GraphicsContext gc,
                                         double startX, double startY,
                                         double endX, double endY,
                                         boolean selected) {
        gc.setStroke(selected ? COLOR_NODE_SELECTED : Color.rgb(80, 80, 80));
        gc.setLineWidth(selected ? 3 : 2);

        // Draw bezier curve
        double ctrlOffset = Math.abs(endX - startX) / 2;
        gc.beginPath();
        gc.moveTo(startX, startY);
        gc.bezierCurveTo(
            startX + ctrlOffset, startY,
            endX - ctrlOffset, endY,
            endX, endY
        );
        gc.stroke();

        // Draw arrow at end
        drawArrowHead(gc, endX - ctrlOffset / 2, endY, endX, endY);
    }

    /**
     * Draw an arrow head at the end of a connection.
     */
    private static void drawArrowHead(GraphicsContext gc, double fromX, double fromY,
                                       double toX, double toY) {
        double angle = Math.atan2(toY - fromY, toX - fromX);
        double arrowLength = 10;
        double arrowAngle = Math.PI / 6;

        double x1 = toX - arrowLength * Math.cos(angle - arrowAngle);
        double y1 = toY - arrowLength * Math.sin(angle - arrowAngle);
        double x2 = toX - arrowLength * Math.cos(angle + arrowAngle);
        double y2 = toY - arrowLength * Math.sin(angle + arrowAngle);

        gc.fillPolygon(new double[]{toX, x1, x2}, new double[]{toY, y1, y2}, 3);
    }

    /**
     * Get the input connection point position for a node.
     */
    public static double[] getInputPoint(double nodeX, double nodeY, double nodeHeight, int inputIndex) {
        double y = nodeY + nodeHeight / 2;
        if (inputIndex == 1) {
            y += 20; // Second input offset
        }
        return new double[]{nodeX, y};
    }

    /**
     * Get the output connection point position for a node.
     */
    public static double[] getOutputPoint(double nodeX, double nodeY,
                                           double nodeWidth, double nodeHeight,
                                           int outputIndex, int totalOutputs) {
        double spacing = nodeHeight / (totalOutputs + 1);
        return new double[]{nodeX + nodeWidth, nodeY + spacing * (outputIndex + 1)};
    }
}
