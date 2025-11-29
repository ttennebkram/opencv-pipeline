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
    public static final int PROCESSING_NODE_THUMB_WIDTH = 100;
    public static final int PROCESSING_NODE_THUMB_HEIGHT = 70;
    public static final int SOURCE_NODE_THUMB_WIDTH = 100;  // Same as processing nodes
    public static final int SOURCE_NODE_THUMB_HEIGHT = 70;

    public static final int NODE_WIDTH = PROCESSING_NODE_THUMB_WIDTH + 80;  // Extra space for In:/Out: labels
    public static final int NODE_HEIGHT = PROCESSING_NODE_THUMB_HEIGHT + 50;
    public static final int SOURCE_NODE_HEIGHT = SOURCE_NODE_THUMB_HEIGHT + 42;

    // Checkbox constants
    private static final int CHECKBOX_SIZE = 12;
    private static final int CHECKBOX_MARGIN = 5;

    // Help icon constants
    private static final int HELP_ICON_SIZE = 14;
    private static final int HELP_ICON_MARGIN = 5;

    // Connection point constants
    public static final int CONNECTION_RADIUS = 6;

    // Line width constants
    private static final int LINE_WIDTH_NORMAL = 2;
    private static final int LINE_WIDTH_SELECTED = 3;

    // Colors
    private static final Color COLOR_NODE_BG = Color.rgb(200, 220, 255);
    private static final Color COLOR_NODE_BORDER = Color.rgb(100, 100, 150);
    private static final Color COLOR_NODE_SELECTED = Color.rgb(0, 0, 255);  // Bright blue for maximum visibility
    private static final Color COLOR_NODE_DISABLED_BG = Color.rgb(220, 220, 220);

    // Public color for container and boundary nodes (shared across classes)
    public static final Color COLOR_CONTAINER_NODE = Color.rgb(255, 200, 200);  // Light red

    // Shared instructions text for both Pipeline Editor and Container Editor
    public static final String INSTRUCTIONS_TEXT =
        "- Click node name to create\n" +
        "- Drag nodes to move\n" +
        "- Click and drag circles to connect\n" +
        "- Double-click for properties\n" +
        "- Click node to see preview\n" +
        "- Edit while running for live updates";
    private static final Color COLOR_INPUT_POINT = Color.rgb(100, 150, 100);
    private static final Color COLOR_OUTPUT_POINT = Color.rgb(150, 100, 100);
    private static final Color COLOR_HELP_ICON = Color.rgb(50, 100, 180);
    private static final Color COLOR_CHECKBOX_CHECK = Color.rgb(0, 150, 0);
    private static final Color COLOR_QUEUE_BG_NORMAL = Color.rgb(0, 0, 139);  // Dark blue
    private static final Color COLOR_QUEUE_BG_BACKPRESSURE = Color.rgb(139, 0, 0);  // Dark red
    private static final Color COLOR_QUEUE_TEXT = Color.WHITE;
    private static final Font QUEUE_STATS_FONT = Font.font("System", FontWeight.NORMAL, 10);
    private static final Font STATS_LINE_FONT = Font.font("Arial", FontWeight.BOLD, 9);
    private static final Color COLOR_STATS_NORMAL = Color.rgb(40, 40, 40);  // Near-black for visibility
    private static final Color COLOR_STATS_SLOWED = Color.rgb(180, 0, 0);  // Red for priority < 5

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
                   bgColor, hasInput, hasDualInput, outputCount, null, false);
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
        renderNode(gc, x, y, width, height, label, selected, enabled,
                   bgColor, hasInput, hasDualInput, outputCount, thumbnail, false);
    }

    /**
     * Render a processing node with a thumbnail image and container flag.
     */
    public static void renderNode(GraphicsContext gc, double x, double y,
                                   double width, double height, String label,
                                   boolean selected, boolean enabled,
                                   Color bgColor, boolean hasInput,
                                   boolean hasDualInput, int outputCount,
                                   Image thumbnail, boolean isContainer) {
        renderNode(gc, x, y, width, height, label, selected, enabled, bgColor,
                   hasInput, hasDualInput, outputCount, thumbnail, isContainer, null);
    }

    /**
     * Render a processing node with a thumbnail image, container flag, and node type for help lookup.
     */
    public static void renderNode(GraphicsContext gc, double x, double y,
                                   double width, double height, String label,
                                   boolean selected, boolean enabled,
                                   Color bgColor, boolean hasInput,
                                   boolean hasDualInput, int outputCount,
                                   Image thumbnail, boolean isContainer, String nodeType) {
        renderNode(gc, x, y, width, height, label, selected, enabled, bgColor,
                   hasInput, hasDualInput, outputCount, thumbnail, isContainer, nodeType, false);
    }

    /**
     * Render a processing node with all options including boundary node indicator.
     */
    public static void renderNode(GraphicsContext gc, double x, double y,
                                   double width, double height, String label,
                                   boolean selected, boolean enabled,
                                   Color bgColor, boolean hasInput,
                                   boolean hasDualInput, int outputCount,
                                   Image thumbnail, boolean isContainer, String nodeType,
                                   boolean isBoundaryNode) {

        // Draw background
        Color bg = enabled ? (bgColor != null ? bgColor : COLOR_NODE_BG) : COLOR_NODE_DISABLED_BG;
        gc.setFill(bg);
        gc.fillRoundRect(x, y, width, height, 10, 10);

        // Draw border (thicker if selected, double-line for containers)
        gc.setStroke(selected ? COLOR_NODE_SELECTED : COLOR_NODE_BORDER);
        gc.setLineWidth(selected ? LINE_WIDTH_SELECTED : LINE_WIDTH_NORMAL);
        gc.strokeRoundRect(x, y, width, height, 10, 10);

        // Draw inner border for container nodes (double-line effect)
        if (isContainer) {
            gc.strokeRoundRect(x + 4, y + 4, width - 8, height - 8, 6, 6);
        }

        // Draw 3 vertical "bolt" dots for boundary nodes (indicates "fixed" node)
        // Input node: dots on left (where it attaches to container edge)
        // Output node: dots on right (where it attaches to container edge)
        if (isBoundaryNode) {
            gc.setFill(Color.rgb(80, 90, 100));
            double dotRadius = 4;
            double dotX;
            if ("ContainerInput".equals(nodeType)) {
                // Input node: bolts on left side
                dotX = x + 8;
            } else {
                // Output node: bolts on right side
                dotX = x + width - 8;
            }
            double dotY1 = y + 30;
            double dotY2 = y + height / 2;
            double dotY3 = y + height - 30;
            gc.fillOval(dotX - dotRadius, dotY1 - dotRadius, dotRadius * 2, dotRadius * 2);
            gc.fillOval(dotX - dotRadius, dotY2 - dotRadius, dotRadius * 2, dotRadius * 2);
            gc.fillOval(dotX - dotRadius, dotY3 - dotRadius, dotRadius * 2, dotRadius * 2);
        }

        // Draw enabled checkbox (not for boundary nodes or Monitor nodes - they can't/shouldn't be disabled)
        boolean skipCheckbox = isBoundaryNode || "Monitor".equals(nodeType);
        if (!skipCheckbox) {
            drawCheckbox(gc, x + CHECKBOX_MARGIN, y + CHECKBOX_MARGIN, enabled);
        }

        // Draw title (shift left if no checkbox)
        gc.setFill(Color.BLACK);
        gc.setFont(Font.font("Arial", FontWeight.BOLD, 10));
        double titleX = skipCheckbox ? x + CHECKBOX_MARGIN : x + CHECKBOX_MARGIN + CHECKBOX_SIZE + 5;
        gc.fillText(label, titleX, y + 15);

        // Draw help icon (grayed out if no help available)
        boolean hasHelp = nodeType != null && FXHelpBrowser.hasHelp(nodeType);
        drawHelpIcon(gc, x + width - HELP_ICON_SIZE - HELP_ICON_MARGIN,
                     y + HELP_ICON_MARGIN, hasHelp);

        // Draw thumbnail or placeholder
        // Determine if this is a source node (no input) and use appropriate thumbnail size
        int thumbMaxW = hasInput ? PROCESSING_NODE_THUMB_WIDTH : SOURCE_NODE_THUMB_WIDTH;
        int thumbMaxH = hasInput ? PROCESSING_NODE_THUMB_HEIGHT : SOURCE_NODE_THUMB_HEIGHT;
        double thumbX = hasInput ? x + 40 : x + (width - thumbMaxW) / 2;
        double thumbY = y + 32;

        if (thumbnail != null) {
            // Draw the actual thumbnail image, scaling to fit
            double scale = Math.min(
                (double) thumbMaxW / thumbnail.getWidth(),
                (double) thumbMaxH / thumbnail.getHeight()
            );
            double thumbW = thumbnail.getWidth() * scale;
            double thumbH = thumbnail.getHeight() * scale;
            // Center the thumbnail
            double drawX = thumbX + (thumbMaxW - thumbW) / 2;
            double drawY = thumbY + (thumbMaxH - thumbH) / 2;

            // Draw scaled thumbnail
            gc.drawImage(thumbnail, drawX, drawY, thumbW, thumbH);
        } else {
            // Draw placeholder
            gc.setStroke(Color.LIGHTGRAY);
            gc.setLineWidth(1);
            gc.strokeRect(thumbX, thumbY, thumbMaxW, thumbMaxH);
            gc.setFill(Color.GRAY);
            gc.setFont(Font.font("Arial", 9));
            gc.fillText("(no output)", thumbX + thumbMaxW / 2 - 25, thumbY + thumbMaxH / 2 + 5);
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
     * Render a processing node with counters (legacy without nodeType).
     */
    public static void renderNode(GraphicsContext gc, double x, double y,
                                   double width, double height, String label,
                                   boolean selected, boolean enabled,
                                   Color bgColor, boolean hasInput,
                                   boolean hasDualInput, int outputCount,
                                   Image thumbnail, boolean isContainer,
                                   int inputCounter, int[] outputCounters) {
        renderNode(gc, x, y, width, height, label, selected, enabled, bgColor,
                   hasInput, hasDualInput, outputCount, thumbnail, isContainer,
                   inputCounter, 0, outputCounters, null);
    }

    /**
     * Render a processing node with counters and node type for help lookup.
     */
    public static void renderNode(GraphicsContext gc, double x, double y,
                                   double width, double height, String label,
                                   boolean selected, boolean enabled,
                                   Color bgColor, boolean hasInput,
                                   boolean hasDualInput, int outputCount,
                                   Image thumbnail, boolean isContainer,
                                   int inputCounter, int[] outputCounters, String nodeType) {
        renderNode(gc, x, y, width, height, label, selected, enabled, bgColor,
                   hasInput, hasDualInput, outputCount, thumbnail, isContainer,
                   inputCounter, 0, outputCounters, nodeType, false);
    }

    /**
     * Render a processing node with counters including input2Counter for dual-input nodes.
     */
    public static void renderNode(GraphicsContext gc, double x, double y,
                                   double width, double height, String label,
                                   boolean selected, boolean enabled,
                                   Color bgColor, boolean hasInput,
                                   boolean hasDualInput, int outputCount,
                                   Image thumbnail, boolean isContainer,
                                   int inputCounter, int inputCounter2, int[] outputCounters, String nodeType) {
        renderNode(gc, x, y, width, height, label, selected, enabled, bgColor,
                   hasInput, hasDualInput, outputCount, thumbnail, isContainer,
                   inputCounter, inputCounter2, outputCounters, nodeType, false);
    }

    /**
     * Render a processing node with counters, node type, and boundary node indicator.
     * This is the full version with all parameters.
     */
    public static void renderNode(GraphicsContext gc, double x, double y,
                                   double width, double height, String label,
                                   boolean selected, boolean enabled,
                                   Color bgColor, boolean hasInput,
                                   boolean hasDualInput, int outputCount,
                                   Image thumbnail, boolean isContainer,
                                   int inputCounter, int inputCounter2, int[] outputCounters, String nodeType,
                                   boolean isBoundaryNode) {

        // Draw background
        Color bg = enabled ? (bgColor != null ? bgColor : COLOR_NODE_BG) : COLOR_NODE_DISABLED_BG;
        gc.setFill(bg);
        gc.fillRoundRect(x, y, width, height, 10, 10);

        // Draw border (thicker if selected, double-line for containers)
        gc.setStroke(selected ? COLOR_NODE_SELECTED : COLOR_NODE_BORDER);
        gc.setLineWidth(selected ? LINE_WIDTH_SELECTED : LINE_WIDTH_NORMAL);
        gc.strokeRoundRect(x, y, width, height, 10, 10);

        // Draw inner border for container nodes (double-line effect)
        if (isContainer) {
            gc.strokeRoundRect(x + 4, y + 4, width - 8, height - 8, 6, 6);
        }

        // Draw 3 vertical "bolt" dots for boundary nodes (indicates "fixed" node)
        // Input node: dots on left (where it attaches to container edge)
        // Output node: dots on right (where it attaches to container edge)
        if (isBoundaryNode) {
            gc.setFill(Color.rgb(80, 90, 100));
            double dotRadius = 4;
            double dotX;
            if ("ContainerInput".equals(nodeType)) {
                // Input node: bolts on left side
                dotX = x + 8;
            } else {
                // Output node: bolts on right side
                dotX = x + width - 8;
            }
            double dotY1 = y + 30;
            double dotY2 = y + height / 2;
            double dotY3 = y + height - 30;
            gc.fillOval(dotX - dotRadius, dotY1 - dotRadius, dotRadius * 2, dotRadius * 2);
            gc.fillOval(dotX - dotRadius, dotY2 - dotRadius, dotRadius * 2, dotRadius * 2);
            gc.fillOval(dotX - dotRadius, dotY3 - dotRadius, dotRadius * 2, dotRadius * 2);
        }

        // Draw enabled checkbox (not for boundary nodes or Monitor nodes - they can't/shouldn't be disabled)
        boolean skipCheckbox = isBoundaryNode || "Monitor".equals(nodeType);
        if (!skipCheckbox) {
            drawCheckbox(gc, x + CHECKBOX_MARGIN, y + CHECKBOX_MARGIN, enabled);
        }

        // Draw title (shift left if no checkbox)
        gc.setFill(Color.BLACK);
        gc.setFont(Font.font("Arial", FontWeight.BOLD, 10));
        double titleX = skipCheckbox ? x + CHECKBOX_MARGIN : x + CHECKBOX_MARGIN + CHECKBOX_SIZE + 5;
        gc.fillText(label, titleX, y + 15);

        // Draw help icon (grayed out if no help available)
        boolean hasHelp = nodeType != null && FXHelpBrowser.hasHelp(nodeType);
        drawHelpIcon(gc, x + width - HELP_ICON_SIZE - HELP_ICON_MARGIN,
                     y + HELP_ICON_MARGIN, hasHelp);

        // Draw thumbnail or placeholder - centered in node, leaving room for In/Out labels
        int thumbMaxW = hasInput ? PROCESSING_NODE_THUMB_WIDTH : SOURCE_NODE_THUMB_WIDTH;
        int thumbMaxH = hasInput ? PROCESSING_NODE_THUMB_HEIGHT : SOURCE_NODE_THUMB_HEIGHT;
        // Center thumbnail horizontally, offset from edges to leave room for labels
        double thumbX = x + (width - thumbMaxW) / 2;
        double thumbY = y + 28;

        if (thumbnail != null) {
            double scale = Math.min(
                (double) thumbMaxW / thumbnail.getWidth(),
                (double) thumbMaxH / thumbnail.getHeight()
            );
            double thumbW = thumbnail.getWidth() * scale;
            double thumbH = thumbnail.getHeight() * scale;
            double drawX = thumbX + (thumbMaxW - thumbW) / 2;
            double drawY = thumbY + (thumbMaxH - thumbH) / 2;
            gc.drawImage(thumbnail, drawX, drawY, thumbW, thumbH);
        } else {
            gc.setStroke(Color.LIGHTGRAY);
            gc.setLineWidth(1);
            gc.strokeRect(thumbX, thumbY, thumbMaxW, thumbMaxH);
            gc.setFill(Color.GRAY);
            gc.setFont(Font.font("Arial", 9));
            gc.fillText("(no output)", thumbX + thumbMaxW / 2 - 25, thumbY + thumbMaxH / 2 + 5);
        }

        // Draw input connection point(s) with counter
        gc.setFont(Font.font("Arial", FontWeight.BOLD, 9));
        if (hasInput) {
            double inputY = y + height / 2;
            drawConnectionPoint(gc, x, inputY, true);
            // Draw input counter label - dark color for visibility
            gc.setFill(Color.rgb(60, 60, 60));
            // For dual-input nodes, label as "In1:" instead of "In:"
            String inLabel = hasDualInput ? "In1:" + inputCounter : "In:" + inputCounter;
            gc.fillText(inLabel, x + 10, inputY + 4);
            if (hasDualInput) {
                drawConnectionPoint(gc, x, inputY + 20, true);
                // Draw second input counter label - reset fill color after drawing connection point
                gc.setFill(Color.rgb(60, 60, 60));
                gc.fillText("In2:" + inputCounter2, x + 10, inputY + 24);
            }
        }

        // For ContainerInput boundary nodes: show "In:" on left side (frames entering container)
        // ContainerInput has no hasInput but receives data from outside the container
        if (isBoundaryNode && "ContainerInput".equals(nodeType)) {
            gc.setFill(Color.rgb(60, 60, 60));
            double inputY = y + height / 2;
            // Use outputCounter[0] as the "In:" value since that's what flows through
            int inVal = (outputCounters != null && outputCounters.length > 0) ? outputCounters[0] : 0;
            gc.fillText("In:" + inVal, x + 14, inputY + 4);
        }

        // Draw output connection point(s) with counters
        double outSpacing = height / (outputCount + 1);
        for (int i = 0; i < outputCount; i++) {
            double outputY = y + outSpacing * (i + 1);
            drawConnectionPoint(gc, x + width, outputY, false);
            // Draw output counter label - dark color for visibility
            int outVal = (outputCounters != null && i < outputCounters.length) ? outputCounters[i] : 0;
            String outLabel = outputCount > 1 ? "Out" + (i + 1) + ":" + outVal : "Out:" + outVal;
            gc.setFill(Color.rgb(60, 60, 60));
            // Measure text width approximately and right-align
            double textWidth = outLabel.length() * 5.5;
            gc.fillText(outLabel, x + width - textWidth - 10, outputY + 4);
        }

        // For ContainerOutput boundary nodes: show "Out:" on right side (frames leaving container)
        // ContainerOutput has no outputs but sends data to outside the container
        if (isBoundaryNode && "ContainerOutput".equals(nodeType)) {
            gc.setFill(Color.rgb(60, 60, 60));
            double outputY = y + height / 2;
            // Use inputCounter as the "Out:" value since that's what flows through
            String outLabel = "Out:" + inputCounter;
            double textWidth = outLabel.length() * 5.5;
            gc.fillText(outLabel, x + width - textWidth - 14, outputY + 4);
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
        // Draw white background for better visibility
        gc.setFill(Color.WHITE);
        gc.fillRect(x, y, CHECKBOX_SIZE, CHECKBOX_SIZE);

        // Draw checkbox border
        gc.setStroke(Color.DARKGRAY);
        gc.setLineWidth(1);
        gc.strokeRect(x, y, CHECKBOX_SIZE, CHECKBOX_SIZE);

        // Draw checkmark if checked (black checkmark on white background)
        if (checked) {
            gc.setStroke(Color.BLACK);
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
    private static void drawHelpIcon(GraphicsContext gc, double x, double y, boolean hasHelp) {
        // Use blue for available help, gray for unavailable
        gc.setFill(hasHelp ? COLOR_HELP_ICON : Color.rgb(180, 180, 180));
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
        Color lineColor = selected ? COLOR_NODE_SELECTED : Color.rgb(80, 80, 80);
        gc.setStroke(lineColor);
        gc.setLineWidth(selected ? LINE_WIDTH_SELECTED : LINE_WIDTH_NORMAL);

        // Draw bezier curve with control points that create a natural S-curve
        double ctrlOffset = Math.abs(endX - startX) / 2;
        // Ensure minimum control offset for vertical lines
        if (ctrlOffset < 30) ctrlOffset = 30;
        double ctrl1X = startX + ctrlOffset;
        double ctrl1Y = startY;
        double ctrl2X = endX - ctrlOffset;
        double ctrl2Y = endY;

        gc.beginPath();
        gc.moveTo(startX, startY);
        gc.bezierCurveTo(ctrl1X, ctrl1Y, ctrl2X, ctrl2Y, endX, endY);
        gc.stroke();

        // Draw arrow at end - use the tangent direction at the end of the bezier
        // For a cubic bezier, tangent at t=1 is proportional to (P3 - P2) = (end - ctrl2)
        double tangentX = endX - ctrl2X;
        double tangentY = endY - ctrl2Y;
        // If tangentY is 0 (horizontal tangent), use a small value based on the line direction
        // to ensure the arrow points in the direction the line approaches
        if (Math.abs(tangentY) < 0.001 && Math.abs(tangentX) > 0) {
            // Use the overall line direction to give a slight vertical hint
            tangentY = (endY - startY) * 0.1;
        }
        drawArrowHead(gc, endX, endY, tangentX, tangentY, lineColor);
    }

    /**
     * Render a connection line with queue statistics displayed at the midpoint.
     */
    public static void renderConnection(GraphicsContext gc,
                                         double startX, double startY,
                                         double endX, double endY,
                                         boolean selected,
                                         int queueSize, long totalFrames) {
        // Draw the basic connection first
        renderConnection(gc, startX, startY, endX, endY, selected);

        // Draw queue stats at midpoint of the bezier curve
        // Approximate midpoint using simple average (good enough for display)
        double midX = (startX + endX) / 2;
        double midY = (startY + endY) / 2;

        // Format the stats text
        String statsText = String.format("%,d / %,d", queueSize, totalFrames);

        // Save current state
        Font oldFont = gc.getFont();
        gc.setFont(QUEUE_STATS_FONT);

        // Measure text width using JavaFX Text helper
        javafx.scene.text.Text textHelper = new javafx.scene.text.Text(statsText);
        textHelper.setFont(QUEUE_STATS_FONT);
        double textWidth = textHelper.getLayoutBounds().getWidth();
        double textHeight = textHelper.getLayoutBounds().getHeight();

        // Background color based on backpressure
        Color bgColor = queueSize >= 5 ? COLOR_QUEUE_BG_BACKPRESSURE : COLOR_QUEUE_BG_NORMAL;

        // Draw background rounded rectangle - centered on midpoint
        double padding = 4;
        double rectW = textWidth + padding * 2;
        double rectH = textHeight + padding;
        double rectX = midX - rectW / 2;
        double rectY = midY - rectH / 2;

        gc.setFill(bgColor);
        gc.fillRoundRect(rectX, rectY, rectW, rectH, 6, 6);

        // Draw text - centered in the rectangle
        gc.setFill(COLOR_QUEUE_TEXT);
        // fillText draws from baseline, so we need to offset by ascent
        double textX = midX - textWidth / 2;
        double textY = midY + textHeight / 4;  // Approximate vertical centering
        gc.fillText(statsText, textX, textY);

        // Restore font
        gc.setFont(oldFont);
    }

    /**
     * Draw an arrow head at the end of a connection.
     */
    private static void drawArrowHead(GraphicsContext gc, double tipX, double tipY,
                                       double tangentX, double tangentY, Color color) {
        // Normalize tangent
        double len = Math.sqrt(tangentX * tangentX + tangentY * tangentY);
        if (len < 0.001) {
            tangentX = 1;
            tangentY = 0;
            len = 1;
        }
        double dirX = tangentX / len;
        double dirY = tangentY / len;

        // Arrow parameters
        double arrowLength = 12;
        double arrowWidth = 6;

        // Calculate arrow base center (back from tip)
        double baseX = tipX - dirX * arrowLength;
        double baseY = tipY - dirY * arrowLength;

        // Perpendicular direction
        double perpX = -dirY;
        double perpY = dirX;

        // Arrow wing points
        double x1 = baseX + perpX * arrowWidth;
        double y1 = baseY + perpY * arrowWidth;
        double x2 = baseX - perpX * arrowWidth;
        double y2 = baseY - perpY * arrowWidth;

        gc.setFill(color);
        gc.fillPolygon(new double[]{tipX, x1, x2}, new double[]{tipY, y1, y2}, 3);
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

    /**
     * Draw the stats line for processing nodes: "Pri: N   Work: N"
     * @param gc Graphics context
     * @param x X position to start drawing
     * @param y Y position for baseline
     * @param priority Current thread priority (1-10)
     * @param workUnits Number of work units completed
     */
    public static void drawStatsLine(GraphicsContext gc, double x, double y,
                                      int priority, long workUnits) {
        gc.setFont(STATS_LINE_FONT);
        // Red text if priority is below 5, otherwise dark gray
        gc.setFill(priority < 5 ? COLOR_STATS_SLOWED : COLOR_STATS_NORMAL);
        String statsLine = String.format("Pri: %d   Work: %,d", priority, workUnits);
        gc.fillText(statsLine, x, y);
    }

    /**
     * Draw the stats line for source nodes: "Pri: N   Work: N   FPS: N.NNN"
     * @param gc Graphics context
     * @param x X position to start drawing
     * @param y Y position for baseline
     * @param priority Current thread priority (1-10)
     * @param workUnits Number of work units completed
     * @param effectiveFps Current effective FPS after slowdown
     */
    public static void drawSourceStatsLine(GraphicsContext gc, double x, double y,
                                            int priority, long workUnits, double effectiveFps) {
        gc.setFont(STATS_LINE_FONT);
        // Red text if priority is below 5, otherwise dark gray
        gc.setFill(priority < 5 ? COLOR_STATS_SLOWED : COLOR_STATS_NORMAL);
        String statsLine = String.format("Pri: %d   Work: %,d   FPS: %.3f", priority, workUnits, effectiveFps);
        gc.fillText(statsLine, x, y);
    }
}
