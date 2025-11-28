package com.ttennebkram.pipeline.fx;

/**
 * Lightweight connection data class for JavaFX rendering.
 * Represents a connection between two nodes (or with free endpoints).
 */
public class FXConnection {
    // Connected nodes (null if that end is free)
    public FXNode source;
    public FXNode target;

    // Connection point indices
    public int sourceOutputIndex;
    public int targetInputIndex;

    // Free endpoint positions (used when corresponding node is null)
    public double freeSourceX, freeSourceY;
    public double freeTargetX, freeTargetY;

    // Selection state
    public boolean selected;

    /**
     * Create a complete connection between two nodes.
     */
    public FXConnection(FXNode source, FXNode target) {
        this(source, 0, target, 0);
    }

    /**
     * Create a complete connection with specified indices.
     */
    public FXConnection(FXNode source, int outputIndex, FXNode target, int inputIndex) {
        this.source = source;
        this.target = target;
        this.sourceOutputIndex = outputIndex;
        this.targetInputIndex = inputIndex;
        this.selected = false;
    }

    /**
     * Create a dangling connection from a source node.
     */
    public static FXConnection createFromSource(FXNode source, int outputIndex, double freeX, double freeY) {
        FXConnection conn = new FXConnection(source, outputIndex, null, 0);
        conn.freeTargetX = freeX;
        conn.freeTargetY = freeY;
        return conn;
    }

    /**
     * Create a dangling connection to a target node.
     */
    public static FXConnection createToTarget(FXNode target, int inputIndex, double freeX, double freeY) {
        FXConnection conn = new FXConnection(null, 0, target, inputIndex);
        conn.freeSourceX = freeX;
        conn.freeSourceY = freeY;
        return conn;
    }

    /**
     * Check if this is a complete connection.
     */
    public boolean isComplete() {
        return source != null && target != null;
    }

    /**
     * Get the start point of this connection.
     */
    public double[] getStartPoint() {
        if (source != null) {
            return source.getOutputPoint(sourceOutputIndex);
        }
        return new double[]{freeSourceX, freeSourceY};
    }

    /**
     * Get the end point of this connection.
     */
    public double[] getEndPoint() {
        if (target != null) {
            return target.getInputPoint(targetInputIndex);
        }
        return new double[]{freeTargetX, freeTargetY};
    }

    /**
     * Check if a point is near this connection line.
     */
    public boolean isNear(double px, double py, double tolerance) {
        double[] start = getStartPoint();
        double[] end = getEndPoint();
        if (start == null || end == null) return false;

        // Simple distance check to bezier curve (approximation using line segments)
        // For better accuracy, could sample the bezier curve
        double dist = pointToLineDistance(px, py, start[0], start[1], end[0], end[1]);
        return dist <= tolerance;
    }

    private double pointToLineDistance(double px, double py, double x1, double y1, double x2, double y2) {
        double A = px - x1;
        double B = py - y1;
        double C = x2 - x1;
        double D = y2 - y1;

        double dot = A * C + B * D;
        double lenSq = C * C + D * D;
        double param = lenSq != 0 ? dot / lenSq : -1;

        double xx, yy;
        if (param < 0) {
            xx = x1;
            yy = y1;
        } else if (param > 1) {
            xx = x2;
            yy = y2;
        } else {
            xx = x1 + param * C;
            yy = y1 + param * D;
        }

        double dx = px - xx;
        double dy = py - yy;
        return Math.sqrt(dx * dx + dy * dy);
    }
}
