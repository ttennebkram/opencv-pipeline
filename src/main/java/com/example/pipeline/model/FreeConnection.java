package com.example.pipeline.model;

import org.eclipse.swt.graphics.Point;

/**
 * Free connection (both ends free, no nodes attached).
 */
public class FreeConnection {
    public Point startEnd;  // Non-arrow end
    public Point arrowEnd;  // Arrow end

    public FreeConnection(Point startEnd, Point arrowEnd) {
        this.startEnd = new Point(startEnd.x, startEnd.y);
        this.arrowEnd = new Point(arrowEnd.x, arrowEnd.y);
    }
}
