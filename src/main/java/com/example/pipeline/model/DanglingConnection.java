package com.example.pipeline.model;

import com.example.pipeline.nodes.PipelineNode;
import org.eclipse.swt.graphics.Point;

/**
 * Dangling connection with one end free (source fixed, target free).
 */
public class DanglingConnection {
    public PipelineNode source;
    public Point freeEnd;

    public DanglingConnection(PipelineNode source, Point freeEnd) {
        this.source = source;
        this.freeEnd = new Point(freeEnd.x, freeEnd.y);
    }
}
