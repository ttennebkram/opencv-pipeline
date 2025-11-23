package com.example.pipeline.model;

import com.example.pipeline.nodes.PipelineNode;
import org.eclipse.swt.graphics.Point;

/**
 * Reverse dangling connection (target fixed, source free).
 */
public class ReverseDanglingConnection {
    public PipelineNode target;
    public Point freeEnd;

    public ReverseDanglingConnection(PipelineNode target, Point freeEnd) {
        this.target = target;
        this.freeEnd = new Point(freeEnd.x, freeEnd.y);
    }
}
