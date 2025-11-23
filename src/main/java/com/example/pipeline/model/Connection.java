package com.example.pipeline.model;

import com.example.pipeline.nodes.PipelineNode;

/**
 * Connection between two nodes.
 */
public class Connection {
    public PipelineNode source;
    public PipelineNode target;

    public Connection(PipelineNode source, PipelineNode target) {
        this.source = source;
        this.target = target;
    }
}
