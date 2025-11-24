package com.example.pipeline.nodes;

import org.eclipse.swt.widgets.Composite;
import org.eclipse.swt.widgets.Display;
import org.eclipse.swt.widgets.Shell;
import org.opencv.core.Mat;

/**
 * Abstract base class for all source nodes (File, Webcam, Blank, etc.).
 * Source nodes generate frames and have no input connections.
 */
public abstract class SourceNode extends PipelineNode {
    protected Shell shell;

    /**
     * Get the next frame from this source.
     * @return The next Mat frame, or null if no frame available
     */
    public abstract Mat getNextFrame();

    /**
     * Get the frames per second for this source.
     * @return FPS value
     */
    public abstract double getFps();

    /**
     * Show the properties dialog for this source node.
     */
    public abstract void showPropertiesDialog();

    /**
     * Dispose of resources held by this source node.
     * This should clean up overlays, thumbnails, and any other resources.
     */
    public abstract void dispose();

    /**
     * Check if this source is a video source (continuous frames).
     * @return true if video source, false for static images
     */
    public boolean isVideoSource() {
        return true; // Default to true, override if needed
    }

    public Shell getShell() {
        return shell;
    }

    public Display getDisplay() {
        return display;
    }
}
