package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonObject;
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

    @Override
    public String getNodeName() {
        return this.getClass().getSimpleName();
    }

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

    /**
     * Source nodes have no input connection point.
     */
    @Override
    public org.eclipse.swt.graphics.Point getInputPoint() {
        return null;
    }

    /**
     * Start processing - generates frames from this source.
     * This is the standard implementation for all source nodes.
     * Subclasses typically don't need to override this.
     */
    @Override
    public void startProcessing() {
        if (running.get()) {
            return;
        }

        running.set(true);
        workUnitsCompleted = 0; // Reset counter on start
        double fps = getFps();
        frameDelayMs = fps > 0 ? (long) (1000.0 / fps) : 0;

        processingThread = new Thread(() -> {
            while (running.get()) {
                try {
                    Mat frame = getNextFrame();

                    // Increment work units regardless of output (even if null)
                    incrementWorkUnits();

                    if (frame != null) {
                        // Clone for persistent storage (outputMat for preview/thumbnail)
                        setOutputMat(frame.clone());

                        // Clone for preview callback (callback may run async after frame is released)
                        Mat previewClone = frame.clone();
                        notifyFrame(previewClone);
                        // Note: previewClone will be released by the callback

                        // Put frame on output queue (blocks if full)
                        if (outputQueue != null) {
                            // Clone for downstream node (they will release it)
                            outputQueue.put(frame.clone());
                        }

                        // Release the original frame from getNextFrame()
                        frame.release();
                    }

                    // Throttle frame rate
                    if (frameDelayMs > 0) {
                        Thread.sleep(frameDelayMs);
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }, getClass().getSimpleName() + "-Thread");
        processingThread.setPriority(threadPriority);
        processingThread.start();
    }
}
