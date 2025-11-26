package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonObject;
import org.eclipse.swt.graphics.Image;
import org.eclipse.swt.graphics.ImageData;
import org.eclipse.swt.graphics.PaletteData;
import org.eclipse.swt.widgets.Composite;
import org.eclipse.swt.widgets.Display;
import org.eclipse.swt.widgets.Shell;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;

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

    /**
     * Save thumbnail to cache directory.
     * Uses SOURCE_NODE_THUMB dimensions for proper scaling.
     */
    @Override
    public void saveThumbnailToCache(String cacheDir, int nodeIndex) {
        if (outputMat != null && !outputMat.empty()) {
            try {
                File cacheFolder = new File(cacheDir);
                if (!cacheFolder.exists()) {
                    cacheFolder.mkdirs();
                }
                String thumbPath = cacheDir + File.separator + "node_" + nodeIndex + "_thumb.png";
                // Save the output mat as thumbnail
                Mat resized = new Mat();
                double scale = Math.min((double) SOURCE_NODE_THUMB_WIDTH / outputMat.width(),
                                        (double) SOURCE_NODE_THUMB_HEIGHT / outputMat.height());
                Imgproc.resize(outputMat, resized,
                    new Size(outputMat.width() * scale, outputMat.height() * scale));
                Imgcodecs.imwrite(thumbPath, resized);
                resized.release();
            } catch (Exception e) {
                System.err.println("Failed to save source node thumbnail: " + e.getMessage());
            }
        }
    }

    /**
     * Load thumbnail from cache directory.
     */
    @Override
    public boolean loadThumbnailFromCache(String cacheDir, int nodeIndex) {
        String thumbPath = cacheDir + File.separator + "node_" + nodeIndex + "_thumb.png";
        File thumbFile = new File(thumbPath);
        if (thumbFile.exists()) {
            try {
                Mat loaded = Imgcodecs.imread(thumbPath);
                if (!loaded.empty()) {
                    // Convert to RGB for display
                    Mat rgb = new Mat();
                    if (loaded.channels() == 3) {
                        Imgproc.cvtColor(loaded, rgb, Imgproc.COLOR_BGR2RGB);
                    } else if (loaded.channels() == 1) {
                        Imgproc.cvtColor(loaded, rgb, Imgproc.COLOR_GRAY2RGB);
                    } else {
                        rgb = loaded;
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

                    if (thumbnail != null && !thumbnail.isDisposed()) {
                        thumbnail.dispose();
                    }
                    thumbnail = new Image(display, imageData);

                    // Also set outputMat so preview works before pipeline runs
                    Mat bgr = loaded.clone();

                    // Release old outputMat if it exists
                    if (outputMat != null && !outputMat.empty()) {
                        outputMat.release();
                    }
                    outputMat = bgr;

                    loaded.release();
                    rgb.release();
                    return true;
                }
            } catch (Exception e) {
                System.err.println("Failed to load source node thumbnail: " + e.getMessage());
            }
        }
        return false;
    }
}
