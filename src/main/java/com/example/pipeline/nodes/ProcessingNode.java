package com.example.pipeline.nodes;

import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.*;
import org.eclipse.swt.widgets.Display;
import org.eclipse.swt.widgets.Shell;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import java.io.File;

/**
 * Base class for processing nodes with properties dialog support.
 */
public abstract class ProcessingNode extends PipelineNode {
    protected String name;

    @Override
    public String getNodeName() {
        return name;
    }
    protected Shell shell;
    protected boolean enabled = true;
    protected Runnable onChanged;  // Callback when properties change

    public ProcessingNode(Display display, Shell shell, String name, int x, int y) {
        this.display = display;
        this.shell = shell;
        this.name = name;
        this.x = x;
        this.y = y;
    }

    public void setOnChanged(Runnable onChanged) {
        this.onChanged = onChanged;
    }

    protected void notifyChanged() {
        if (onChanged != null) {
            onChanged.run();
        }
    }

    // Process input Mat and return output Mat
    public abstract Mat process(Mat input);

    // Show properties dialog
    public abstract void showPropertiesDialog();

    // Get description for tooltip
    public abstract String getDescription();

    // Get display name for toolbar button (longer, more descriptive)
    public abstract String getDisplayName();

    // Get category for toolbar grouping (e.g., "Basic", "Blur", "Edge Detection")
    public abstract String getCategory();

    @Override
    public void paint(GC gc) {
        // Draw node background
        gc.setBackground(new Color(230, 255, 230));
        gc.fillRoundRectangle(x, y, width, height, 10, 10);

        // Draw border
        gc.setForeground(new Color(0, 100, 0));
        gc.setLineWidth(2);
        gc.drawRoundRectangle(x, y, width, height, 10, 10);

        // Draw title
        gc.setForeground(display.getSystemColor(SWT.COLOR_BLACK));
        Font boldFont = new Font(display, "Arial", 10, SWT.BOLD);
        gc.setFont(boldFont);
        gc.drawString(name, x + 10, y + 5, true);
        boldFont.dispose();

        // Draw thread priority label
        Font smallFont = new Font(display, "Arial", 8, SWT.NORMAL);
        gc.setFont(smallFont);
        // Red text if priority is below 5, otherwise dark gray
        int currentPriority = getThreadPriority();
        if (currentPriority < 5) {
            gc.setForeground(new Color(200, 0, 0)); // Red for low priority
        } else {
            gc.setForeground(display.getSystemColor(SWT.COLOR_DARK_GRAY));
        }
        gc.drawString(getThreadPriorityLabel(), x + 10, y + 20, true);
        smallFont.dispose();

        // Draw thumbnail if available
        if (thumbnail != null && !thumbnail.isDisposed()) {
            Rectangle bounds = thumbnail.getBounds();
            int thumbX = x + (width - bounds.width) / 2;
            int thumbY = y + 35;
            gc.drawImage(thumbnail, thumbX, thumbY);
        } else {
            // Draw placeholder
            gc.setForeground(display.getSystemColor(SWT.COLOR_GRAY));
            gc.drawString("(no output)", x + 10, y + 50, true);
        }

        // Draw connection points
        drawConnectionPoints(gc);
    }

    public String getName() {
        return name;
    }

    public boolean isEnabled() {
        return enabled;
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }

    public Shell getShell() {
        return shell;
    }

    public Display getDisplay() {
        return display;
    }

    // Save thumbnail to cache directory
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
                double scale = Math.min((double) PROCESSING_NODE_THUMB_WIDTH / outputMat.width(),
                                        (double) PROCESSING_NODE_THUMB_HEIGHT / outputMat.height());
                Imgproc.resize(outputMat, resized,
                    new Size(outputMat.width() * scale, outputMat.height() * scale));
                Imgcodecs.imwrite(thumbPath, resized);
                resized.release();
            } catch (Exception e) {
                System.err.println("Failed to save thumbnail: " + e.getMessage());
            }
        }
    }

    // Load thumbnail from cache directory
    public boolean loadThumbnailFromCache(String cacheDir, int nodeIndex) {
        String thumbPath = cacheDir + File.separator + "node_" + nodeIndex + "_thumb.png";
        File thumbFile = new File(thumbPath);
        if (thumbFile.exists()) {
            try {
                Mat loaded = Imgcodecs.imread(thumbPath);
                if (!loaded.empty()) {
                    // Convert to RGB for display
                    Mat rgb = new Mat();
                    Imgproc.cvtColor(loaded, rgb, Imgproc.COLOR_BGR2RGB);

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

                    loaded.release();
                    rgb.release();
                    return true;
                }
            } catch (Exception e) {
                System.err.println("Failed to load thumbnail: " + e.getMessage());
            }
        }
        return false;
    }

    @Override
    public void startProcessing() {
        if (running.get()) {
            return;
        }

        running.set(true);

        processingThread = new Thread(() -> {

            while (running.get()) {
                try {
                    // Take from input queue (blocks until available)
                    if (inputQueue == null) {
                        Thread.sleep(100);
                        continue;
                    }

                    Mat input = inputQueue.take();
                    if (input == null) {
                        continue;
                    }

                    // Process the frame
                    Mat output = process(input);

                    // Update thumbnail and put on output queue
                    if (output != null) {
                        setOutputMat(output);
                        notifyFrame(output);

                        if (outputQueue != null) {
                            outputQueue.put(output);
                        }
                    }

                    // Check for backpressure and signal upstream if needed
                    checkBackpressure();

                    // Release input if it's different from output
                    if (input != output) {
                        input.release();
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }, "Processing-" + name + "-Thread");
        processingThread.setPriority(threadPriority);
        processingThread.start();
    }
}
