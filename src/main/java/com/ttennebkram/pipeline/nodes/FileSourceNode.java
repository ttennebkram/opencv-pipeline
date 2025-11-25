package com.ttennebkram.pipeline.nodes;

import org.eclipse.swt.SWT;
import org.eclipse.swt.events.*;
import org.eclipse.swt.graphics.*;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import java.io.File;

/**
 * File source node with file chooser and thumbnail.
 */
public class FileSourceNode extends SourceNode {
    private String imagePath = null;
    private Mat loadedImage = null;
    private Canvas canvas;

    // Video support
    private VideoCapture videoCapture = null;
    private boolean isVideo = false;
    private boolean loopVideo = true;
    private double fps = 30.0;

    // FPS mode selection
    private int fpsMode = 1;  // Default to Automatic
    private static final String[] FPS_OPTIONS = {
        "Just Once", "Automatic", "1 fps", "5 fps", "10 fps", "15 fps", "24 fps", "30 fps", "60 fps"
    };
    private static final double[] FPS_VALUES = {
        0.0, -1.0, 1.0, 5.0, 10.0, 15.0, 24.0, 30.0, 60.0
    };

    // Thumbnail caching support
    private Mat thumbnailMat = null;

    // Constants
    private static final int SOURCE_NODE_HEIGHT = 120;

    public FileSourceNode(Shell shell, Display display, Canvas canvas, int x, int y) {
        this.shell = shell;
        this.display = display;
        this.canvas = canvas;
        this.x = x;
        this.y = y;
        this.height = SOURCE_NODE_HEIGHT;
    }

    // Getters/setters for serialization
    public String getImagePath() { return imagePath; }
    public void setImagePath(String v) { imagePath = v; }
    public int getFpsMode() { return fpsMode; }
    public void setFpsMode(int mode) { this.fpsMode = Math.max(0, Math.min(mode, FPS_OPTIONS.length - 1)); }
    public boolean isLoopVideo() { return loopVideo; }
    public void setLoopVideo(boolean v) { loopVideo = v; }

    private void chooseImage() {
        FileDialog dialog = new FileDialog(shell, SWT.OPEN);
        dialog.setText("Select Image or Video");
        dialog.setFilterExtensions(new String[]{
            "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.mp4;*.avi;*.mov;*.mkv;*.webm",
            "*.png;*.jpg;*.jpeg;*.bmp;*.tiff",
            "*.mp4;*.avi;*.mov;*.mkv;*.webm",
            "*.*"
        });
        dialog.setFilterNames(new String[]{
            "All Media Files",
            "Image Files",
            "Video Files",
            "All Files"
        });

        String path = dialog.open();
        if (path != null) {
            imagePath = path;
            loadMedia(path);
            if (canvas != null && !canvas.isDisposed()) {
                canvas.redraw();
            }
        }
    }

    public void loadMedia(String path) {
        String lower = path.toLowerCase();
        if (lower.endsWith(".mp4") || lower.endsWith(".avi") ||
            lower.endsWith(".mov") || lower.endsWith(".mkv") ||
            lower.endsWith(".webm")) {
            loadVideo(path);
        } else {
            loadImage(path);
        }
    }

    private void loadVideo(String path) {
        if (videoCapture != null) {
            videoCapture.release();
        }

        videoCapture = new VideoCapture(path);
        if (!videoCapture.isOpened()) {
            isVideo = false;
            return;
        }

        isVideo = true;
        fps = videoCapture.get(Videoio.CAP_PROP_FPS);
        if (fps <= 0) fps = 30.0;

        // Read first frame for thumbnail
        Mat firstFrame = new Mat();
        if (videoCapture.read(firstFrame) && !firstFrame.empty()) {
            loadedImage = firstFrame.clone();
            setOutputMat(firstFrame);
            firstFrame.release();
        }

        // Reset to beginning
        videoCapture.set(Videoio.CAP_PROP_POS_FRAMES, 0);
    }

    public Mat getNextFrame() {
        if (isVideo && videoCapture != null && videoCapture.isOpened()) {
            Mat frame = new Mat();
            if (videoCapture.read(frame)) {
                return frame;
            } else if (loopVideo) {
                videoCapture.set(Videoio.CAP_PROP_POS_FRAMES, 0);
                if (videoCapture.read(frame)) {
                    return frame;
                }
            }
            frame.release();
            return null;
        } else if (loadedImage != null && !loadedImage.empty()) {
            return loadedImage.clone();
        }
        return null;
    }

    public boolean isVideoSource() {
        return isVideo;
    }

    public double getFps() {
        double selectedFps = FPS_VALUES[fpsMode];
        if (selectedFps == -1.0) {
            return isVideo ? fps : 1.0;
        }
        return selectedFps;
    }

    private void loadImage(String path) {
        loadedImage = Imgcodecs.imread(path);
        if (loadedImage.empty()) {
            return;
        }
        setOutputMat(loadedImage);
    }

    public void saveThumbnailToCache(String cacheDir) {
        if (thumbnailMat != null && imagePath != null) {
            try {
                File cacheFolder = new File(cacheDir);
                if (!cacheFolder.exists()) {
                    cacheFolder.mkdirs();
                }
                String thumbPath = getThumbnailCachePath(cacheDir);
                Imgcodecs.imwrite(thumbPath, thumbnailMat);
            } catch (Exception e) {
                System.err.println("Failed to save thumbnail: " + e.getMessage());
            }
        }
    }

    public boolean loadThumbnailFromCache(String cacheDir) {
        if (imagePath == null) return false;

        String thumbPath = getThumbnailCachePath(cacheDir);
        File thumbFile = new File(thumbPath);
        if (!thumbFile.exists()) return false;

        try {
            Mat cached = Imgcodecs.imread(thumbPath);
            if (cached.empty()) return false;

            thumbnailMat = cached;
            // Convert to SWT Image for thumbnail
            Mat rgb = new Mat();
            if (cached.channels() == 3) {
                Imgproc.cvtColor(cached, rgb, Imgproc.COLOR_BGR2RGB);
            } else {
                rgb = cached;
            }

            int w = rgb.width();
            int h = rgb.height();
            byte[] data = new byte[w * h * 3];
            rgb.get(0, 0, data);

            PaletteData palette = new PaletteData(0xFF0000, 0x00FF00, 0x0000FF);
            ImageData imageData = new ImageData(w, h, 24, palette);
            for (int y = 0; y < h; y++) {
                for (int xp = 0; xp < w; xp++) {
                    int srcIdx = (y * w + xp) * 3;
                    int r = data[srcIdx] & 0xFF;
                    int g = data[srcIdx + 1] & 0xFF;
                    int b = data[srcIdx + 2] & 0xFF;
                    imageData.setPixel(xp, y, (r << 16) | (g << 8) | b);
                }
            }

            if (thumbnail != null && !thumbnail.isDisposed()) {
                thumbnail.dispose();
            }
            thumbnail = new Image(display, imageData);
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    private String getThumbnailCachePath(String cacheDir) {
        int hash = imagePath.hashCode();
        String ext = imagePath.toLowerCase().endsWith(".png") ? ".png" : ".jpg";
        return cacheDir + File.separator + "thumb_" + Math.abs(hash) + ext;
    }

    @Override
    public void paint(GC gc) {
        // Draw node background
        gc.setBackground(new Color(230, 240, 255));
        gc.fillRoundRectangle(x, y, width, height, 10, 10);

        // Draw border
        gc.setForeground(new Color(0, 0, 139));
        gc.setLineWidth(2);
        gc.drawRoundRectangle(x, y, width, height, 10, 10);

        // Draw title
        gc.setForeground(display.getSystemColor(SWT.COLOR_BLACK));
        Font boldFont = new Font(display, "Arial", 10, SWT.BOLD);
        gc.setFont(boldFont);
        gc.drawString("File Source", x + 10, y + 4, true);
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
        gc.drawString(getThreadPriorityLabel(), x + 10, y + 19, true);
        smallFont.dispose();

        // Draw thumbnail if available
        if (thumbnail != null && !thumbnail.isDisposed()) {
            Rectangle bounds = thumbnail.getBounds();
            int thumbX = x + (width - bounds.width) / 2;
            int thumbY = y + 34;
            gc.drawImage(thumbnail, thumbX, thumbY);
        } else {
            // Draw placeholder
            gc.setForeground(display.getSystemColor(SWT.COLOR_GRAY));
            gc.drawString("(no image)", x + 10, y + 50, true);
        }

        // Draw connection points (output only - this is a source node)
        drawConnectionPoints(gc);
    }

    @Override
    protected void drawConnectionPoints(GC gc) {
        int radius = 6;
        Point output = getOutputPoint();
        gc.setBackground(new Color(255, 230, 200));
        gc.fillOval(output.x - radius, output.y - radius, radius * 2, radius * 2);
        gc.setForeground(new Color(200, 120, 50));
        gc.setLineWidth(2);
        gc.drawOval(output.x - radius, output.y - radius, radius * 2, radius * 2);
        gc.setLineWidth(1);
    }

    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Image Source Properties");
        dialog.setLayout(new GridLayout(2, false));
        dialog.setSize(500, 200);

        // Description
        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText("Read from File\ncv2.imread(filename) / cv2.VideoCapture(filename)");
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sigGd.horizontalSpan = 2;
        sigLabel.setLayoutData(sigGd);

        // Image/Video source row
        new Label(dialog, SWT.NONE).setText("Source:");

        Composite sourceComposite = new Composite(dialog, SWT.NONE);
        sourceComposite.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));
        sourceComposite.setLayout(new GridLayout(2, false));

        Text pathText = new Text(sourceComposite, SWT.BORDER);
        pathText.setText(imagePath != null ? imagePath : "");
        pathText.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        Button chooseButton = new Button(sourceComposite, SWT.PUSH);
        chooseButton.setText("Choose...");
        chooseButton.addSelectionListener(new SelectionAdapter() {
            @Override
            public void widgetSelected(SelectionEvent e) {
                chooseImage();
                pathText.setText(imagePath != null ? imagePath : "");
            }
        });

        // FPS Mode
        new Label(dialog, SWT.NONE).setText("FPS Mode:");
        Combo fpsCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        fpsCombo.setItems(FPS_OPTIONS);
        fpsCombo.select(fpsMode);
        fpsCombo.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Buttons
        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(2, true));
        GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
        gd.horizontalSpan = 2;
        buttonComp.setLayoutData(gd);

        Button okButton = new Button(buttonComp, SWT.PUSH);
        okButton.setText("OK");
        okButton.addSelectionListener(new SelectionAdapter() {
            @Override
            public void widgetSelected(SelectionEvent e) {
                String newPath = pathText.getText().trim();
                if (!newPath.isEmpty() && (imagePath == null || !newPath.equals(imagePath))) {
                    imagePath = newPath;
                    loadMedia(newPath);
                }
                fpsMode = fpsCombo.getSelectionIndex();
                dialog.close();
            }
        });

        Button cancelButton = new Button(buttonComp, SWT.PUSH);
        cancelButton.setText("Cancel");
        cancelButton.addSelectionListener(new SelectionAdapter() {
            @Override
            public void widgetSelected(SelectionEvent e) {
                dialog.close();
            }
        });

        dialog.setDefaultButton(okButton);
        dialog.pack();
        Point cursor = shell.getDisplay().getCursorLocation();
        dialog.setLocation(cursor.x, cursor.y);
        dialog.open();
    }

    public Mat getLoadedImage() {
        return loadedImage;
    }

    @Override
    public Mat getOutputMat() {
        // Return loadedImage if outputMat is not set (for static images before pipeline runs)
        Mat mat = super.getOutputMat();
        if ((mat == null || mat.empty()) && loadedImage != null && !loadedImage.empty()) {
            return loadedImage;
        }
        return mat;
    }

    @Override
    public void dispose() {
        if (videoCapture != null) {
            videoCapture.release();
        }
    }

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
            boolean firstFrame = true;
            while (running.get()) {
                try {
                    Mat frame = getNextFrame();

                    // Increment work units regardless of output (even if null)
                    incrementWorkUnits();

                    if (frame != null) {
                        setOutputMat(frame);
                        notifyFrame(frame);
                        if (outputQueue != null) {
                            outputQueue.put(frame);
                        }
                    } else if (frame == null && !isVideo) {
                        if (!firstFrame && fps == 0) {
                            break;
                        }
                    }
                    firstFrame = false;

                    if (frameDelayMs > 0) {
                        Thread.sleep(frameDelayMs);
                    } else if (fps == 0 && !firstFrame) {
                        Thread.sleep(100);
                        break;
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }, "FileSource-Thread");
        processingThread.setPriority(threadPriority);
        processingThread.start();
    }
}
