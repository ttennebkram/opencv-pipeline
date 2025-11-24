package com.example.pipeline.nodes;

import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.*;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

/**
 * Webcam source node that captures from a camera device.
 */
public class WebcamSourceNode extends SourceNode {

    // Webcam settings
    private int cameraIndex = -1; // -1 means auto-detect highest
    private int resolutionIndex = 1; // Default to 640x480
    private boolean mirrorHorizontal = true;
    private boolean skipAutoInit = false; // Skip auto-init for deserialization
    private int fpsIndex = 0; // Default to 1 fps

    // FPS options
    private static final String[] FPS_NAMES = {"1 fps", "5 fps", "10 fps", "15 fps", "30 fps"};
    private static final double[] FPS_VALUES = {1.0, 5.0, 10.0, 15.0, 30.0};

    // Video capture
    private VideoCapture videoCapture = null;
    private boolean isCapturing = false;

    // Cached list of available cameras (populated on init)
    private java.util.List<String> availableCameras = new java.util.ArrayList<>();

    // Canvas reference for redraw after thumbnail update
    private Canvas canvas;

    // Resolution options
    private static final String[] RESOLUTION_NAMES = {
        "320x240", "640x480", "1280x720", "1920x1080"
    };
    private static final int[][] RESOLUTIONS = {
        {320, 240}, {640, 480}, {1280, 720}, {1920, 1080}
    };

    // Constants
    private static final int SOURCE_NODE_HEIGHT = 120;

    public WebcamSourceNode(Shell shell, Display display, Canvas canvas, int x, int y) {
        this.shell = shell;
        this.display = display;
        this.canvas = canvas;
        this.x = x;
        this.y = y;
        this.height = SOURCE_NODE_HEIGHT;

        // Defer initialization - will be triggered by initAfterLoad() or immediately for new nodes
        display.asyncExec(() -> {
            // Check if we're being deserialized (skipAutoInit will be set by setters)
            display.timerExec(100, () -> {
                if (!skipAutoInit) {
                    // New node - auto-detect and open camera
                    new Thread(() -> {
                        initializeCamera();
                    }).start();
                }
            });
        });
    }

    // Getters/setters for serialization
    public int getCameraIndex() { return cameraIndex; }
    public void setCameraIndex(int v) {
        cameraIndex = v;
        skipAutoInit = true;
    }

    /**
     * Called after all properties are loaded from JSON to initialize camera.
     */
    public void initAfterLoad() {
        new Thread(() -> {
            // Detect available cameras for properties dialog
            availableCameras.clear();
            for (int i = 0; i <= 1; i++) {
                VideoCapture test = new VideoCapture(i);
                if (test.isOpened()) {
                    availableCameras.add("Camera " + i);
                }
                test.release();
            }
            // Open the saved camera (skip thumbnail capture since we'll load from cache)
            if (cameraIndex >= 0) {
                openCameraSkipThumbnail();
            }
        }).start();
    }

    /**
     * Open camera without capturing initial thumbnail (used when loading from cache).
     */
    private void openCameraSkipThumbnail() {
        if (videoCapture != null) {
            videoCapture.release();
        }

        System.out.println("Opening camera " + cameraIndex + " (skip thumbnail)");
        videoCapture = new VideoCapture(cameraIndex);
        if (videoCapture.isOpened()) {
            int[] res = RESOLUTIONS[resolutionIndex];
            videoCapture.set(Videoio.CAP_PROP_FRAME_WIDTH, res[0]);
            videoCapture.set(Videoio.CAP_PROP_FRAME_HEIGHT, res[1]);
            isCapturing = true;
            System.out.println("Camera opened successfully, resolution: " + res[0] + "x" + res[1]);
        } else {
            System.err.println("Failed to open camera " + cameraIndex);
            isCapturing = false;
        }
    }
    public int getResolutionIndex() { return resolutionIndex; }
    public void setResolutionIndex(int v) { resolutionIndex = v; }
    public boolean isMirrorHorizontal() { return mirrorHorizontal; }
    public void setMirrorHorizontal(boolean v) { mirrorHorizontal = v; }
    public int getFpsIndex() { return fpsIndex; }
    public void setFpsIndex(int v) { fpsIndex = v; }

    private void initializeCamera() {
        // Find available cameras - on macOS, usually just 0 or 0-1
        // Only check 0 and 1 to avoid OpenCV hanging on non-existent cameras
        int maxCamera = -1;
        availableCameras.clear();

        for (int i = 0; i <= 1; i++) {
            VideoCapture test = new VideoCapture(i);
            boolean opened = test.isOpened();
            test.release();

            if (opened) {
                maxCamera = i;
                availableCameras.add("Camera " + i);
                System.out.println("Found camera at index " + i);
            }
        }

        // Skip auto-selection if settings were loaded from deserialization
        if (skipAutoInit) {
            // Just open the camera with saved settings
            if (cameraIndex >= 0) {
                openCamera();
            }
            return;
        }

        if (maxCamera >= 0) {
            cameraIndex = maxCamera;
            System.out.println("Using camera index " + cameraIndex);
            openCamera();
        } else {
            System.err.println("No cameras found");
        }
    }

    public void openCamera() {
        // Release existing capture
        if (videoCapture != null) {
            videoCapture.release();
        }

        System.out.println("Opening camera " + cameraIndex);
        videoCapture = new VideoCapture(cameraIndex);
        if (videoCapture.isOpened()) {
            // Set resolution
            int[] res = RESOLUTIONS[resolutionIndex];
            videoCapture.set(Videoio.CAP_PROP_FRAME_WIDTH, res[0]);
            videoCapture.set(Videoio.CAP_PROP_FRAME_HEIGHT, res[1]);
            isCapturing = true;
            System.out.println("Camera opened successfully, resolution: " + res[0] + "x" + res[1]);

            // Capture first non-black frame for thumbnail
            Mat frame = new Mat();
            for (int attempt = 0; attempt < 10; attempt++) {
                if (videoCapture.read(frame) && !frame.empty()) {
                    // Check if frame is not all black (sample center pixel)
                    double[] pixel = frame.get(frame.height() / 2, frame.width() / 2);
                    if (pixel != null && (pixel[0] > 5 || pixel[1] > 5 || pixel[2] > 5)) {
                        System.out.println("Got valid frame on attempt " + (attempt + 1));
                        // Mirror if needed before creating thumbnail
                        if (mirrorHorizontal) {
                            Core.flip(frame, frame, 1);
                        }
                        setOutputMat(frame);
                        // Trigger canvas redraw on UI thread
                        if (canvas != null && !canvas.isDisposed()) {
                            display.asyncExec(() -> {
                                if (!canvas.isDisposed()) {
                                    canvas.redraw();
                                }
                            });
                        }
                        break;
                    }
                }
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
            frame.release();
        } else {
            System.err.println("Failed to open camera " + cameraIndex);
            isCapturing = false;
        }
    }

    public Mat getNextFrame() {
        if (!isCapturing || videoCapture == null || !videoCapture.isOpened()) {
            return null;
        }

        Mat frame = new Mat();
        if (videoCapture.read(frame) && !frame.empty()) {
            if (mirrorHorizontal) {
                Core.flip(frame, frame, 1);
            }
            return frame;
        }
        frame.release();
        return null;
    }

    public boolean isVideoSource() {
        return true;
    }

    public double getFps() {
        return FPS_VALUES[fpsIndex];
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
        gc.drawString("Webcam Source", x + 10, y + 4, true);
        boldFont.dispose();

        // Draw thumbnail if available
        if (thumbnail != null && !thumbnail.isDisposed()) {
            Rectangle bounds = thumbnail.getBounds();
            int thumbX = x + (width - bounds.width) / 2;
            int thumbY = y + 25;
            gc.drawImage(thumbnail, thumbX, thumbY);
        } else {
            // Draw placeholder
            gc.setForeground(display.getSystemColor(SWT.COLOR_GRAY));
            gc.drawString("(no output)", x + 10, y + 40, true);
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
        dialog.setText("Webcam Properties");
        dialog.setLayout(new GridLayout(2, false));

        // Description
        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText("Webcam Capture\ncv2.VideoCapture(index)");
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sigGd.horizontalSpan = 2;
        sigLabel.setLayoutData(sigGd);

        // Camera selector
        new Label(dialog, SWT.NONE).setText("Camera:");
        Combo cameraCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);

        // Use cached camera list
        java.util.List<String> cameras = availableCameras.isEmpty()
            ? java.util.Collections.singletonList("No cameras found")
            : availableCameras;

        cameraCombo.setItems(cameras.toArray(new String[0]));
        if (cameraIndex >= 0 && cameraIndex < cameras.size()) {
            cameraCombo.select(cameraIndex);
        } else if (!cameras.isEmpty()) {
            cameraCombo.select(cameras.size() - 1);
        }
        cameraCombo.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Resolution selector
        new Label(dialog, SWT.NONE).setText("Resolution:");
        Combo resCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        resCombo.setItems(RESOLUTION_NAMES);
        resCombo.select(resolutionIndex);
        resCombo.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Mirror checkbox
        new Label(dialog, SWT.NONE).setText("Mirror:");
        Button mirrorCheck = new Button(dialog, SWT.CHECK);
        mirrorCheck.setText("Mirror Left/Right");
        mirrorCheck.setSelection(mirrorHorizontal);

        // FPS selector
        new Label(dialog, SWT.NONE).setText("FPS:");
        Combo fpsCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        fpsCombo.setItems(FPS_NAMES);
        fpsCombo.select(fpsIndex);
        fpsCombo.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Buttons
        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(2, true));
        GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
        gd.horizontalSpan = 2;
        buttonComp.setLayoutData(gd);

        Button okBtn = new Button(buttonComp, SWT.PUSH);
        okBtn.setText("OK");
        okBtn.addListener(SWT.Selection, e -> {
            int newCameraIndex = cameraCombo.getSelectionIndex();
            int newResolution = resCombo.getSelectionIndex();
            boolean newMirror = mirrorCheck.getSelection();
            int newFps = fpsCombo.getSelectionIndex();

            boolean needReopen = (newCameraIndex != cameraIndex) || (newResolution != resolutionIndex);

            cameraIndex = newCameraIndex;
            resolutionIndex = newResolution;
            mirrorHorizontal = newMirror;
            fpsIndex = newFps;

            if (needReopen) {
                openCamera();
            }

            dialog.dispose();
        });

        Button cancelBtn = new Button(buttonComp, SWT.PUSH);
        cancelBtn.setText("Cancel");
        cancelBtn.addListener(SWT.Selection, e -> dialog.dispose());

        dialog.pack();
        Point cursor = shell.getDisplay().getCursorLocation();
        dialog.setLocation(cursor.x, cursor.y);
        dialog.open();
    }

    @Override
    public void dispose() {
        if (videoCapture != null) {
            videoCapture.release();
        }
    }

    // Save thumbnail to cache directory
    public void saveThumbnailToCache(String cacheDir, int nodeIndex) {
        if (outputMat != null && !outputMat.empty()) {
            try {
                java.io.File cacheFolder = new java.io.File(cacheDir);
                if (!cacheFolder.exists()) {
                    cacheFolder.mkdirs();
                }
                String thumbPath = cacheDir + java.io.File.separator + "webcam_" + nodeIndex + "_thumb.png";
                Mat resized = new Mat();
                double scale = Math.min((double) PROCESSING_NODE_THUMB_WIDTH / outputMat.width(),
                                        (double) PROCESSING_NODE_THUMB_HEIGHT / outputMat.height());
                org.opencv.imgproc.Imgproc.resize(outputMat, resized,
                    new org.opencv.core.Size(outputMat.width() * scale, outputMat.height() * scale));
                org.opencv.imgcodecs.Imgcodecs.imwrite(thumbPath, resized);
                resized.release();
            } catch (Exception e) {
                System.err.println("Failed to save webcam thumbnail: " + e.getMessage());
            }
        }
    }

    // Load thumbnail from cache directory
    public boolean loadThumbnailFromCache(String cacheDir, int nodeIndex) {
        String thumbPath = cacheDir + java.io.File.separator + "webcam_" + nodeIndex + "_thumb.png";
        java.io.File thumbFile = new java.io.File(thumbPath);
        if (thumbFile.exists()) {
            try {
                Mat loaded = org.opencv.imgcodecs.Imgcodecs.imread(thumbPath);
                if (!loaded.empty()) {
                    Mat rgb = new Mat();
                    org.opencv.imgproc.Imgproc.cvtColor(loaded, rgb, org.opencv.imgproc.Imgproc.COLOR_BGR2RGB);

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
                System.err.println("Failed to load webcam thumbnail: " + e.getMessage());
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
        frameDelayMs = (long) (1000.0 / getFps());

        processingThread = new Thread(() -> {
            while (running.get()) {
                try {
                    Mat frame = getNextFrame();
                    if (frame != null) {
                        // Update thumbnail
                        setOutputMat(frame);
                        notifyFrame(frame);
                        // Put frame on output queue (blocks if full)
                        if (outputQueue != null) {
                            outputQueue.put(frame);
                        }
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
        }, "WebcamSource-Thread");
        processingThread.start();
    }
}
