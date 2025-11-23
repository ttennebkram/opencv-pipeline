package com.example.pipeline.nodes;

import org.eclipse.swt.SWT;
import org.eclipse.swt.events.*;
import org.eclipse.swt.graphics.*;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

/**
 * Webcam source node that captures from a camera device.
 */
public class WebcamSourceNode extends PipelineNode {
    private Shell shell;
    private Canvas parentCanvas;
    private Composite overlayComposite;
    private Image thumbnail = null;

    // Webcam settings
    private int cameraIndex = -1; // -1 means auto-detect highest
    private int resolutionIndex = 1; // Default to 640x480
    private boolean mirrorHorizontal = true;
    private boolean skipAutoInit = false; // Skip auto-init for deserialization
    private int fpsIndex = 2; // Default to 10 fps

    // FPS options
    private static final String[] FPS_NAMES = {"1 fps", "5 fps", "10 fps", "15 fps", "30 fps"};
    private static final double[] FPS_VALUES = {1.0, 5.0, 10.0, 15.0, 30.0};

    // Video capture
    private VideoCapture videoCapture = null;
    private boolean isCapturing = false;
    private volatile boolean thumbnailUpdatePending = false; // Skip updates if one is pending

    // Resolution options
    private static final String[] RESOLUTION_NAMES = {
        "320x240", "640x480", "1280x720", "1920x1080"
    };
    private static final int[][] RESOLUTIONS = {
        {320, 240}, {640, 480}, {1280, 720}, {1920, 1080}
    };

    // Constants
    private static final int SOURCE_NODE_HEIGHT = 120;
    private static final int SOURCE_NODE_THUMB_HEIGHT = 80;
    private static final int SOURCE_NODE_THUMB_WIDTH = 100;

    public WebcamSourceNode(Shell shell, Display display, Canvas canvas, int x, int y) {
        this.shell = shell;
        this.display = display;
        this.parentCanvas = canvas;
        this.x = x;
        this.y = y;
        this.height = SOURCE_NODE_HEIGHT;

        createOverlay();

        // Auto-detect and open camera on a background thread to avoid blocking UI
        // but update thumbnail on UI thread via asyncExec
        new Thread(() -> {
            initializeCamera();
        }).start();
    }

    // Getters/setters for serialization
    public int getCameraIndex() { return cameraIndex; }
    public void setCameraIndex(int v) { cameraIndex = v; skipAutoInit = true; }
    public int getResolutionIndex() { return resolutionIndex; }
    public void setResolutionIndex(int v) { resolutionIndex = v; }
    public boolean isMirrorHorizontal() { return mirrorHorizontal; }
    public void setMirrorHorizontal(boolean v) { mirrorHorizontal = v; }
    public int getFpsIndex() { return fpsIndex; }
    public void setFpsIndex(int v) { fpsIndex = v; }

    private void createOverlay() {
        overlayComposite = new Composite(parentCanvas, SWT.NONE);
        overlayComposite.setBackground(new Color(230, 240, 255)); // Match node background
        overlayComposite.setLayout(new GridLayout(1, false));
        overlayComposite.setBounds(x + 5, y + 22, width - 10, SOURCE_NODE_THUMB_HEIGHT + 6);

        // Thumbnail label
        Label thumbnailLabel = new Label(overlayComposite, SWT.BORDER | SWT.CENTER);
        GridData gd = new GridData(SWT.FILL, SWT.FILL, true, true);
        gd.heightHint = SOURCE_NODE_THUMB_HEIGHT;
        thumbnailLabel.setLayoutData(gd);
        thumbnailLabel.setText("Webcam");
        thumbnailLabel.setBackground(display.getSystemColor(SWT.COLOR_WHITE));

        // Add mouse listeners for dragging
        MouseListener dragMouseListener = new MouseAdapter() {
            @Override
            public void mouseDown(MouseEvent e) {
                if (e.button == 1) {
                    Point canvasPoint = overlayComposite.toDisplay(e.x, e.y);
                    canvasPoint = parentCanvas.toControl(canvasPoint);
                    Event event = new Event();
                    event.x = canvasPoint.x;
                    event.y = canvasPoint.y;
                    event.button = e.button;
                    event.stateMask = e.stateMask;
                    parentCanvas.notifyListeners(SWT.MouseDown, event);
                }
            }

            @Override
            public void mouseUp(MouseEvent e) {
                Point canvasPoint = overlayComposite.toDisplay(e.x, e.y);
                canvasPoint = parentCanvas.toControl(canvasPoint);
                Event event = new Event();
                event.x = canvasPoint.x;
                event.y = canvasPoint.y;
                event.button = e.button;
                event.stateMask = e.stateMask;
                parentCanvas.notifyListeners(SWT.MouseUp, event);
            }

            @Override
            public void mouseDoubleClick(MouseEvent e) {
                showPropertiesDialog();
            }
        };

        MouseMoveListener dragMoveListener = e -> {
            Point canvasPoint = overlayComposite.toDisplay(e.x, e.y);
            canvasPoint = parentCanvas.toControl(canvasPoint);
            Event event = new Event();
            event.x = canvasPoint.x;
            event.y = canvasPoint.y;
            event.stateMask = e.stateMask;
            parentCanvas.notifyListeners(SWT.MouseMove, event);
        };

        overlayComposite.addMouseListener(dragMouseListener);
        overlayComposite.addMouseMoveListener(dragMoveListener);
        thumbnailLabel.addMouseListener(dragMouseListener);
        thumbnailLabel.addMouseMoveListener(dragMoveListener);

        // Context menu
        Menu contextMenu = new Menu(overlayComposite);
        MenuItem editItem = new MenuItem(contextMenu, SWT.PUSH);
        editItem.setText("Edit Properties...");
        editItem.addListener(SWT.Selection, evt -> showPropertiesDialog());

        overlayComposite.setMenu(contextMenu);
        thumbnailLabel.setMenu(contextMenu);

        overlayComposite.moveAbove(null);
        overlayComposite.layout();
    }

    private void initializeCamera() {
        // Skip auto-init if settings were loaded from deserialization
        if (skipAutoInit) {
            return;
        }

        // Find available cameras - on macOS, usually just 0 or 0-1
        int maxCamera = -1;
        int consecutiveFailures = 0;

        for (int i = 0; i <= 10; i++) {
            VideoCapture test = new VideoCapture(i);
            boolean opened = test.isOpened();
            test.release();

            if (opened) {
                maxCamera = i;
                consecutiveFailures = 0;
                System.out.println("Found camera at index " + i);
            } else {
                consecutiveFailures++;
                // Stop after 3 consecutive failures to avoid OpenCV errors
                if (consecutiveFailures >= 3) {
                    break;
                }
            }
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
            // Webcams often return black frames initially while warming up
            Mat frame = new Mat();
            for (int attempt = 0; attempt < 10; attempt++) {
                if (videoCapture.read(frame) && !frame.empty()) {
                    // Check if frame is not all black (sample center pixel)
                    double[] pixel = frame.get(frame.height() / 2, frame.width() / 2);
                    if (pixel != null && (pixel[0] > 5 || pixel[1] > 5 || pixel[2] > 5)) {
                        System.out.println("Got valid frame on attempt " + (attempt + 1) + ": " + frame.width() + "x" + frame.height());
                        // Mirror if needed before creating thumbnail
                        if (mirrorHorizontal) {
                            Core.flip(frame, frame, 1);
                        }
                        updateThumbnail(frame);
                        break;
                    }
                    System.out.println("Frame " + (attempt + 1) + " is black, retrying...");
                }
                try {
                    Thread.sleep(100); // Wait 100ms between attempts
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

    private void updateThumbnail(Mat frame) {
        // Skip if an update is already pending to prevent queue backup
        if (thumbnailUpdatePending) {
            return;
        }

        // Frame is already processed (mirrored if needed), just create thumbnail
        // Create thumbnail Mat (can be done on any thread)
        Mat resized = new Mat();
        double scale = Math.min((double) SOURCE_NODE_THUMB_WIDTH / frame.width(),
                                (double) SOURCE_NODE_THUMB_HEIGHT / frame.height());
        Imgproc.resize(frame, resized,
            new Size(frame.width() * scale, frame.height() * scale));

        // Convert to RGB data (can be done on any thread)
        Mat rgb = new Mat();
        if (resized.channels() == 3) {
            Imgproc.cvtColor(resized, rgb, Imgproc.COLOR_BGR2RGB);
        } else if (resized.channels() == 1) {
            Imgproc.cvtColor(resized, rgb, Imgproc.COLOR_GRAY2RGB);
        } else {
            rgb = resized;
        }

        int w = rgb.width();
        int h = rgb.height();
        byte[] data = new byte[w * h * 3];
        rgb.get(0, 0, data);

        // Create ImageData with direct data copy (much faster than setPixel loop)
        PaletteData palette = new PaletteData(0xFF0000, 0x00FF00, 0x0000FF);
        ImageData imageData = new ImageData(w, h, 24, palette);

        // Copy data row by row to handle scanline padding
        int bytesPerLine = imageData.bytesPerLine;
        for (int y = 0; y < h; y++) {
            int srcOffset = y * w * 3;
            int dstOffset = y * bytesPerLine;
            for (int x = 0; x < w; x++) {
                int srcIdx = srcOffset + x * 3;
                int dstIdx = dstOffset + x * 3;
                // Direct copy - data is already RGB from cvtColor
                imageData.data[dstIdx] = data[srcIdx];         // R
                imageData.data[dstIdx + 1] = data[srcIdx + 1]; // G
                imageData.data[dstIdx + 2] = data[srcIdx + 2]; // B
            }
        }

        // Update label with thumbnail on UI thread (Image must be created on UI thread)
        if (!display.isDisposed()) {
            thumbnailUpdatePending = true;
            display.asyncExec(() -> {
                thumbnailUpdatePending = false;
                if (overlayComposite.isDisposed()) return;

                // Create Image on UI thread
                final Image oldThumbnail = thumbnail;
                thumbnail = new Image(display, imageData);

                // Update the label with the thumbnail
                Control[] children = overlayComposite.getChildren();
                if (children.length > 0 && children[0] instanceof Label) {
                    Label label = (Label) children[0];
                    label.setText("");
                    label.setImage(thumbnail);
                }

                if (!parentCanvas.isDisposed()) {
                    parentCanvas.redraw();
                }
                // Dispose old thumbnail after setting new one
                if (oldThumbnail != null && !oldThumbnail.isDisposed()) {
                    oldThumbnail.dispose();
                }
            });
        }

        resized.release();
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
        return true; // Webcam is always a continuous source
    }

    @Override
    public void setOutputMat(Mat mat) {
        this.outputMat = mat;
        // Update the label-based thumbnail for source nodes
        // This is called from UI thread via asyncExec, so update directly
        if (mat != null && !mat.empty()) {
            updateThumbnailSync(mat);
        }
    }

    // Synchronous thumbnail update for when already on UI thread
    private void updateThumbnailSync(Mat frame) {
        // Create thumbnail Mat
        Mat resized = new Mat();
        double scale = Math.min((double) SOURCE_NODE_THUMB_WIDTH / frame.width(),
                                (double) SOURCE_NODE_THUMB_HEIGHT / frame.height());
        Imgproc.resize(frame, resized,
            new Size(frame.width() * scale, frame.height() * scale));

        // Convert to RGB
        Mat rgb = new Mat();
        if (resized.channels() == 3) {
            Imgproc.cvtColor(resized, rgb, Imgproc.COLOR_BGR2RGB);
        } else if (resized.channels() == 1) {
            Imgproc.cvtColor(resized, rgb, Imgproc.COLOR_GRAY2RGB);
        } else {
            rgb = resized;
        }

        int w = rgb.width();
        int h = rgb.height();
        byte[] data = new byte[w * h * 3];
        rgb.get(0, 0, data);

        // Create ImageData
        PaletteData palette = new PaletteData(0xFF0000, 0x00FF00, 0x0000FF);
        ImageData imageData = new ImageData(w, h, 24, palette);

        int bytesPerLine = imageData.bytesPerLine;
        for (int y = 0; y < h; y++) {
            int srcOffset = y * w * 3;
            int dstOffset = y * bytesPerLine;
            for (int x = 0; x < w; x++) {
                int srcIdx = srcOffset + x * 3;
                int dstIdx = dstOffset + x * 3;
                imageData.data[dstIdx] = data[srcIdx];
                imageData.data[dstIdx + 1] = data[srcIdx + 1];
                imageData.data[dstIdx + 2] = data[srcIdx + 2];
            }
        }

        // Update directly on UI thread
        if (!overlayComposite.isDisposed()) {
            final Image oldThumbnail = thumbnail;
            thumbnail = new Image(display, imageData);

            Control[] children = overlayComposite.getChildren();
            if (children.length > 0 && children[0] instanceof Label) {
                Label label = (Label) children[0];
                label.setText("");
                label.setImage(thumbnail);
            }

            if (oldThumbnail != null && !oldThumbnail.isDisposed()) {
                oldThumbnail.dispose();
            }
        }

        resized.release();
    }

    public double getFps() {
        return FPS_VALUES[fpsIndex];
    }

    private Image matToSwtImage(Mat mat) {
        Mat rgb = new Mat();
        if (mat.channels() == 3) {
            Imgproc.cvtColor(mat, rgb, Imgproc.COLOR_BGR2RGB);
        } else if (mat.channels() == 1) {
            Imgproc.cvtColor(mat, rgb, Imgproc.COLOR_GRAY2RGB);
        } else {
            rgb = mat;
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

        return new Image(display, imageData);
    }

    @Override
    public void paint(GC gc) {
        // Update overlay position
        overlayComposite.setBounds(x + 5, y + 22, width - 10, SOURCE_NODE_THUMB_HEIGHT + 6);
        overlayComposite.moveAbove(null);

        // Draw node background (same as File Source)
        gc.setBackground(new Color(230, 240, 255)); // Light blue
        gc.fillRoundRectangle(x, y, width, height, 10, 10);

        // Draw border
        gc.setForeground(new Color(0, 0, 139)); // Dark blue
        gc.setLineWidth(2);
        gc.drawRoundRectangle(x, y, width, height, 10, 10);

        // Draw title
        gc.setForeground(display.getSystemColor(SWT.COLOR_BLACK));
        Font boldFont = new Font(display, "Arial", 10, SWT.BOLD);
        gc.setFont(boldFont);
        gc.drawString("Webcam Source", x + 10, y + 4, true);
        boldFont.dispose();

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

        // Separator
        Label sep = new Label(dialog, SWT.SEPARATOR | SWT.HORIZONTAL);
        GridData sepGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sepGd.horizontalSpan = 2;
        sep.setLayoutData(sepGd);

        // Camera selector
        new Label(dialog, SWT.NONE).setText("Camera:");
        Combo cameraCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);

        // Detect available cameras
        java.util.List<String> cameras = new java.util.ArrayList<>();
        for (int i = 0; i <= 10; i++) {
            VideoCapture test = new VideoCapture(i);
            if (test.isOpened()) {
                cameras.add("Camera " + i);
                test.release();
            } else {
                test.release();
                break;
            }
        }

        if (cameras.isEmpty()) {
            cameras.add("No cameras found");
        }

        cameraCombo.setItems(cameras.toArray(new String[0]));
        if (cameraIndex >= 0 && cameraIndex < cameras.size()) {
            cameraCombo.select(cameraIndex);
        } else if (!cameras.isEmpty()) {
            cameraCombo.select(cameras.size() - 1); // Default to highest
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

        // Center on parent
        Rectangle parentBounds = shell.getBounds();
        Rectangle dialogBounds = dialog.getBounds();
        dialog.setLocation(
            parentBounds.x + (parentBounds.width - dialogBounds.width) / 2,
            parentBounds.y + (parentBounds.height - dialogBounds.height) / 2
        );

        dialog.open();
    }

    public Composite getOverlayComposite() {
        return overlayComposite;
    }

    public void dispose() {
        if (thumbnail != null && !thumbnail.isDisposed()) {
            thumbnail.dispose();
        }
        if (overlayComposite != null && !overlayComposite.isDisposed()) {
            overlayComposite.dispose();
        }
        if (videoCapture != null) {
            videoCapture.release();
        }
    }
}
