package com.example.pipeline.nodes;

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
public class FileSourceNode extends PipelineNode {
    private Shell shell;
    private Canvas parentCanvas;
    private String imagePath = null;
    private Image thumbnail = null;
    private Mat loadedImage = null;
    private Composite overlayComposite;

    // Video support
    private VideoCapture videoCapture = null;
    private boolean isVideo = false;
    private boolean loopVideo = true;
    private double fps = 30.0;

    // FPS mode selection
    // 0 = "Just Once" (0 fps, single shot)
    // 1 = "Automatic" (video fps or 1 fps for static)
    // 2-N = specific fps values (e.g., 1, 5, 10, 15, 24, 30, 60)
    private int fpsMode = 1;  // Default to Automatic
    private static final String[] FPS_OPTIONS = {
        "Just Once", "Automatic", "1 fps", "5 fps", "10 fps", "15 fps", "24 fps", "30 fps", "60 fps"
    };
    private static final double[] FPS_VALUES = {
        0.0, -1.0, 1.0, 5.0, 10.0, 15.0, 24.0, 30.0, 60.0
    };  // -1 means automatic

    // Static image repeat (default 1 fps)
    private boolean repeatImage = true;
    private double staticFps = 1.0;

    // Thumbnail caching support
    private Mat thumbnailMat = null;

    // Constants - these should match PipelineEditor values
    private static final int SOURCE_NODE_HEIGHT = 120;
    private static final int SOURCE_NODE_THUMB_HEIGHT = 80;
    private static final int SOURCE_NODE_THUMB_WIDTH = 100;

    public FileSourceNode(Shell shell, Display display, Canvas canvas, int x, int y) {
        this.shell = shell;
        this.display = display;
        this.parentCanvas = canvas;
        this.x = x;
        this.y = y;
        this.height = SOURCE_NODE_HEIGHT;

        createOverlay();
    }

    // Getters/setters for serialization
    public String getImagePath() { return imagePath; }
    public void setImagePath(String v) { imagePath = v; }
    public int getFpsMode() { return fpsMode; }
    public void setFpsMode(int mode) { this.fpsMode = Math.max(0, Math.min(mode, FPS_OPTIONS.length - 1)); }
    public boolean isLoopVideo() { return loopVideo; }
    public void setLoopVideo(boolean v) { loopVideo = v; }

    private void createOverlay() {
        overlayComposite = new Composite(parentCanvas, SWT.NONE);
        overlayComposite.setLayout(new GridLayout(1, false));
        // Tighten bounds: start at y+22, use exact thumbnail height + small padding
        overlayComposite.setBounds(x + 5, y + 22, width - 10, SOURCE_NODE_THUMB_HEIGHT + 6);

        // Thumbnail label only - Choose button moved to Properties dialog
        Label thumbnailLabel = new Label(overlayComposite, SWT.BORDER | SWT.CENTER);
        GridData gd = new GridData(SWT.FILL, SWT.FILL, true, true);
        gd.heightHint = SOURCE_NODE_THUMB_HEIGHT;
        thumbnailLabel.setLayoutData(gd);
        thumbnailLabel.setText("No image");

        // Add mouse listeners for dragging from thumbnail area
        MouseListener dragMouseListener = new MouseAdapter() {
            @Override
            public void mouseDown(MouseEvent e) {
                if (e.button == 1) {  // Left click
                    // Convert to canvas coordinates and forward to canvas
                    Point canvasPoint = overlayComposite.toDisplay(e.x, e.y);
                    canvasPoint = parentCanvas.toControl(canvasPoint);

                    // Create a synthetic mouse event for the canvas
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
                // Convert to canvas coordinates and forward to canvas
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

        // Add mouse move listener for dragging
        MouseMoveListener dragMoveListener = e -> {
            // Convert to canvas coordinates and forward to canvas
            Point canvasPoint = overlayComposite.toDisplay(e.x, e.y);
            canvasPoint = parentCanvas.toControl(canvasPoint);

            Event event = new Event();
            event.x = canvasPoint.x;
            event.y = canvasPoint.y;
            event.stateMask = e.stateMask;
            parentCanvas.notifyListeners(SWT.MouseMove, event);
        };

        // Apply listeners to both overlay and thumbnail
        overlayComposite.addMouseListener(dragMouseListener);
        overlayComposite.addMouseMoveListener(dragMoveListener);
        thumbnailLabel.addMouseListener(dragMouseListener);
        thumbnailLabel.addMouseMoveListener(dragMoveListener);

        // Add right-click menu to overlay composite and thumbnail
        Menu contextMenu = new Menu(overlayComposite);
        MenuItem editItem = new MenuItem(contextMenu, SWT.PUSH);
        editItem.setText("Edit Properties...");
        editItem.addListener(SWT.Selection, evt -> showPropertiesDialog());

        overlayComposite.setMenu(contextMenu);
        thumbnailLabel.setMenu(contextMenu);

        // Ensure the overlay is visible
        overlayComposite.moveAbove(null);
        overlayComposite.layout();
    }

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
        }
    }

    public void loadMedia(String path) {
        // Check if it's a video file
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
        // Release any existing video capture
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

            // Create thumbnail
            Mat resized = new Mat();
            double scale = Math.min((double) SOURCE_NODE_THUMB_WIDTH / firstFrame.width(),
                                    (double) SOURCE_NODE_THUMB_HEIGHT / firstFrame.height());
            Imgproc.resize(firstFrame, resized,
                new Size(firstFrame.width() * scale, firstFrame.height() * scale));

            if (thumbnail != null) {
                thumbnail.dispose();
            }
            thumbnail = matToSwtImage(resized);

            // Capture thumbnail reference for asyncExec
            final Image thumbToSet = thumbnail;

            // Update the label - use asyncExec like loadImage() for consistency
            display.asyncExec(() -> {
                if (overlayComposite.isDisposed()) {
                    return;
                }
                if (thumbToSet == null || thumbToSet.isDisposed()) {
                    return;
                }
                Control[] children = overlayComposite.getChildren();
                if (children.length > 0 && children[0] instanceof Label) {
                    Label label = (Label) children[0];
                    label.setText("");
                    label.setImage(thumbToSet);
                }
            });

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
                // Loop back to start
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

    private void updateVideoThumbnail(Mat frame) {
        // Create thumbnail from current frame
        Mat resized = new Mat();
        double scale = Math.min((double) SOURCE_NODE_THUMB_WIDTH / frame.width(),
                                (double) SOURCE_NODE_THUMB_HEIGHT / frame.height());
        Imgproc.resize(frame, resized,
            new Size(frame.width() * scale, frame.height() * scale));

        final Image oldThumbnail = thumbnail;
        thumbnail = matToSwtImage(resized);
        resized.release();

        // Update label with new thumbnail and dispose old one on UI thread
        final Image thumbToSet = thumbnail;
        if (!display.isDisposed()) {
            display.asyncExec(() -> {
                if (overlayComposite.isDisposed()) return;
                Control[] children = overlayComposite.getChildren();
                if (children.length > 0 && children[0] instanceof Label) {
                    Label label = (Label) children[0];
                    label.setImage(thumbToSet);
                }
                // Dispose old thumbnail after setting new one
                if (oldThumbnail != null && !oldThumbnail.isDisposed()) {
                    oldThumbnail.dispose();
                }
            });
        }
    }

    public boolean isVideoSource() {
        return isVideo;
    }

    public double getFps() {
        // Handle FPS based on mode selection
        double selectedFps = FPS_VALUES[fpsMode];
        if (selectedFps == -1.0) {
            // Automatic mode: use video fps or 1 fps for static
            return isVideo ? fps : 1.0;
        }
        return selectedFps;
    }

    private void loadImage(String path) {
        loadedImage = Imgcodecs.imread(path);

        if (loadedImage.empty()) {
            return;
        }

        // Create thumbnail
        Mat resized = new Mat();
        double scale = Math.min((double) SOURCE_NODE_THUMB_WIDTH / loadedImage.width(),
                                (double) SOURCE_NODE_THUMB_HEIGHT / loadedImage.height());
        Imgproc.resize(loadedImage, resized,
            new Size(loadedImage.width() * scale, loadedImage.height() * scale));

        // Store the thumbnail Mat for caching
        thumbnailMat = resized;

        if (thumbnail != null) {
            thumbnail.dispose();
        }
        thumbnail = matToSwtImage(resized);

        // Capture the thumbnail reference for the asyncExec closure
        final Image thumbToSet = thumbnail;

        // Update the label - use asyncExec to defer until after UI is fully initialized
        display.asyncExec(() -> {
            if (overlayComposite.isDisposed()) {
                return;
            }
            if (thumbToSet == null || thumbToSet.isDisposed()) {
                return;
            }
            Control[] children = overlayComposite.getChildren();
            if (children.length > 0 && children[0] instanceof Label) {
                Label label = (Label) children[0];
                label.setText("");
                label.setImage(thumbToSet);

                // Force pack to resize label for the image
                label.pack();

                // Force complete layout refresh
                overlayComposite.layout(true, true);

                // Make sure it's visible and on top
                overlayComposite.setVisible(true);
                overlayComposite.moveAbove(null);

                // Force full repaint
                label.redraw();
                label.update();
                overlayComposite.redraw();
                overlayComposite.update();
                if (parentCanvas != null && !parentCanvas.isDisposed()) {
                    parentCanvas.redraw();
                    parentCanvas.update();
                }
            }
        });
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
            if (thumbnail != null) {
                thumbnail.dispose();
            }
            thumbnail = matToSwtImage(cached);

            // Update the label (thumbnail is now first child)
            Control[] children = overlayComposite.getChildren();
            if (children.length > 0 && children[0] instanceof Label) {
                Label label = (Label) children[0];
                label.setText("");
                label.setImage(thumbnail);
            }
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    private String getThumbnailCachePath(String cacheDir) {
        // Create a simple hash from the image path
        int hash = imagePath.hashCode();
        String ext = imagePath.toLowerCase().endsWith(".png") ? ".png" : ".jpg";
        return cacheDir + File.separator + "thumb_" + Math.abs(hash) + ext;
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

        // Create ImageData with proper scanline padding
        PaletteData palette = new PaletteData(0xFF0000, 0x00FF00, 0x0000FF);
        ImageData imageData = new ImageData(w, h, 24, palette);

        // Copy data row by row to handle scanline padding
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
        // Update overlay position (must match createOverlay)
        overlayComposite.setBounds(x + 5, y + 22, width - 10, SOURCE_NODE_THUMB_HEIGHT + 6);

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

        // Draw connection points (output only - this is a source node)
        drawConnectionPoints(gc);
    }

    @Override
    protected void drawConnectionPoints(GC gc) {
        // ImageSourceNode only has output point (it's a source)
        int radius = 6;
        Point output = getOutputPoint();
        gc.setBackground(new Color(255, 230, 200)); // Light orange fill
        gc.fillOval(output.x - radius, output.y - radius, radius * 2, radius * 2);
        gc.setForeground(new Color(200, 120, 50));  // Orange border
        gc.setLineWidth(2);
        gc.drawOval(output.x - radius, output.y - radius, radius * 2, radius * 2);
        gc.setLineWidth(1);  // Reset line width
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
        Label sourceLabel = new Label(dialog, SWT.NONE);
        sourceLabel.setText("Source:");

        // Source button and display
        Composite sourceComposite = new Composite(dialog, SWT.NONE);
        sourceComposite.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));
        sourceComposite.setLayout(new GridLayout(2, false));

        // Editable text field for path (allows copy/paste)
        Text pathText = new Text(sourceComposite, SWT.BORDER);
        String displayPath = imagePath != null ? imagePath : "";
        pathText.setText(displayPath);
        pathText.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Choose button
        Button chooseButton = new Button(sourceComposite, SWT.PUSH);
        chooseButton.setText("Choose...");
        chooseButton.addSelectionListener(new SelectionAdapter() {
            @Override
            public void widgetSelected(SelectionEvent e) {
                chooseImage();
                // Update the path text field
                String newPath = imagePath != null ? imagePath : "";
                pathText.setText(newPath);
            }
        });

        // FPS Mode label
        Label fpsLabel = new Label(dialog, SWT.NONE);
        fpsLabel.setText("FPS Mode:");

        // FPS Mode dropdown
        Combo fpsCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        fpsCombo.setItems(FPS_OPTIONS);
        fpsCombo.select(fpsMode);
        fpsCombo.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // OK button
        Button okButton = new Button(dialog, SWT.PUSH);
        okButton.setText("OK");
        okButton.setLayoutData(new GridData(SWT.RIGHT, SWT.CENTER, false, false));
        okButton.addSelectionListener(new SelectionAdapter() {
            @Override
            public void widgetSelected(SelectionEvent e) {
                // Get the path from the text field
                String newPath = pathText.getText().trim();
                if (!newPath.isEmpty() && (imagePath == null || !newPath.equals(imagePath))) {
                    // Path changed, load the new image
                    imagePath = newPath;
                    loadImage(newPath);
                }
                fpsMode = fpsCombo.getSelectionIndex();
                dialog.close();
            }
        });

        // Cancel button
        Button cancelButton = new Button(dialog, SWT.PUSH);
        cancelButton.setText("Cancel");
        cancelButton.addSelectionListener(new SelectionAdapter() {
            @Override
            public void widgetSelected(SelectionEvent e) {
                dialog.close();
            }
        });

        dialog.setDefaultButton(okButton);

        // Center dialog on parent
        Rectangle parentBounds = shell.getBounds();
        Rectangle dialogBounds = dialog.getBounds();
        dialog.setLocation(
            parentBounds.x + (parentBounds.width - dialogBounds.width) / 2,
            parentBounds.y + (parentBounds.height - dialogBounds.height) / 2
        );

        dialog.open();
    }

    public Mat getLoadedImage() {
        return loadedImage;
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
