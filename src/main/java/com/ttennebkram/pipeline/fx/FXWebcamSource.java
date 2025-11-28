package com.ttennebkram.pipeline.fx;

import javafx.application.Platform;
import javafx.scene.image.Image;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

/**
 * JavaFX-native webcam source that captures from a camera.
 * This is independent of SWT and can be used directly with JavaFX.
 */
public class FXWebcamSource {

    // Resolution options
    public static final String[] RESOLUTION_NAMES = {
        "320x240", "640x480", "1280x720", "1920x1080"
    };
    public static final int[][] RESOLUTIONS = {
        {320, 240}, {640, 480}, {1280, 720}, {1920, 1080}
    };

    // FPS options
    public static final String[] FPS_NAMES = {"1 fps", "5 fps", "10 fps", "15 fps", "30 fps"};
    public static final double[] FPS_VALUES = {1.0, 5.0, 10.0, 15.0, 30.0};

    private VideoCapture videoCapture;
    private int cameraIndex = 0;
    private int resolutionIndex = 1; // Default 640x480
    private int fpsIndex = 2; // Default 10 fps
    private boolean mirrorHorizontal = true;
    private boolean isOpen = false;

    private Thread captureThread;
    private AtomicBoolean running = new AtomicBoolean(false);
    private Consumer<Image> onFrame;
    private Consumer<Mat> onMatFrame;

    private Mat lastFrame;
    private Image lastImage;

    public FXWebcamSource() {
        this(0);
    }

    public FXWebcamSource(int cameraIndex) {
        this.cameraIndex = cameraIndex;
    }

    /**
     * Open the webcam with current settings.
     * @return true if successful
     */
    public boolean open() {
        close(); // Close any existing capture

        videoCapture = new VideoCapture(cameraIndex);
        if (videoCapture.isOpened()) {
            // Set resolution
            int[] res = RESOLUTIONS[resolutionIndex];
            videoCapture.set(Videoio.CAP_PROP_FRAME_WIDTH, res[0]);
            videoCapture.set(Videoio.CAP_PROP_FRAME_HEIGHT, res[1]);
            isOpen = true;
            System.out.println("Webcam opened: camera " + cameraIndex + ", " + res[0] + "x" + res[1]);
            return true;
        } else {
            System.err.println("Failed to open webcam at index " + cameraIndex);
            isOpen = false;
            return false;
        }
    }

    /**
     * Close the webcam.
     */
    public void close() {
        stop();
        if (videoCapture != null) {
            videoCapture.release();
            videoCapture = null;
        }
        isOpen = false;
    }

    /**
     * Check if webcam is open.
     */
    public boolean isOpen() {
        return isOpen && videoCapture != null && videoCapture.isOpened();
    }

    /**
     * Capture a single frame.
     * @return The captured Mat, or null if failed
     */
    public Mat captureFrame() {
        if (!isOpen()) return null;

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

    /**
     * Capture a single frame and convert to JavaFX Image.
     * @return The captured Image, or null if failed
     */
    public Image captureImage() {
        Mat frame = captureFrame();
        if (frame != null) {
            Image image = FXImageUtils.matToImage(frame);
            frame.release();
            return image;
        }
        return null;
    }

    /**
     * Start continuous capture in a background thread.
     * Frames are delivered via the onFrame callback on the JavaFX Application Thread.
     */
    public void start() {
        if (running.get()) return;
        if (!isOpen()) {
            if (!open()) return;
        }

        running.set(true);
        captureThread = new Thread(() -> {
            long frameDelayMs = (long) (1000.0 / FPS_VALUES[fpsIndex]);

            while (running.get()) {
                long startTime = System.currentTimeMillis();

                Mat frame = captureFrame();
                if (frame != null) {
                    // Store last frame
                    if (lastFrame != null) {
                        lastFrame.release();
                    }
                    lastFrame = frame.clone();

                    // Convert to Image
                    Image image = FXImageUtils.matToImage(frame);
                    lastImage = image;

                    // Deliver on JavaFX thread
                    if (onFrame != null && image != null) {
                        Platform.runLater(() -> onFrame.accept(image));
                    }
                    if (onMatFrame != null) {
                        Mat frameCopy = frame.clone();
                        Platform.runLater(() -> {
                            onMatFrame.accept(frameCopy);
                            // Note: caller is responsible for releasing
                        });
                    }

                    frame.release();
                }

                // Maintain frame rate
                long elapsed = System.currentTimeMillis() - startTime;
                long sleepTime = frameDelayMs - elapsed;
                if (sleepTime > 0) {
                    try {
                        Thread.sleep(sleepTime);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        break;
                    }
                }
            }
        }, "FXWebcamSource-" + cameraIndex);
        captureThread.setDaemon(true);
        captureThread.start();
    }

    /**
     * Stop continuous capture.
     */
    public void stop() {
        running.set(false);
        if (captureThread != null) {
            captureThread.interrupt();
            try {
                captureThread.join(1000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            captureThread = null;
        }
    }

    /**
     * Check if capture is running.
     */
    public boolean isRunning() {
        return running.get();
    }

    /**
     * Set callback for receiving frames as JavaFX Images.
     * Called on the JavaFX Application Thread.
     */
    public void setOnFrame(Consumer<Image> callback) {
        this.onFrame = callback;
    }

    /**
     * Set callback for receiving frames as OpenCV Mats.
     * Called on the JavaFX Application Thread.
     * Note: Caller is responsible for releasing the Mat.
     */
    public void setOnMatFrame(Consumer<Mat> callback) {
        this.onMatFrame = callback;
    }

    /**
     * Get the last captured frame as an Image.
     */
    public Image getLastImage() {
        return lastImage;
    }

    /**
     * Get the last captured frame as a Mat.
     * Note: Returns a clone, caller must release.
     */
    public Mat getLastFrameClone() {
        if (lastFrame != null && !lastFrame.empty()) {
            return lastFrame.clone();
        }
        return null;
    }

    // Getters and setters
    public int getCameraIndex() { return cameraIndex; }
    public void setCameraIndex(int index) {
        if (this.cameraIndex != index) {
            this.cameraIndex = index;
            if (isOpen()) {
                boolean wasRunning = isRunning();
                close();
                open();
                if (wasRunning) start();
            }
        }
    }

    public int getResolutionIndex() { return resolutionIndex; }
    public void setResolutionIndex(int index) {
        if (this.resolutionIndex != index) {
            this.resolutionIndex = index;
            if (isOpen()) {
                int[] res = RESOLUTIONS[resolutionIndex];
                videoCapture.set(Videoio.CAP_PROP_FRAME_WIDTH, res[0]);
                videoCapture.set(Videoio.CAP_PROP_FRAME_HEIGHT, res[1]);
            }
        }
    }

    public int getFpsIndex() { return fpsIndex; }
    public void setFpsIndex(int index) { this.fpsIndex = index; }

    public boolean isMirrorHorizontal() { return mirrorHorizontal; }
    public void setMirrorHorizontal(boolean mirror) { this.mirrorHorizontal = mirror; }

    public double getCurrentFps() { return FPS_VALUES[fpsIndex]; }

    /**
     * Detect available cameras.
     * @return Number of available cameras found
     */
    public static int detectCameras() {
        int count = 0;
        for (int i = 0; i <= 2; i++) {
            VideoCapture test = new VideoCapture(i);
            if (test.isOpened()) {
                count++;
                System.out.println("Found camera at index " + i);
            }
            test.release();
        }
        return count;
    }
}
