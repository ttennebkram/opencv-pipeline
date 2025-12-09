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

    // FPS options (for UI dropdown)
    public static final String[] FPS_NAMES = {"1 fps", "5 fps", "10 fps", "15 fps", "30 fps"};
    public static final double[] FPS_VALUES = {1.0, 5.0, 10.0, 15.0, 30.0};

    private VideoCapture videoCapture;
    private int cameraIndex = 1; // Default to 1 (0 is often a virtual camera like iPhone)
    private int resolutionIndex = 1; // Default 640x480
    private double fps = 10.0; // Target FPS (any value allowed)
    private boolean mirrorHorizontal = true;
    private boolean isOpen = false;
    private boolean usingGStreamer = false; // Track if using GStreamer pipeline
    private Process rpicamProcess = null; // rpicam-still process for Pi camera
    private String rpicamTempFile = null; // Temp file for rpicam-still output
    private boolean usingRpicamPolling = false; // Track if using rpicam-still polling
    private int rpicamWidth = 640;
    private int rpicamHeight = 480;

    private Thread captureThread;
    private AtomicBoolean running = new AtomicBoolean(false);
    private Consumer<Image> onFrame;
    private Consumer<Mat> onMatFrame;

    private volatile Mat lastFrame;
    private volatile Image lastImage;

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

        int[] res = RESOLUTIONS[resolutionIndex];

        // First try standard V4L2 capture (works for USB webcams, preferred for higher FPS)
        videoCapture = new VideoCapture(cameraIndex);
        if (videoCapture.isOpened()) {
            // Set resolution
            videoCapture.set(Videoio.CAP_PROP_FRAME_WIDTH, res[0]);
            videoCapture.set(Videoio.CAP_PROP_FRAME_HEIGHT, res[1]);
            isOpen = true;
            usingGStreamer = false;
            System.out.println("Webcam opened (V4L2): camera " + cameraIndex + ", " + res[0] + "x" + res[1]);
            return true;
        }

        // V4L2 failed - on Raspberry Pi with libcamera CSI cameras, try rpicam-still as fallback
        // This is needed because OpenCV's V4L2 backend doesn't work with libcamera
        if (tryOpenWithRpicam(res[0], res[1])) {
            return true;
        }

        System.err.println("Failed to open webcam at index " + cameraIndex);
        isOpen = false;
        return false;
    }

    /**
     * Try to open the camera using rpicam-still frame polling.
     * This is needed for Raspberry Pi cameras that use libcamera.
     * We continuously capture JPEG frames to a temp file and read them.
     */
    private boolean tryOpenWithRpicam(int width, int height) {
        // Check if rpicam-still is available (Raspberry Pi with libcamera)
        try {
            Process check = Runtime.getRuntime().exec(new String[]{"which", "rpicam-still"});
            int exitCode = check.waitFor();
            if (exitCode != 0) {
                return false; // rpicam-still not available
            }
        } catch (Exception e) {
            return false;
        }

        // Create a unique temp file for this camera
        String tempFile = "/tmp/opencv_camera_" + System.currentTimeMillis() + ".jpg";
        rpicamTempFile = tempFile;

        try {
            // Test capture a single frame to verify camera works
            String[] testCmd = {
                "rpicam-still",
                "-n",  // No preview
                "--width", String.valueOf(width),
                "--height", String.valueOf(height),
                "-o", tempFile,
                "-t", "1"  // Very short timeout
            };

            System.out.println("Testing rpicam-still...");
            ProcessBuilder pb = new ProcessBuilder(testCmd);
            pb.redirectErrorStream(true);
            Process testProcess = pb.start();
            int exitCode = testProcess.waitFor();

            java.io.File testFile = new java.io.File(tempFile);
            if (exitCode != 0 || !testFile.exists() || testFile.length() == 0) {
                System.out.println("rpicam-still test capture failed");
                testFile.delete();
                return false;
            }

            // Load the test frame with OpenCV to verify it works
            Mat testFrame = org.opencv.imgcodecs.Imgcodecs.imread(tempFile);
            if (testFrame.empty()) {
                System.out.println("OpenCV failed to read rpicam-still output");
                testFrame.release();
                testFile.delete();
                return false;
            }
            testFrame.release();
            testFile.delete();

            // Start continuous capture process using rpicam-still --timelapse
            // This will write frames to the temp file at regular intervals
            String[] captureCmd = {
                "rpicam-still",
                "-n",  // No preview
                "--width", String.valueOf(width),
                "--height", String.valueOf(height),
                "-o", tempFile,
                "-t", "0",  // Run forever
                "--timelapse", "100"  // Capture every 100ms (10 fps)
            };

            System.out.println("Starting rpicam-still continuous capture...");
            pb = new ProcessBuilder(captureCmd);
            pb.redirectErrorStream(true);
            rpicamProcess = pb.start();

            // Give it time to start
            Thread.sleep(500);

            if (!rpicamProcess.isAlive()) {
                System.out.println("rpicam-still continuous capture failed to start");
                return false;
            }

            // Mark as using rpicam polling mode
            isOpen = true;
            usingRpicamPolling = true;
            rpicamWidth = width;
            rpicamHeight = height;
            System.out.println("Webcam opened (rpicam-still polling): " + width + "x" + height);
            return true;

        } catch (Exception e) {
            System.out.println("rpicam-still setup failed: " + e.getMessage());
            e.printStackTrace();
        }

        // Cleanup on failure
        stopRpicamProcess();
        return false;
    }

    /**
     * Stop the rpicam-still background process if running.
     */
    private void stopRpicamProcess() {
        if (rpicamProcess != null) {
            System.out.println("Stopping rpicam-still process...");
            rpicamProcess.destroyForcibly();
            try {
                rpicamProcess.waitFor();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            rpicamProcess = null;
            System.out.println("Stopped rpicam-still process");
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
        stopRpicamProcess(); // Stop rpicam-still if it was running

        // Clean up temp file
        if (rpicamTempFile != null) {
            new java.io.File(rpicamTempFile).delete();
            rpicamTempFile = null;
        }

        usingRpicamPolling = false;
        isOpen = false;
    }

    /**
     * Check if webcam is open.
     */
    public boolean isOpen() {
        if (usingRpicamPolling) {
            return isOpen && rpicamProcess != null && rpicamProcess.isAlive();
        }
        return isOpen && videoCapture != null && videoCapture.isOpened();
    }

    /**
     * Capture a single frame.
     * @return The captured Mat, or null if failed
     */
    public Mat captureFrame() {
        if (!isOpen) return null;

        // For rpicam polling mode, read from the temp file
        if (usingRpicamPolling) {
            return captureFrameFromRpicam();
        }

        // Standard V4L2/VideoCapture mode
        if (videoCapture == null) return null;

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
     * Capture a frame from rpicam-still polling mode.
     * Reads the latest JPEG frame from the temp file.
     */
    private Mat captureFrameFromRpicam() {
        if (rpicamTempFile == null) return null;

        java.io.File file = new java.io.File(rpicamTempFile);
        if (!file.exists() || file.length() == 0) {
            return null;
        }

        // Read the JPEG file with OpenCV
        Mat frame = org.opencv.imgcodecs.Imgcodecs.imread(rpicamTempFile);
        if (frame.empty()) {
            frame.release();
            return null;
        }

        if (mirrorHorizontal) {
            Core.flip(frame, frame, 1);
        }
        return frame;
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
            long frameDelayMs = (long) (1000.0 / fps);

            while (running.get()) {
                long startTime = System.currentTimeMillis();

                Mat frame = captureFrame();
                if (frame != null) {
                    // Store last frame (synchronized for thread-safe access)
                    synchronized (this) {
                        if (lastFrame != null) {
                            lastFrame.release();
                        }
                        lastFrame = frame.clone();
                    }

                    // Convert to Image
                    Image image = FXImageUtils.matToImage(frame);
                    lastImage = image;

                    // Deliver on JavaFX thread
                    if (onFrame != null && image != null) {
                        final Image finalImage = image;
                        Platform.runLater(() -> onFrame.accept(finalImage));
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
    public synchronized Mat getLastFrameClone() {
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

    /** Get FPS index for UI dropdown (finds closest match to current fps). */
    public int getFpsIndex() {
        int closest = 0;
        double minDiff = Math.abs(fps - FPS_VALUES[0]);
        for (int i = 1; i < FPS_VALUES.length; i++) {
            double diff = Math.abs(fps - FPS_VALUES[i]);
            if (diff < minDiff) {
                minDiff = diff;
                closest = i;
            }
        }
        return closest;
    }

    /** Set FPS from UI dropdown index. */
    public void setFpsIndex(int index) {
        if (index >= 0 && index < FPS_VALUES.length) {
            this.fps = FPS_VALUES[index];
        }
    }

    /** Set FPS directly to any value. */
    public void setFps(double fps) {
        this.fps = fps;
    }

    /** Get current FPS value. */
    public double getFps() { return fps; }

    public boolean isMirrorHorizontal() { return mirrorHorizontal; }
    public void setMirrorHorizontal(boolean mirror) { this.mirrorHorizontal = mirror; }

    public double getCurrentFps() { return fps; }

    /**
     * Detect available cameras.
     * @return Number of available cameras found
     */
    public static int detectCameras() {
        int count = 0;
        for (int i = 0; i <= 5; i++) {
            VideoCapture test = new VideoCapture(i);
            if (test.isOpened()) {
                count++;
                System.out.println("Found camera at index " + i);
            }
            test.release();
        }
        return count;
    }

    /**
     * Find the highest numbered available camera.
     * This is useful because virtual cameras (like iPhone) often take lower indices,
     * so the real webcam tends to be at the highest index.
     * @return The index of the highest available camera, or 0 if none found
     */
    public static int findHighestCamera() {
        int highest = -1;
        for (int i = 0; i <= 5; i++) {
            VideoCapture test = new VideoCapture(i);
            if (test.isOpened()) {
                highest = i;
                System.out.println("Found camera at index " + i);
            }
            test.release();
        }
        return highest >= 0 ? highest : 0;
    }

    /**
     * Find and open an available camera, returning a ready-to-use FXWebcamSource.
     * Auto-detects by finding the first camera that returns non-blank frames.
     * @return A new FXWebcamSource with the camera already opened, or null if none found
     */
    public static FXWebcamSource findAndOpenCamera() {
        return findAndOpenCamera(-1);
    }

    /**
     * Check if a frame is blank (all zeros or all same color).
     * @param frame The frame to check
     * @return true if the frame appears to be blank
     */
    private static boolean isBlankFrame(Mat frame) {
        if (frame == null || frame.empty()) {
            return true;
        }
        // Check if all pixels are the same (blank)
        // Calculate mean and stddev - a real image has variation
        org.opencv.core.MatOfDouble mean = new org.opencv.core.MatOfDouble();
        org.opencv.core.MatOfDouble stddev = new org.opencv.core.MatOfDouble();
        Core.meanStdDev(frame, mean, stddev);
        double[] stddevValues = stddev.toArray();
        mean.release();
        stddev.release();

        // If standard deviation is very low across all channels, it's likely blank
        double totalStdDev = 0;
        for (double v : stddevValues) {
            totalStdDev += v;
        }
        return totalStdDev < 1.0; // Very low variation = blank
    }

    /**
     * Find and open an available camera, trying the preferred index first.
     * When preferredIndex is -1 (auto-detect), finds the first camera returning non-blank frames.
     * @param preferredIndex The camera index to try first (-1 for auto-detect with frame validation)
     * @return A new FXWebcamSource with the camera already opened, or null if none found
     */
    public static FXWebcamSource findAndOpenCamera(int preferredIndex) {
        // If a specific camera is requested, just try to open it
        if (preferredIndex >= 0) {
            FXWebcamSource source = new FXWebcamSource(preferredIndex);
            if (source.open()) {
                return source;
            }
            return null;
        }

        // Auto-detect mode: find first camera that returns non-blank frames
        System.out.println("Auto-detecting cameras...");
        for (int i = 0; i <= 9; i++) {
            VideoCapture testCapture = new VideoCapture(i);
            if (testCapture.isOpened()) {
                System.out.println("Testing camera " + i + "...");

                // Try to grab a few frames and check if they're non-blank
                // Some cameras need a few frames to "warm up"
                boolean hasValidFrames = false;
                for (int attempt = 0; attempt < 5; attempt++) {
                    Mat testFrame = new Mat();
                    if (testCapture.read(testFrame) && !testFrame.empty()) {
                        if (!isBlankFrame(testFrame)) {
                            System.out.println("Camera " + i + " returns valid frames");
                            hasValidFrames = true;
                            testFrame.release();
                            break;
                        }
                    }
                    testFrame.release();
                    // Brief pause between attempts
                    try { Thread.sleep(100); } catch (InterruptedException e) { break; }
                }

                testCapture.release();

                if (hasValidFrames) {
                    // Found a good camera - create and return the source
                    FXWebcamSource source = new FXWebcamSource(i);
                    if (source.open()) {
                        return source;
                    }
                } else {
                    System.out.println("Camera " + i + " opens but returns blank frames, skipping");
                }
            }
        }

        // Also try rpicam-still fallback for Raspberry Pi
        // (open() already tries this internally if V4L2 fails)
        FXWebcamSource piSource = new FXWebcamSource(0);
        if (piSource.open()) {
            return piSource;
        }

        return null;
    }
}
