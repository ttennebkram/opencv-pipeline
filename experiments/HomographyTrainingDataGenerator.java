// HomographyTrainingDataGenerator.java
//
// Generates training data for homography estimation neural network.
//
// Uses JavaFX WebView to render web pages, then applies random homography
// transformations. Saves distorted images paired with their homography matrices.
//
// Run with:
//   mvn compile
//   java --module-path /path/to/javafx/lib --add-modules javafx.controls,javafx.web \
//        -cp "target/classes:experiments" HomographyTrainingDataGenerator
//
// Or via Maven (if configured):
//   mvn exec:exec@ml -Dml.class=HomographyTrainingDataGenerator '-Dml.args=--count 1000'
//

import javafx.application.Application;
import javafx.application.Platform;
import javafx.concurrent.Worker;
import javafx.embed.swing.SwingFXUtils;
import javafx.scene.Scene;
import javafx.scene.SnapshotParameters;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.StackPane;
import javafx.scene.web.WebEngine;
import javafx.scene.web.WebView;
import javafx.stage.Stage;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoWriter;
import org.opencv.videoio.Videoio;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;

/**
 * Generates training data for homography estimation:
 * 1. Renders web pages as images using JavaFX WebView
 * 2. Applies random homography (perspective) transformations
 * 3. Saves: original image, distorted image, homography matrix (and inverse)
 */
public class HomographyTrainingDataGenerator extends Application {

    // Load OpenCV native library
    static {
        nu.pattern.OpenCV.loadLocally();
    }

    // Configuration - 1080p output (high quality source data)
    // IMPORTANT: Downsample significantly in Python for training (e.g., 224x224, 320x240)
    private static int imageWidth = 1920;
    private static int imageHeight = 1080;
    // Render WebView at 8.5x11 inch paper proportions (~120 DPI)
    private static int renderWidth = 1020;
    private static int renderHeight = 1320;
    private static int samplesPerPage = 30;  // How many distortions per source image
    private static int totalPages = -1;       // How many pages to render (-1 = continuous until Ctrl+C)
    private static String outputDir = "/Volumes/SamsungBlue/ml-training/homography/training_data";
    private static List<String> urls = new ArrayList<>();

    // State
    private WebView webView;
    private WebEngine webEngine;
    private Stage previewStage;
    private javafx.scene.image.ImageView previewImageView;
    private int currentUrlIndex = 0;
    private int currentSampleIndex = 0;
    private int totalSamplesGenerated = 0;
    private int darkModePages = 0;
    private int lightModePages = 0;
    private long startTimeMillis = 0;
    private String runTimestamp = "";  // Timestamp for output filenames
    private Random random;  // Initialized with worker-specific seed

    // Video output - 4 videos: human (sorted) and training (random) for content and white
    private VideoWriter humanContentWriter;   // Content video sorted by rotation (for humans)
    private VideoWriter humanWhiteWriter;     // White video sorted by rotation (for humans)
    private VideoWriter trainingContentWriter; // Content video in random order (for training)
    private VideoWriter trainingWhiteWriter;   // White video in random order (for training)
    private PrintWriter videoLabelsWriter;
    private static int fps = 30;  // Video fps (default 30), also controls display delay if > 0
    private static int progressInterval = 100;  // Print dot every N samples
    private static boolean estimateOnly = false;  // Just show estimates, don't run
    private static boolean headless = false;  // Suppress preview window for speed
    private static boolean showTiming = false;  // Show detailed timing per sample
    private static boolean clearAllData = false;  // Clear all existing training data before starting
    private static int workerId = 0;  // Worker ID for multi-process mode (0 = single process)
    private static int numWorkers = 10;  // Number of worker processes to spawn (default 10, 0 = single process)
    private static int maxSeconds = -1;  // Maximum seconds to run (-1 = no limit)
    private static int maxFrames = -1;   // Maximum frames to output (-1 = no limit)
    private static final String STATS_CACHE_FILE = System.getProperty("user.home") + "/.homography_generator_stats.json";
    private static final String WORKER_STATS_PREFIX = ".worker_stats_";  // Worker stats files in output dir

    // Timing accumulators (microseconds)
    private static long timeHomography = 0;
    private static long timeRandu1 = 0;
    private static long timeWarp1 = 0;
    private static long timeShadow = 0;
    private static long timeCopyTo1 = 0;
    private static long timeImwrite1 = 0;
    private static long timeRandu2 = 0;
    private static long timeWarp2 = 0;
    private static long timeCopyTo2 = 0;
    private static long timeImwrite2 = 0;
    private static long timeLabelsJson = 0;
    private static long timeVideoBuffer = 0;
    private static long timePreview = 0;
    private static int timingSamples = 0;

    // Cached stats from previous runs
    private static double cachedFps = 10.0;  // Default fallback
    private static long cachedTotalFrames = 0;
    private static long cachedTotalRuntimeMs = 0;
    private static int cachedTotalRuns = 0;

    // Frame buffering for sorted/random video output
    private final List<FrameData> pageFrameBuffer = new ArrayList<>();
    private final Object bufferLock = new Object();

    /**
     * Holds frame data for deferred video writing (allows sorting by rotation angle).
     */
    private static class FrameData {
        Mat contentFrame;
        Mat whiteFrame;
        double paperRotation;
        String labelJson;

        FrameData(Mat content, Mat white, double rotation, String label) {
            this.contentFrame = content.clone();
            this.whiteFrame = white.clone();
            this.paperRotation = rotation;
            this.labelJson = label;
        }

        void release() {
            if (contentFrame != null) contentFrame.release();
            if (whiteFrame != null) whiteFrame.release();
        }
    }

    // Synchronization
    private CountDownLatch completionLatch;
    private volatile boolean shuttingDown = false;
    private volatile boolean paused = false;
    private static HomographyTrainingDataGenerator instance;

    // Static copies of stats for reliable access from any thread (updated after each page)
    private static volatile int staticTotalSamplesGenerated = 0;
    private static volatile int staticCurrentUrlIndex = 0;
    private static volatile int staticDarkModePages = 0;
    private static volatile int staticLightModePages = 0;
    private static volatile long staticStartTimeMillis = 0;

    // Dark mode - randomly chosen per page for training variety
    private boolean currentPageDarkMode = false;

    // Current page title extracted from Wikipedia URL (for filenames)
    private String currentPageTitle = "unknown";

    // Track all generated sample IDs for manifest (thread-safe)
    private final List<String> generatedSampleIds = Collections.synchronizedList(new ArrayList<>());

    public static void main(String[] args) {
        // Parse command line arguments
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--width":
                    imageWidth = Integer.parseInt(args[++i]);
                    break;
                case "--height":
                    imageHeight = Integer.parseInt(args[++i]);
                    break;
                case "--samples_per_page":
                case "--samples":
                    samplesPerPage = Integer.parseInt(args[++i]);
                    break;
                case "--pages":
                    totalPages = Integer.parseInt(args[++i]);
                    break;
                case "--output":
                    outputDir = args[++i];
                    break;
                case "--fps":
                    fps = Integer.parseInt(args[++i]);
                    break;
                case "--estimate":
                    estimateOnly = true;
                    break;
                case "--headless":
                    headless = true;
                    break;
                case "--timing":
                    showTiming = true;
                    break;
                case "--clear_all_generated_data":
                    clearAllData = true;
                    break;
                case "--worker-id":
                case "--worker":
                    workerId = Integer.parseInt(args[++i]);
                    break;
                case "--workers":
                    numWorkers = Integer.parseInt(args[++i]);
                    break;
                case "--seconds":
                case "--time":
                    maxSeconds = Integer.parseInt(args[++i]);
                    break;
                case "--frames":
                case "--max_frames":
                    maxFrames = Integer.parseInt(args[++i]);
                    break;
                case "-h":
                case "--help":
                    printUsage();
                    return;
            }
        }

        // If --workers N is specified, spawn N separate processes and act as coordinator
        if (numWorkers > 0 && workerId == 0) {
            spawnWorkers(args);
            return;  // Coordinator exits after workers complete
        }

        // Load cached stats from previous runs
        loadCachedStats();

        // Generate list of Wikipedia URLs
        generateUrlList();

        System.out.println("Homography Training Data Generator");
        System.out.println("===================================");
        if (workerId > 0) {
            System.out.println("Worker ID: " + workerId);
        }
        System.out.println("Image size: " + imageWidth + "x" + imageHeight);
        System.out.println("Samples per page: " + samplesPerPage);
        System.out.println("Output directory: " + outputDir);
        if (totalPages < 0) {
            System.out.println("Mode: CONTINUOUS (Ctrl+C to stop)");
            if (maxSeconds > 0 || maxFrames > 0) {
                System.out.print("Limits: ");
                if (maxSeconds > 0) System.out.print(maxSeconds + " seconds");
                if (maxSeconds > 0 && maxFrames > 0) System.out.print(", ");
                if (maxFrames > 0) System.out.print(maxFrames + " frames");
                System.out.println();
            } else {
                System.out.println("WARNING: Disk usage grows without limit! (~200 KB/frame)");
            }
        } else {
            int totalSamples = samplesPerPage * totalPages;
            long estimatedMB = (totalSamples * 200L) / 1024;  // ~200KB per frame
            System.out.println("Total pages: " + totalPages);
            System.out.printf("Total samples: %,d (estimated disk: ~%,d MB)%n", totalSamples, estimatedMB);
        }
        System.out.println();

        // Show projections based on cached stats
        printProjections();

        // If estimate-only mode, exit now
        if (estimateOnly) {
            System.out.println("(--estimate mode: exiting without generating data)");
            System.exit(0);  // Force exit - OpenCV native library may leave threads running
        }

        // Clear existing training data if requested (only in single-worker mode or coordinator)
        if (clearAllData && workerId == 0) {
            clearTrainingData();
        }

        // Create output directories
        try {
            Files.createDirectories(Paths.get(outputDir));
            Files.createDirectories(Paths.get(outputDir, "images"));
            Files.createDirectories(Paths.get(outputDir, "labels"));
            Files.createDirectories(Paths.get(outputDir, "videos"));
            Files.createDirectories(Paths.get(outputDir, "manifests"));
        } catch (IOException e) {
            System.err.println("Failed to create output directory: " + e.getMessage());
            return;
        }

        // Add shutdown hook for Ctrl-C
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("\n[Worker " + workerId + "] Shutdown requested, waiting for background threads...");

            // Signal shutdown and give background threads time to finish current work
            if (instance != null) {
                instance.shuttingDown = true;
            }
            try {
                Thread.sleep(3000);  // Wait 3 seconds for threads to finish
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }

            System.out.println("[Worker " + workerId + "] Cleaning up resources...");
            if (instance != null) {
                instance.cleanup();
            }
            System.out.println("[Worker " + workerId + "] Cleanup complete.");
        }));

        // Launch JavaFX application
        launch(args);
    }

    /**
     * Clean up resources - close video writers and labels file safely.
     */
    private synchronized void cleanup() {
        if (shuttingDown) return;
        shuttingDown = true;

        System.out.println("Closing resources...");

        // Flush any remaining buffered frames
        flushPageFrameBuffer();

        // Close all 4 video writers
        closeVideoWriter(humanContentWriter, "human content video");
        humanContentWriter = null;
        closeVideoWriter(humanWhiteWriter, "human white video");
        humanWhiteWriter = null;
        closeVideoWriter(trainingContentWriter, "training content video");
        trainingContentWriter = null;
        closeVideoWriter(trainingWhiteWriter, "training white video");
        trainingWhiteWriter = null;

        // Close video labels file
        if (videoLabelsWriter != null) {
            try {
                videoLabelsWriter.println("  null");
                videoLabelsWriter.println("]");
                videoLabelsWriter.close();
            } catch (Exception e) {
                System.err.println("Error closing labels file: " + e.getMessage());
            }
            videoLabelsWriter = null;
        }

        // Write manifest with samples generated so far
        writeManifest();

        // If we're a worker, write stats file for coordinator to aggregate
        // Do this BEFORE saving cached stats so coordinator can find it
        if (workerId > 0) {
            writeWorkerStats();
        }

        // Save stats to cache file for future estimates
        saveCachedStats();

        // Print timing summary if enabled
        if (showTiming && timingSamples > 0) {
            printTimingSummary();
        }

        // Print summary of output files (only for non-workers or single-process mode)
        if (workerId == 0) {
            printOutputSummary();
        }
    }

    /**
     * Print detailed timing breakdown per operation.
     */
    private void printTimingSummary() {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("TIMING BREAKDOWN (average per sample, " + timingSamples + " samples)");
        System.out.println("=".repeat(70));

        long total = timeHomography + timeRandu1 + timeWarp1 + timeShadow + timeCopyTo1 +
                     timeImwrite1 + timeRandu2 + timeWarp2 + timeCopyTo2 + timeImwrite2 +
                     timeLabelsJson + timeVideoBuffer + timePreview;

        // Print each timing in milliseconds with percentage
        printTimingLine("Homography compute", timeHomography, total);
        printTimingLine("Randu (content bg)", timeRandu1, total);
        printTimingLine("WarpPerspective x2 (content)", timeWarp1, total);
        printTimingLine("Shadow application", timeShadow, total);
        printTimingLine("CopyTo (content)", timeCopyTo1, total);
        printTimingLine("imwrite (content JPG)", timeImwrite1, total);
        printTimingLine("Randu (white bg)", timeRandu2, total);
        printTimingLine("WarpPerspective x2 (white)", timeWarp2, total);
        printTimingLine("CopyTo (white)", timeCopyTo2, total);
        printTimingLine("imwrite (white JPG)", timeImwrite2, total);
        printTimingLine("Labels JSON write", timeLabelsJson, total);
        printTimingLine("Video buffer add", timeVideoBuffer, total);
        printTimingLine("Preview update", timePreview, total);
        System.out.println("-".repeat(70));
        System.out.printf("TOTAL (measured):      %7.2f ms  (100.0%%)%n", total / 1000.0 / timingSamples);
        System.out.println("=".repeat(70));
    }

    private void printTimingLine(String label, long totalMicros, long grandTotal) {
        double avgMs = totalMicros / 1000.0 / timingSamples;
        double pct = grandTotal > 0 ? 100.0 * totalMicros / grandTotal : 0;
        System.out.printf("%-25s %7.2f ms  (%5.1f%%)%n", label, avgMs, pct);
    }

    /**
     * Print a summary of all output files and their purpose.
     */
    private void printOutputSummary() {
        // Compute absolute path for output directory
        java.io.File outDirFile = new java.io.File(outputDir);
        String absolutePath = outDirFile.getAbsolutePath();
        boolean isRelative = !outputDir.startsWith("/");

        System.out.println("\n" + "=".repeat(70));
        System.out.println("GENERATION COMPLETE");
        System.out.println("=".repeat(70));

        // Stats summary
        System.out.println("STATS:");
        System.out.println("-".repeat(70));
        int totalPages = darkModePages + lightModePages;
        long elapsedMs = System.currentTimeMillis() - startTimeMillis;
        double elapsedSec = elapsedMs / 1000.0;
        double elapsedMin = elapsedSec / 60.0;

        System.out.printf("  Pages processed:    %,d%n", totalPages);
        System.out.printf("  Total frames:       %,d (unique, written to each of 4 videos)%n", totalSamplesGenerated);
        if (totalPages > 0) {
            System.out.printf("  Dark mode pages:    %,d (%.1f%%)%n", darkModePages, 100.0 * darkModePages / totalPages);
            System.out.printf("  Light mode pages:   %,d (%.1f%%)%n", lightModePages, 100.0 * lightModePages / totalPages);
        }
        System.out.println();
        // Runtime stats
        if (elapsedMin >= 1.0) {
            System.out.printf("  Runtime:            %.1f minutes (%.0f seconds)%n", elapsedMin, elapsedSec);
        } else {
            System.out.printf("  Runtime:            %.1f seconds%n", elapsedSec);
        }
        if (elapsedSec > 0 && totalSamplesGenerated > 0) {
            double fps = totalSamplesGenerated / elapsedSec;
            double fpm = totalSamplesGenerated / elapsedMin;
            double fph = fpm * 60;
            double fpd = fph * 24;
            System.out.printf("  Throughput:         %.2f frames/sec, %.0f frames/min, %,.0f frames/hour, %,.0f frames/day%n", fps, fpm, fph, fpd);
            // Disk usage projection
            double mbPerHour = (fph * 200) / 1024;  // ~200KB per frame
            double gbPerDay = (fpd * 200) / (1024 * 1024);
            System.out.printf("  Disk usage rate:    ~%,.0f MB/hour, ~%,.1f GB/day%n", mbPerHour, gbPerDay);
        }
        System.out.println();

        // Print video file paths
        System.out.println("VIDEO FILES:");
        System.out.println("-".repeat(70));
        String[][] videoFiles = {
            {"videos/human_content_" + runTimestamp + ".avi", "Human review only: content sorted by rotation"},
            {"videos/human_white_" + runTimestamp + ".avi", "Human review only: geometry sorted by rotation"},
            {"videos/training_content_" + runTimestamp + ".avi", "For training: content in random order"},
            {"videos/training_white_" + runTimestamp + ".avi", "For training: geometry in random order"}
        };
        for (String[] video : videoFiles) {
            String filename = video[0];
            String description = video[1];
            if (isRelative) {
                System.out.println("  " + outputDir + "/" + filename);
                System.out.println("    -> " + absolutePath + "/" + filename);
            } else {
                System.out.println("  " + absolutePath + "/" + filename);
            }
            System.out.println("       - " + description);
        }
        System.out.println();

        System.out.println("LABELS:");
        String labelsFile = "videos/training_labels_" + runTimestamp + ".json";
        if (isRelative) {
            System.out.println("  " + outputDir + "/" + labelsFile);
            System.out.println("    -> " + absolutePath + "/" + labelsFile);
        } else {
            System.out.println("  " + absolutePath + "/" + labelsFile);
        }
        System.out.println("       Frame-by-frame labels (homography, inverse, corners)");
        System.out.println();

        System.out.println("INDIVIDUAL SAMPLES:");
        System.out.println("  images/<title>_NNNNN_SS.jpg       - Distorted content image");
        System.out.println("  images/<title>_NNNNN_SS_white.jpg - White page version");
        System.out.println("  labels/<title>_NNNNN_SS.json      - Full labels for each sample");
        System.out.println();

        System.out.println("METADATA:");
        System.out.println("  manifests/manifest.json   - Sample list and generation metadata");
        System.out.println();

        System.out.println("JSON LABEL FORMAT:");
        System.out.println("  image_width, image_height  - Output dimensions (" + imageWidth + "x" + imageHeight + ")");
        System.out.println("  homography[9]              - 3x3 transform matrix (row-major)");
        System.out.println("  inverse[9]                 - Inverse transform matrix");
        System.out.println("  corners[8]                 - Paper corner positions (TL,TR,BR,BL x,y)");
        System.out.println("  corners_normalized[8]      - Corners in [-1,1] range (centered)");
        System.out.println("=".repeat(70));
    }

    /**
     * Load cached stats from previous runs.
     */
    private static void loadCachedStats() {
        try {
            Path statsPath = Paths.get(STATS_CACHE_FILE);
            if (Files.exists(statsPath)) {
                String content = Files.readString(statsPath);
                // Simple JSON parsing without dependencies
                cachedFps = parseJsonDouble(content, "avg_fps", 10.0);
                cachedTotalFrames = parseJsonLong(content, "total_frames", 0);
                cachedTotalRuntimeMs = parseJsonLong(content, "total_runtime_ms", 0);
                cachedTotalRuns = (int) parseJsonLong(content, "total_runs", 0);
            }
        } catch (Exception e) {
            // Ignore errors, use defaults
        }
    }

    /**
     * Save stats from this run to cache file.
     */
    private void saveCachedStats() {
        if (totalSamplesGenerated == 0 || startTimeMillis == 0) return;

        long runtimeMs = System.currentTimeMillis() - startTimeMillis;
        double thisFps = totalSamplesGenerated / (runtimeMs / 1000.0);

        // Update cumulative stats
        long newTotalFrames = cachedTotalFrames + totalSamplesGenerated;
        long newTotalRuntimeMs = cachedTotalRuntimeMs + runtimeMs;
        int newTotalRuns = cachedTotalRuns + 1;
        double newAvgFps = newTotalFrames / (newTotalRuntimeMs / 1000.0);

        try (PrintWriter writer = new PrintWriter(STATS_CACHE_FILE)) {
            writer.println("{");
            writer.println("  \"last_updated\": \"" + new java.util.Date() + "\",");
            writer.println("  \"total_runs\": " + newTotalRuns + ",");
            writer.println("  \"total_frames\": " + newTotalFrames + ",");
            writer.println("  \"total_runtime_ms\": " + newTotalRuntimeMs + ",");
            writer.println("  \"avg_fps\": " + String.format("%.4f", newAvgFps) + ",");
            writer.println("  \"last_run\": {");
            writer.println("    \"frames\": " + totalSamplesGenerated + ",");
            writer.println("    \"runtime_ms\": " + runtimeMs + ",");
            writer.println("    \"fps\": " + String.format("%.4f", thisFps) + ",");
            writer.println("    \"pages_completed\": " + currentUrlIndex + ",");
            writer.println("    \"dark_mode_pages\": " + darkModePages + ",");
            writer.println("    \"light_mode_pages\": " + lightModePages + ",");
            writer.println("    \"image_width\": " + imageWidth + ",");
            writer.println("    \"image_height\": " + imageHeight + ",");
            writer.println("    \"samples_per_page\": " + samplesPerPage);
            writer.println("  }");
            writer.println("}");
            System.out.println("Stats saved to: " + STATS_CACHE_FILE);
        } catch (Exception e) {
            System.err.println("Warning: Could not save stats: " + e.getMessage());
        }
    }

    /**
     * Write worker stats to a file for the coordinator to aggregate.
     * Uses atomic write (temp file + rename) to prevent corruption.
     */
    private void writeWorkerStats() {
        if (workerId <= 0) return;

        long runtimeMs = System.currentTimeMillis() - startTimeMillis;
        String statsFile = outputDir + "/" + WORKER_STATS_PREFIX + workerId + ".json";
        String tempFile = outputDir + "/" + WORKER_STATS_PREFIX + workerId + ".tmp.json";

        System.out.println("[Worker " + workerId + "] Writing final stats to: " + statsFile);
        System.out.println("[Worker " + workerId + "] Frames: " + totalSamplesGenerated + ", Pages: " + currentUrlIndex);

        try {
            // Write to temp file first
            try (PrintWriter writer = new PrintWriter(tempFile)) {
                writer.println("{");
                writer.println("  \"worker_id\": " + workerId + ",");
                writer.println("  \"frames\": " + totalSamplesGenerated + ",");
                writer.println("  \"pages\": " + currentUrlIndex + ",");
                writer.println("  \"runtime_ms\": " + runtimeMs + ",");
                writer.println("  \"dark_mode_pages\": " + darkModePages + ",");
                writer.println("  \"light_mode_pages\": " + lightModePages);
                writer.println("}");
                writer.flush();
            }
            // Atomic rename (clobbers existing file)
            try {
                Files.move(Paths.get(tempFile), Paths.get(statsFile), 
                          StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            } catch (java.nio.file.AtomicMoveNotSupportedException e) {
                // Fallback if atomic move not supported
                Files.move(Paths.get(tempFile), Paths.get(statsFile), 
                          StandardCopyOption.REPLACE_EXISTING);
            }
            System.out.println("[Worker " + workerId + "] Stats file written successfully.");
        } catch (Exception e) {
            System.err.println("[Worker " + workerId + "] WARNING: Could not write worker stats: " + e.getMessage());
            e.printStackTrace();
            // Try to clean up temp file
            try { Files.deleteIfExists(Paths.get(tempFile)); } catch (Exception ignored) {}
        }
    }

    /**
     * Update static variables from instance variables.
     * Called after each page to ensure shutdown hook has current data.
     */
    private void updateStaticStats() {
        staticTotalSamplesGenerated = totalSamplesGenerated;
        staticCurrentUrlIndex = currentUrlIndex;
        staticDarkModePages = darkModePages;
        staticLightModePages = lightModePages;
    }

    /**
     * Write worker stats incrementally after each page completes.
     * Uses atomic write (temp file + rename) to prevent corruption.
     * This ensures we always have valid stats even if killed mid-run.
     */
    private void writeWorkerStatsIncremental() {
        if (workerId <= 0) return;

        long runtimeMs = System.currentTimeMillis() - staticStartTimeMillis;
        String statsFile = outputDir + "/" + WORKER_STATS_PREFIX + workerId + ".json";
        String tempFile = outputDir + "/" + WORKER_STATS_PREFIX + workerId + ".tmp.json";

        try {
            // Write to temp file first
            try (PrintWriter writer = new PrintWriter(tempFile)) {
                writer.println("{");
                writer.println("  \"worker_id\": " + workerId + ",");
                writer.println("  \"frames\": " + staticTotalSamplesGenerated + ",");
                writer.println("  \"pages\": " + staticCurrentUrlIndex + ",");
                writer.println("  \"runtime_ms\": " + runtimeMs + ",");
                writer.println("  \"dark_mode_pages\": " + staticDarkModePages + ",");
                writer.println("  \"light_mode_pages\": " + staticLightModePages);
                writer.println("}");
                writer.flush();
            }
            
            // Atomic rename (clobbers existing file)
            try {
                Files.move(Paths.get(tempFile), Paths.get(statsFile), 
                          StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            } catch (java.nio.file.AtomicMoveNotSupportedException e) {
                // Fallback if atomic move not supported
                Files.move(Paths.get(tempFile), Paths.get(statsFile), 
                          StandardCopyOption.REPLACE_EXISTING);
            }
        } catch (Exception e) {
            // Log failures for debugging
            System.err.println("[Worker " + workerId + "] ERROR writing incremental stats: " + e.getMessage());
            // Try to clean up temp file
            try { Files.deleteIfExists(Paths.get(tempFile)); } catch (Exception ignored) {}
        }
    }

    private static double parseJsonDouble(String json, String key, double defaultVal) {
        try {
            String pattern = "\"" + key + "\"\\s*:\\s*([0-9.\\-]+)";
            java.util.regex.Matcher m = java.util.regex.Pattern.compile(pattern).matcher(json);
            if (m.find()) return Double.parseDouble(m.group(1));
        } catch (Exception e) { }
        return defaultVal;
    }

    private static long parseJsonLong(String json, String key, long defaultVal) {
        try {
            String pattern = "\"" + key + "\"\\s*:\\s*([0-9\\-]+)";
            java.util.regex.Matcher m = java.util.regex.Pattern.compile(pattern).matcher(json);
            if (m.find()) return Long.parseLong(m.group(1));
        } catch (Exception e) { }
        return defaultVal;
    }

    /**
     * Print projections based on cached stats from previous runs.
     */
    private static void printProjections() {
        System.out.println("PROJECTIONS (based on previous runs):");
        System.out.println("-".repeat(50));

        if (cachedTotalRuns == 0) {
            System.out.println("  (No previous runs - using default estimates)");
            System.out.printf("  Estimated throughput: ~%.1f fps%n", cachedFps);
        } else {
            System.out.printf("  Based on: %d run(s), %,d total frames%n", cachedTotalRuns, cachedTotalFrames);
            System.out.printf("  Average throughput:   %.2f fps%n", cachedFps);
        }

        double fpm = cachedFps * 60;
        double fph = fpm * 60;
        double fpd = fph * 24;

        System.out.printf("  Projected rates:      %.0f frames/min, %,.0f frames/hour, %,.0f frames/day%n", fpm, fph, fpd);

        // Disk usage projections
        double mbPerHour = (fph * 200) / 1024;
        double gbPerDay = (fpd * 200) / (1024 * 1024);
        System.out.printf("  Disk usage rate:      ~%,.0f MB/hour, ~%,.1f GB/day%n", mbPerHour, gbPerDay);

        // Time estimates for specific frame counts
        if (totalPages > 0) {
            int targetFrames = totalPages * samplesPerPage;
            double estimatedSeconds = targetFrames / cachedFps;
            double estimatedMinutes = estimatedSeconds / 60;
            double estimatedHours = estimatedMinutes / 60;
            System.out.println();
            System.out.printf("  For %,d frames (%d pages × %d samples):%n", targetFrames, totalPages, samplesPerPage);
            if (estimatedHours >= 1) {
                System.out.printf("    Estimated time:     ~%.1f hours%n", estimatedHours);
            } else if (estimatedMinutes >= 1) {
                System.out.printf("    Estimated time:     ~%.1f minutes%n", estimatedMinutes);
            } else {
                System.out.printf("    Estimated time:     ~%.0f seconds%n", estimatedSeconds);
            }
            long estimatedMB = (targetFrames * 200L) / 1024;
            System.out.printf("    Estimated disk:     ~%,d MB%n", estimatedMB);
        }
        System.out.println();
    }

    private static void printUsage() {
        System.out.println("Usage: HomographyTrainingDataGenerator [options]");
        System.out.println("Options:");
        System.out.println("  --width N            Image width (default: 1920)");
        System.out.println("  --height N           Image height (default: 1080)");
        System.out.println("  --samples_per_page N Distortions per page (default: 30)");
        System.out.println("  --pages N            Number of pages (-1 = continuous, default: -1)");
        System.out.println("  --output DIR         Output directory (default: /Volumes/SamsungBlue/ml-training/homography/training_data)");
        System.out.println("  --fps N              Video fps and display delay (default: 30)");
        System.out.println("  --estimate           Show projections based on previous runs, don't generate");
        System.out.println("  --headless           Suppress all windows (faster processing)");
        System.out.println("  --timing             Show detailed timing breakdown per operation");
        System.out.println("  --clear_all_generated_data  DELETE all existing training data before starting");
        System.out.println("  --worker-id N        Worker ID for multi-process mode (adds _wN suffix to files)");
        System.out.println("  --workers N          Spawn N parallel worker processes (default: 10, 0=single)");
        System.out.println("  --seconds N          Stop after N seconds (-1 = no limit, default: -1)");
        System.out.println("  --frames N           Stop after N frames (-1 = no limit, default: -1)");
        System.out.println("  -h, --help           Show this help");
        System.out.println();
        System.out.println("*** IMPORTANT: RESIZE FOR TRAINING! ***");
        System.out.println("Default 1080p output is for high-quality source data.");
        System.out.println("You MUST resize down significantly for neural network training:");
        System.out.println("  - 224x224 (typical CNN input)");
        System.out.println("  - 320x240 (faster training)");
        System.out.println("  - 160x120 (very fast training)");
        System.out.println("Training on 1080p directly will be extremely slow and wasteful!");
        System.out.println();
        System.out.println("Disk Space Estimates (1920x1080, ~1 MB/frame):");
        System.out.println("  1,000 frames   = ~1 GB     (~2 min)");
        System.out.println("  10,000 frames  = ~10 GB    (~17 min)");
        System.out.println("  36,000 frames  = ~36 GB    (~1 hour)");
        System.out.println("  100,000 frames = ~100 GB   (~3 hours)");
        System.out.println();
        System.out.println("WARNING: In continuous mode, disk usage grows without limit!");
        System.out.println("         Use Ctrl+C to stop when you have enough data.");
    }

    /**
     * Clear all existing training data from the output directory.
     * Removes images/, labels/ subdirectories and all video/label files.
     */
    private static void clearTrainingData() {
        Path outPath = Paths.get(outputDir);
        if (!Files.exists(outPath)) {
            System.out.println("Output directory does not exist, nothing to clear.");
            return;
        }

        System.out.println("Clearing existing training data from: " + outputDir);
        
        try {
            // Count files before deletion for reporting
            long[] counts = {0, 0, 0, 0};  // images, labels, videos, other
            
            // Delete images/ directory
            Path imagesDir = outPath.resolve("images");
            if (Files.exists(imagesDir)) {
                try (var stream = Files.walk(imagesDir)) {
                    stream.sorted(Comparator.reverseOrder())
                          .forEach(p -> {
                              try {
                                  if (Files.isRegularFile(p)) counts[0]++;
                                  Files.delete(p);
                              } catch (IOException e) {
                                  System.err.println("  Failed to delete: " + p);
                              }
                          });
                }
            }

            // Delete labels/ directory
            Path labelsDir = outPath.resolve("labels");
            if (Files.exists(labelsDir)) {
                try (var stream = Files.walk(labelsDir)) {
                    stream.sorted(Comparator.reverseOrder())
                          .forEach(p -> {
                              try {
                                  if (Files.isRegularFile(p)) counts[1]++;
                                  Files.delete(p);
                              } catch (IOException e) {
                                  System.err.println("  Failed to delete: " + p);
                              }
                          });
                }
            }

            // Delete videos/ directory
            Path videosDir = outPath.resolve("videos");
            if (Files.exists(videosDir)) {
                try (var stream = Files.walk(videosDir)) {
                    stream.sorted(Comparator.reverseOrder())
                          .forEach(p -> {
                              try {
                                  if (Files.isRegularFile(p)) counts[2]++;
                                  Files.delete(p);
                              } catch (IOException e) {
                                  System.err.println("  Failed to delete: " + p);
                              }
                          });
                }
            }

            // Delete manifests/ directory
            Path manifestsDir = outPath.resolve("manifests");
            if (Files.exists(manifestsDir)) {
                try (var stream = Files.walk(manifestsDir)) {
                    stream.sorted(Comparator.reverseOrder())
                          .forEach(p -> {
                              try {
                                  if (Files.isRegularFile(p)) counts[3]++;
                                  Files.delete(p);
                              } catch (IOException e) {
                                  System.err.println("  Failed to delete: " + p);
                              }
                          });
                }
            }

            // Delete other top-level files
            try (var stream = Files.list(outPath)) {
                stream.filter(Files::isRegularFile)
                      .forEach(p -> {
                          try {
                              counts[3]++;
                              Files.delete(p);
                          } catch (IOException e) {
                              System.err.println("  Failed to delete: " + p);
                          }
                      });
            }

            System.out.printf("Cleared: %d images, %d labels, %d videos, %d other files%n",
                             counts[0], counts[1], counts[2], counts[3]);
            System.out.println();

        } catch (IOException e) {
            System.err.println("Error clearing training data: " + e.getMessage());
        }
    }

    private static void generateUrlList() {
        // Popular Wikipedia articles - good variety of content
        String[] topics = {
            "Computer_vision", "Machine_learning", "Neural_network",
            "Image_processing", "OpenCV", "JavaFX",
            "Perspective_(graphical)", "Homography_(computer_vision)",
            "Camera_matrix", "Pinhole_camera_model",
            "Affine_transformation", "Projective_geometry",
            "Feature_detection_(computer_vision)", "Edge_detection",
            "Canny_edge_detector", "Sobel_operator",
            "Gaussian_blur", "Convolution", "Kernel_(image_processing)",
            "Digital_image_processing", "Computer_graphics",
            "3D_projection", "Rotation_matrix", "Euler_angles",
            "Quaternion", "Linear_algebra", "Matrix_(mathematics)",
            "Eigenvalue", "Singular_value_decomposition",
            "Principal_component_analysis", "Deep_learning",
            "Convolutional_neural_network", "Backpropagation",
            "Gradient_descent", "Overfitting", "Regularization",
            "Batch_normalization", "Dropout_(neural_networks)",
            "Activation_function", "ReLU", "Sigmoid_function",
            "Softmax_function", "Cross_entropy", "Loss_function",
            "Python_(programming_language)", "Java_(programming_language)",
            "C%2B%2B", "JavaScript", "HTML", "CSS",
            "World_Wide_Web", "HTTP", "URL", "DNS",
            "Internet", "Computer_network", "TCP/IP",
            "Encryption", "Cryptography", "Hash_function",
            "Algorithm", "Data_structure", "Array",
            "Linked_list", "Binary_tree", "Graph_(abstract_data_type)",
            "Sorting_algorithm", "Search_algorithm", "Big_O_notation",
            "Recursion", "Dynamic_programming", "Greedy_algorithm",
            "Object-oriented_programming", "Functional_programming",
            "Design_pattern", "Software_engineering", "Agile_software_development",
            "Git", "Version_control", "Continuous_integration",
            "Unit_testing", "Test-driven_development", "Debugging",
            "Compiler", "Interpreter_(computing)", "Virtual_machine",
            "Operating_system", "Linux", "macOS", "Windows",
            "File_system", "Memory_management", "Process_(computing)",
            "Thread_(computing)", "Concurrency_(computer_science)",
            "Parallel_computing", "Distributed_computing", "Cloud_computing",
            "Database", "SQL", "NoSQL", "Relational_database",
            "Artificial_intelligence", "Expert_system", "Natural_language_processing",
            "Speech_recognition", "Computer_science", "Information_theory"
        };

        for (int i = 0; i < Math.min(topics.length, totalPages); i++) {
            urls.add("https://en.wikipedia.org/wiki/" + topics[i]);
        }

        // If we need more pages, generate random article URLs
        while (urls.size() < totalPages) {
            urls.add("https://en.wikipedia.org/wiki/Special:Random");
        }
    }

    @Override
    public void start(Stage primaryStage) {
        instance = this;  // For shutdown hook
        completionLatch = new CountDownLatch(1);
        startTimeMillis = System.currentTimeMillis();
        staticStartTimeMillis = startTimeMillis;  // Copy to static for shutdown hook access

        // Initialize random with worker-specific seed for reproducibility
        // Each worker gets a different stream of random numbers
        random = new Random(42 + workerId * 1000000L);

        // Generate timestamp for output filenames (local time, filesystem-safe)
        java.time.LocalDateTime now = java.time.LocalDateTime.now();
        runTimestamp = now.format(java.time.format.DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss"));
        // Add worker ID suffix if in multi-worker mode
        if (workerId > 0) {
            runTimestamp = runTimestamp + "_w" + workerId;
        }

        // Initialize video writers (AVI with MJPG codec - widely supported)
        int fourcc = VideoWriter.fourcc('M', 'J', 'P', 'G');
        int videoFps = fps > 0 ? fps : 30;  // Use configured fps (default 30)
        Size videoSize = new Size(imageWidth, imageHeight);

        // Human videos (sorted by rotation angle for easy viewing)
        humanContentWriter = openVideoWriter(outputDir + "/videos/human_content_" + runTimestamp + ".avi", fourcc, videoFps, videoSize);
        humanWhiteWriter = openVideoWriter(outputDir + "/videos/human_white_" + runTimestamp + ".avi", fourcc, videoFps, videoSize);

        // Training videos (random order to prevent temporal learning)
        trainingContentWriter = openVideoWriter(outputDir + "/videos/training_content_" + runTimestamp + ".avi", fourcc, videoFps, videoSize);
        trainingWhiteWriter = openVideoWriter(outputDir + "/videos/training_white_" + runTimestamp + ".avi", fourcc, videoFps, videoSize);

        // Initialize video labels file (JSON array, one entry per frame)
        try {
            videoLabelsWriter = new PrintWriter(outputDir + "/videos/training_labels_" + runTimestamp + ".json");
            videoLabelsWriter.println("[");
        } catch (IOException e) {
            System.err.println("Warning: Could not open video labels file: " + e.getMessage());
            videoLabelsWriter = null;
        }

        // Print progress header
        System.out.print("Generating samples ");

        // Create WebView at higher resolution for better content capture
        webView = new WebView();
        webView.setPrefSize(renderWidth, renderHeight);
        webView.setMinSize(renderWidth, renderHeight);
        webView.setMaxSize(renderWidth, renderHeight);

        webEngine = webView.getEngine();

        // Set up load listener
        webEngine.getLoadWorker().stateProperty().addListener((obs, oldState, newState) -> {
            if (newState == Worker.State.SUCCEEDED) {
                // Extract page title from resolved URL (after redirect from Special:Random)
                currentPageTitle = extractPageTitle(webEngine.getLocation());

                // Randomly choose dark mode for this page (50% chance)
                currentPageDarkMode = random.nextBoolean();
                if (currentPageDarkMode) {
                    darkModePages++;
                    applyDarkMode();
                } else {
                    lightModePages++;
                }

                // Randomly scroll down the page (0-3 page heights) to avoid always showing logo
                int scrollPages = random.nextInt(4);  // 0, 1, 2, or 3 pages down
                if (scrollPages > 0) {
                    int scrollAmount = scrollPages * renderHeight;
                    webEngine.executeScript("window.scrollTo(0, " + scrollAmount + ");");
                }

                // Wait for scroll to render, then capture and process on background thread
                new Thread(() -> {
                    try {
                        Thread.sleep(400);  // Allow scroll and page to render (reduced for speed)
                        // Snapshot must be on FX thread, but processing will be on background
                        Platform.runLater(this::captureSnapshot);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                }).start();
            } else if (newState == Worker.State.FAILED) {
                System.err.println("Failed to load: " + webEngine.getLocation());
                loadNextPage();
            }
        });

        // Set up the scene for WebView
        StackPane root = new StackPane(webView);
        Scene scene = new Scene(root, renderWidth, renderHeight);
        primaryStage.setScene(scene);
        primaryStage.setTitle("Source Page");
        if (headless) {
            // Minimize window but still show - WebView needs Stage to be shown to render
            primaryStage.setX(0);
            primaryStage.setY(0);
            primaryStage.setIconified(true);  // Minimize to dock
        } else {
            primaryStage.setX(100);
            primaryStage.setY(100);
        }
        primaryStage.show();  // Must show for WebView to render content

        // Create preview window for distorted output (positioned to the right)
        // Skip if --headless mode for faster processing
        if (!headless) {
            previewStage = new Stage();
            previewImageView = new javafx.scene.image.ImageView();
            previewImageView.setFitWidth(imageWidth);
            previewImageView.setFitHeight(imageHeight);
            previewImageView.setPreserveRatio(true);
            StackPane previewRoot = new StackPane(previewImageView);
            previewRoot.setStyle("-fx-background-color: #333;");
            Scene previewScene = new Scene(previewRoot, imageWidth, imageHeight);
            previewStage.setScene(previewScene);
            previewStage.setTitle("Distorted Output (SPACE=pause)");
            previewStage.setX(100 + renderWidth + 20);  // Position to the right of main window
            previewStage.setY(100);

            // Add spacebar handler to toggle pause
            previewScene.setOnKeyPressed(event -> {
                if (event.getCode() == javafx.scene.input.KeyCode.SPACE) {
                    paused = !paused;
                    previewStage.setTitle(paused ? "Distorted Output (PAUSED)" : "Distorted Output (SPACE=pause)");
                    if (paused) {
                        System.out.println(" [PAUSED]");
                    } else {
                        System.out.print(" [RESUMED] ");
                    }
                }
            });

            previewStage.show();
            previewStage.requestFocus();  // Make sure it can receive key events
        }

        // Start loading pages
        loadNextPage();
    }

    private void loadNextPage() {
        if (shuttingDown) return;

        // Check if we've hit any limits (time or frames)
        if (hasReachedLimit()) {
            if (limitMessage != null) {
                System.out.println("\n" + limitMessage);
            }
            System.out.println("Stopping generation.");
            cleanup();
            completionLatch.countDown();
            Platform.exit();
            return;
        }

        // In continuous mode (pages < 0), always load a new random Wikipedia page
        if (totalPages < 0) {
            currentSampleIndex = 0;
            String url = "https://en.wikipedia.org/wiki/Special:Random";
            System.out.print("[Page " + (currentUrlIndex + 1) + "] ");
            System.out.flush();
            webEngine.load(url);
            return;
        }

        // Non-continuous: check if we've processed all pages
        if (currentUrlIndex >= urls.size()) {
            System.out.println("\nGeneration complete!");
            cleanup();
            completionLatch.countDown();
            Platform.exit();
            return;
        }

        currentSampleIndex = 0;
        String url = urls.get(currentUrlIndex);
        System.out.print("[Page " + (currentUrlIndex + 1) + "/" + urls.size() + "] ");
        System.out.flush();
        webEngine.load(url);
    }

    private volatile boolean limitReached = false;
    private volatile String limitMessage = null;

    /**
     * Check if we've reached any configured limits (time or frames).
     */
    private boolean hasReachedLimit() {
        if (limitReached) return true;

        // Check time limit
        if (maxSeconds > 0) {
            long elapsedSeconds = (System.currentTimeMillis() - startTimeMillis) / 1000;
            if (elapsedSeconds >= maxSeconds) {
                limitReached = true;
                limitMessage = String.format("Time limit reached: %d seconds", maxSeconds);
                return true;
            }
        }

        // Check frame limit
        if (maxFrames > 0 && totalSamplesGenerated >= maxFrames) {
            limitReached = true;
            limitMessage = String.format("Frame limit reached: %d frames", maxFrames);
            return true;
        }

        return false;
    }

    /**
     * Capture snapshot on FX thread, then hand off to background thread for processing.
     */
    private void captureSnapshot() {
        if (shuttingDown) return;

        try {
            System.out.println("[DEBUG] captureSnapshot: capturing WebView " + webView.getWidth() + "x" + webView.getHeight());
            // Capture the WebView as an image (must be on FX thread)
            WritableImage fxImage = webView.snapshot(new SnapshotParameters(), null);
            System.out.println("[DEBUG] fxImage: " + (int)fxImage.getWidth() + "x" + (int)fxImage.getHeight());
            BufferedImage bufferedImage = SwingFXUtils.fromFXImage(fxImage, null);
            System.out.println("[DEBUG] bufferedImage: " + bufferedImage.getWidth() + "x" + bufferedImage.getHeight());

            // Process on background thread to keep UI responsive
            final int pageIdx = currentUrlIndex;
            final String pageTitle = currentPageTitle;  // Capture for thread safety
            new Thread(() -> processCapture(bufferedImage, pageIdx, pageTitle)).start();

        } catch (Exception e) {
            System.err.println("Error capturing page: " + e.getMessage());
            e.printStackTrace();
            currentUrlIndex++;
            Platform.runLater(this::loadNextPage);
        }
    }

    /**
     * Process captured image on background thread (heavy OpenCV work).
     */
    private void processCapture(BufferedImage bufferedImage, int pageIdx, String pageTitle) {
        if (shuttingDown) return;

        System.out.println("[DEBUG] processCapture: pageIdx=" + pageIdx + ", title=" + pageTitle);
        System.out.println("[DEBUG] BufferedImage: " + bufferedImage.getWidth() + "x" + bufferedImage.getHeight() + ", type=" + bufferedImage.getType());

        try {
            // Convert to OpenCV Mat
            Mat originalMat = bufferedImageToMat(bufferedImage);
            System.out.println("[DEBUG] originalMat: " + originalMat.cols() + "x" + originalMat.rows() + ", type=" + originalMat.type() + ", channels=" + originalMat.channels());

            // First, add the original unwarped page as frame 0 (identity transform)
            addOriginalPageFrame(originalMat, pageIdx, pageTitle);

            // Generate multiple distorted versions
            for (int i = 0; i < samplesPerPage && !shuttingDown && !hasReachedLimit(); i++) {
                generateSample(originalMat, pageIdx, i, pageTitle);
                currentSampleIndex++;
                totalSamplesGenerated++;
            }

            originalMat.release();

            // Flush buffered frames to videos (sorted for human, random for training)
            flushPageFrameBuffer();

            // Print sample count for this page
            System.out.println(pageTitle + ": " + (samplesPerPage + 1) + " samples (1 original + " + samplesPerPage + " distorted) -> " + totalSamplesGenerated + " total");

            // Update static variables and write stats after each page (for crash resilience)
            updateStaticStats();
            if (workerId > 0) {
                writeWorkerStatsIncremental();
            }

        } catch (Exception e) {
            System.err.println("Error processing page: " + e.getMessage());
            e.printStackTrace();
        }

        // Move to next page (must be on FX thread)
        currentUrlIndex++;
        updateStaticStats();  // Update again after index increment
        Platform.runLater(this::loadNextPage);
    }

    /**
     * Add the original unwarped page as frame 0 with identity transform.
     * The page is centered in the frame with correct 8.5x11 aspect ratio.
     */
    private void addOriginalPageFrame(Mat original, int pageIndex, String pageTitle) {
        String sampleId = String.format("%s_%05d_orig", pageTitle, pageIndex);
        System.out.println("[DEBUG] addOriginalPageFrame: " + sampleId);

        // Create output frame with random noise background (same as distorted samples)
        Mat outputFrame = new Mat(imageHeight, imageWidth, CvType.CV_8UC3);
        Core.randu(outputFrame, 0, 256);

        // Calculate paper size to fit in frame with correct 8.5x11 aspect ratio
        // Paper fills full height, centered horizontally
        double paperAspect = 8.5 / 11.0;  // width/height for letter paper
        double paperHeight = imageHeight;  // Full height
        double paperWidth = paperHeight * paperAspect;

        // Center the paper horizontally, top-aligned (y=0)
        double paperX = (imageWidth - paperWidth) / 2.0;
        double paperY = 0;

        // Resize original to paper dimensions (maintaining content)
        Mat resizedContent = new Mat();
        Imgproc.resize(original, resizedContent, new Size((int)paperWidth, (int)paperHeight));

        // Copy resized content onto the noise background
        Rect paperRect = new Rect((int)paperX, (int)paperY, (int)paperWidth, (int)paperHeight);
        resizedContent.copyTo(outputFrame.submat(paperRect));
        resizedContent.release();

        System.out.println("[DEBUG] paper: " + (int)paperWidth + "x" + (int)paperHeight + " at (" + (int)paperX + "," + (int)paperY + ")");

        // Create white page version (white rectangle on noise background)
        Mat whiteFrame = new Mat(imageHeight, imageWidth, CvType.CV_8UC3);
        Core.randu(whiteFrame, 0, 256);
        Mat whitePaper = new Mat((int)paperHeight, (int)paperWidth, CvType.CV_8UC3, new Scalar(255, 255, 255));
        whitePaper.copyTo(whiteFrame.submat(paperRect));
        whitePaper.release();

        // Save images
        String contentPath = outputDir + "/images/" + sampleId + "_normal.jpg";
        String whitePath = outputDir + "/images/" + sampleId + "_white.jpg";
        boolean wrote1 = Imgcodecs.imwrite(contentPath, outputFrame);
        boolean wrote2 = Imgcodecs.imwrite(whitePath, whiteFrame);
        System.out.println("[DEBUG] Wrote images: content=" + wrote1 + ", white=" + wrote2);

        // Identity homography (3x3 identity matrix)
        Mat H_identity = Mat.eye(3, 3, CvType.CV_64F);
        Mat H_inv_identity = Mat.eye(3, 3, CvType.CV_64F);

        // Corners of the paper rectangle (where the document actually is)
        Point[] paperCorners = {
            new Point(paperX, paperY),                          // TL
            new Point(paperX + paperWidth, paperY),             // TR
            new Point(paperX + paperWidth, paperY + paperHeight), // BR
            new Point(paperX, paperY + paperHeight)             // BL
        };

        // Save labels
        String labelsPath = outputDir + "/labels/" + sampleId + ".json";
        saveLabelsJson(H_identity, H_inv_identity, paperCorners, labelsPath);

        // Track for manifest
        generatedSampleIds.add(sampleId);

        // Build label JSON for video
        String labelJson = buildLabelJson(sampleId, H_identity, H_inv_identity, paperCorners);

        // Add to frame buffer with rotation = -1 (sorts first in human video)
        synchronized (bufferLock) {
            pageFrameBuffer.add(new FrameData(outputFrame, whiteFrame, -1.0, labelJson));
        }

        // Release mats (they were cloned in FrameData)
        outputFrame.release();
        whiteFrame.release();
        H_identity.release();
        H_inv_identity.release();

        totalSamplesGenerated++;
    }

    private void generateSample(Mat original, int pageIndex, int sampleIndex, String pageTitle) {
        long t0, t1;

        // Include page title in sample ID for easier identification
        String sampleId = String.format("%s_%05d_%02d", pageTitle, pageIndex, sampleIndex);

        // Source image is renderWidth x renderHeight, output is imageWidth x imageHeight
        // We'll pick a rectangle from the source with 8.5x11 aspect ratio to map to the destination

        // Source: a rectangle from the rendered webpage with 8.5x11 aspect ratio
        // This ensures the white page shape is correct and content isn't stretched
        double srcAspect = 8.5 / 11.0;  // width/height for letter paper

        // Random position for source rectangle (with margin from edges)
        double margin = 0.05;
        double maxSrcHeight = renderHeight * (1.0 - 2 * margin);
        double maxSrcWidth = renderWidth * (1.0 - 2 * margin);

        // Pick a random height, then calculate width to maintain aspect ratio
        double srcHeight = maxSrcHeight * (0.5 + random.nextDouble() * 0.5);  // 50-100% of available height
        double srcWidth = srcHeight * srcAspect;

        // If width exceeds available space, scale down
        if (srcWidth > maxSrcWidth) {
            srcWidth = maxSrcWidth;
            srcHeight = srcWidth / srcAspect;
        }

        // Random position within the source image
        double srcX1 = renderWidth * margin + random.nextDouble() * (renderWidth * (1 - 2 * margin) - srcWidth);
        double srcY1 = renderHeight * margin + random.nextDouble() * (renderHeight * (1 - 2 * margin) - srcHeight);
        double srcX2 = srcX1 + srcWidth;
        double srcY2 = srcY1 + srcHeight;

        // Source corners - rectangle with 8.5x11 aspect ratio
        Point[] srcCorners = {
            new Point(srcX1, srcY1),  // TL
            new Point(srcX2, srcY1),  // TR
            new Point(srcX2, srcY2),  // BR
            new Point(srcX1, srcY2)   // BL
        };

        // Destination: simulate paper on a table viewed from a tilted camera
        // The paper appears as a distorted quadrilateral in the output image

        double paperRotation = random.nextDouble() * 2 * Math.PI;  // Paper orientation 0-360° (realistic)
        double cameraTiltX = (random.nextDouble() - 0.5) * 0.3;    // Camera tilt left/right (±0.15, was ±0.3)
        double cameraTiltY = (random.nextDouble() - 0.5) * 0.3;    // Camera tilt forward/back (±0.15, was ±0.3)
        // Linear random for scale (was squared/biased toward small)
        double paperScale = 0.20 + random.nextDouble() * 0.50;     // Paper size 20-70% of frame (was 5-80%)

        // Random position - paper mostly on-screen
        // Range allows paper center from 10% to 90% of image dimensions (was -20% to 120%)
        double centerX = imageWidth * (0.1 + random.nextDouble() * 0.8);
        double centerY = imageHeight * (0.1 + random.nextDouble() * 0.8);

        // Base paper size in output - maintain 8.5x11 (letter paper) proportions
        // paperScale controls what fraction of image height the paper takes
        double paperAspect = 8.5 / 11.0;  // width/height ratio for letter paper
        double halfH = imageHeight * paperScale / 2.0;  // Height based on scale
        double halfW = halfH * paperAspect;              // Width maintains letter proportions

        // Paper corners relative to center (before transforms): TL, TR, BR, BL
        double[][] paperCorners = {
            {-halfW, -halfH},
            {halfW, -halfH},
            {halfW, halfH},
            {-halfW, halfH}
        };

        Point[] dstCorners = new Point[4];
        for (int i = 0; i < 4; i++) {
            double x = paperCorners[i][0];
            double y = paperCorners[i][1];

            // 1. Rotate paper on table
            double cosR = Math.cos(paperRotation);
            double sinR = Math.sin(paperRotation);
            double rx = x * cosR - y * sinR;
            double ry = x * sinR + y * cosR;

            // 2. Apply camera perspective (foreshortening)
            // Each axis is scaled independently based on position along that axis
            // This creates trapezoid distortion without changing aspect ratio uniformly
            double scaleX = 1.0 + cameraTiltX * (rx / halfW);
            double scaleY = 1.0 + cameraTiltY * (ry / halfH);
            rx *= scaleX;
            ry *= scaleY;

            // 3. Translate to position in output image
            dstCorners[i] = new Point(centerX + rx, centerY + ry);
        }

        // getPerspectiveTransform: maps srcCorners -> dstCorners
        // warpPerspective uses inverse: for each dst pixel, find where it comes from in src
        MatOfPoint2f srcMat = new MatOfPoint2f(srcCorners);
        MatOfPoint2f dstMat = new MatOfPoint2f(dstCorners);

        // Compute homography
        t0 = System.nanoTime();
        Mat H = Imgproc.getPerspectiveTransform(srcMat, dstMat);
        Mat H_inv = H.inv();
        t1 = System.nanoTime();
        if (showTiming) timeHomography += (t1 - t0) / 1000;

        // Create mask for warping
        Mat mask = Mat.ones(original.size(), CvType.CV_8UC1);
        mask.setTo(new Scalar(255));

        // Create random noise background to avoid the network learning border patterns
        t0 = System.nanoTime();
        Mat distorted = new Mat(imageHeight, imageWidth, CvType.CV_8UC3);
        Core.randu(distorted, 0, 256);
        t1 = System.nanoTime();
        if (showTiming) timeRandu1 += (t1 - t0) / 1000;

        // Warp original onto noise background
        t0 = System.nanoTime();
        Mat warped = new Mat();
        Mat warpedMask = new Mat();
        Imgproc.warpPerspective(original, warped, H, new Size(imageWidth, imageHeight));
        Imgproc.warpPerspective(mask, warpedMask, H, new Size(imageWidth, imageHeight));
        t1 = System.nanoTime();
        if (showTiming) timeWarp1 += (t1 - t0) / 1000;

        // Randomly apply shadows (50% chance) - save parameters for white page
        int shadowType = -1;  // -1 = no shadow
        long shadowSeed = random.nextLong();  // Save seed to reproduce same shadow
        t0 = System.nanoTime();
        if (random.nextBoolean()) {
            shadowType = random.nextInt(4);
            Random shadowRandom = new Random(shadowSeed);
            applyShadowWithParams(warped, warpedMask, shadowType, shadowRandom);
        }
        t1 = System.nanoTime();
        if (showTiming) timeShadow += (t1 - t0) / 1000;

        t0 = System.nanoTime();
        warped.copyTo(distorted, warpedMask);
        t1 = System.nanoTime();
        if (showTiming) timeCopyTo1 += (t1 - t0) / 1000;
        warped.release();
        warpedMask.release();
        mask.release();

        // Save distorted image (JPEG for speed - PNG compression is too slow)
        t0 = System.nanoTime();
        String distortedPath = outputDir + "/images/" + sampleId + "_normal.jpg";
        Imgcodecs.imwrite(distortedPath, distorted);
        t1 = System.nanoTime();
        if (showTiming) timeImwrite1 += (t1 - t0) / 1000;

        // Create and save white page version (same transform, white rectangle on noise)
        // White page uses same size as original - the homography H maps the srcCorners region
        // from this image to the dstCorners (8.5x11 paper shape) in the output
        Mat whitePage = new Mat(original.size(), CvType.CV_8UC3, new Scalar(255, 255, 255));
        Mat whiteMask = Mat.ones(original.size(), CvType.CV_8UC1);
        whiteMask.setTo(new Scalar(255));

        t0 = System.nanoTime();
        Mat whiteDistorted = new Mat(imageHeight, imageWidth, CvType.CV_8UC3);
        Core.randu(whiteDistorted, 0, 256);  // Fresh random noise background
        t1 = System.nanoTime();
        if (showTiming) timeRandu2 += (t1 - t0) / 1000;

        t0 = System.nanoTime();
        Mat whiteWarped = new Mat();
        Mat whiteMaskWarped = new Mat();
        Imgproc.warpPerspective(whitePage, whiteWarped, H, new Size(imageWidth, imageHeight));
        Imgproc.warpPerspective(whiteMask, whiteMaskWarped, H, new Size(imageWidth, imageHeight));
        t1 = System.nanoTime();
        if (showTiming) timeWarp2 += (t1 - t0) / 1000;

        // Apply same hard shadow to white page (only the 100% black parts)
        if (shadowType >= 0) {
            Random shadowRandom = new Random(shadowSeed);
            applyHardShadowOnly(whiteWarped, whiteMaskWarped, shadowType, shadowRandom);
        }

        t0 = System.nanoTime();
        whiteWarped.copyTo(whiteDistorted, whiteMaskWarped);
        t1 = System.nanoTime();
        if (showTiming) timeCopyTo2 += (t1 - t0) / 1000;

        t0 = System.nanoTime();
        String whitePath = outputDir + "/images/" + sampleId + "_white.jpg";
        Imgcodecs.imwrite(whitePath, whiteDistorted);
        t1 = System.nanoTime();
        if (showTiming) timeImwrite2 += (t1 - t0) / 1000;
        whitePage.release();
        whiteMask.release();
        whiteWarped.release();
        whiteMaskWarped.release();

        // Save labels as JSON (homography, inverse, and 4 corners)
        t0 = System.nanoTime();
        String labelsPath = outputDir + "/labels/" + sampleId + ".json";
        saveLabelsJson(H, H_inv, dstCorners, labelsPath);
        t1 = System.nanoTime();
        if (showTiming) timeLabelsJson += (t1 - t0) / 1000;

        // Track generated sample ID for manifest
        generatedSampleIds.add(sampleId);

        // Build label JSON for video labels file
        String labelJson = buildLabelJson(sampleId, H, H_inv, dstCorners);

        // Buffer frames for deferred video writing (sorted vs random order)
        t0 = System.nanoTime();
        synchronized (bufferLock) {
            pageFrameBuffer.add(new FrameData(distorted, whiteDistorted, paperRotation, labelJson));
        }
        t1 = System.nanoTime();
        if (showTiming) timeVideoBuffer += (t1 - t0) / 1000;

        // Release whiteDistorted - we cloned it in FrameData
        whiteDistorted.release();

        // Print progress dot
        if (totalSamplesGenerated > 0 && totalSamplesGenerated % progressInterval == 0) {
            System.out.print(".");
            System.out.flush();
        }

        // Update preview window with distorted image
        t0 = System.nanoTime();
        updatePreview(distorted, sampleIndex);
        t1 = System.nanoTime();
        if (showTiming) timePreview += (t1 - t0) / 1000;

        if (showTiming) timingSamples++;

        // Delay if fps > 0 (slow mode for viewing)
        if (fps > 0) {
            try {
                Thread.sleep(1000 / fps);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        // Wait while paused
        while (paused && !shuttingDown) {
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }

        // Cleanup
        distorted.release();
        H.release();
        H_inv.release();
        srcMat.release();
        dstMat.release();
    }

    private double randomPerturb(double center, double maxPerturbation) {
        return center + (random.nextDouble() * 2 - 1) * maxPerturbation;
    }

    private double randomOffset(double maxOffset) {
        return (random.nextDouble() * 2 - 1) * maxOffset;
    }

    /**
     * Apply shadow effects with specific type and random generator.
     * Used to reproduce the same shadow on both content and white page versions.
     */
    private void applyShadowWithParams(Mat image, Mat mask, int shadowType, Random rng) {
        int width = image.cols();
        int height = image.rows();

        switch (shadowType) {
            case 0:  // Gradient shadow (like light from one side)
                applyGradientShadow(image, mask, width, height, rng, false);
                break;
            case 1:  // Edge shadow (darker near edges)
                applyEdgeShadow(image, mask, width, height, rng, false);
                break;
            case 2:  // Cast shadow (stripe across the image, sometimes completely black)
                applyCastShadow(image, mask, width, height, rng, false);
                break;
            case 3:  // Corner shadow (black out 1-2 corners, like a hand covering)
                applyCornerShadow(image, mask, width, height, rng, false);
                break;
        }
    }

    /**
     * Apply only the 100% black (hard) portions of shadows to the white page.
     * Gradient and edge shadows have no hard black areas, so they're skipped.
     */
    private void applyHardShadowOnly(Mat image, Mat mask, int shadowType, Random rng) {
        int width = image.cols();
        int height = image.rows();

        switch (shadowType) {
            case 0:  // Gradient shadow - no hard black, skip
            case 1:  // Edge shadow - no hard black, skip
                break;
            case 2:  // Cast shadow - apply only hard black parts
                applyCastShadow(image, mask, width, height, rng, true);
                break;
            case 3:  // Corner shadow - always hard black
                applyCornerShadow(image, mask, width, height, rng, true);
                break;
        }
    }

    private void applyGradientShadow(Mat image, Mat mask, int width, int height, Random rng, boolean hardOnly) {
        if (hardOnly) return;  // Gradient shadows have no hard black areas

        // Random angle for light direction
        double angle = rng.nextDouble() * 2 * Math.PI;
        double cosA = Math.cos(angle);
        double sinA = Math.sin(angle);

        // Shadow intensity (30-70% darkening at max)
        double intensity = 0.3 + rng.nextDouble() * 0.4;

        byte[] pixelData = new byte[3];
        byte[] maskData = new byte[1];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                mask.get(y, x, maskData);
                if ((maskData[0] & 0xFF) > 128) {  // Only affect masked area
                    // Calculate position along gradient direction (-1 to 1)
                    double nx = (x - width / 2.0) / (width / 2.0);
                    double ny = (y - height / 2.0) / (height / 2.0);
                    double gradientPos = (nx * cosA + ny * sinA + 1) / 2;  // 0 to 1

                    // Apply darkening
                    double darken = 1.0 - intensity * gradientPos;

                    image.get(y, x, pixelData);
                    for (int c = 0; c < 3; c++) {
                        int val = (int) ((pixelData[c] & 0xFF) * darken);
                        pixelData[c] = (byte) Math.max(0, Math.min(255, val));
                    }
                    image.put(y, x, pixelData);
                }
            }
        }
    }

    private void applyEdgeShadow(Mat image, Mat mask, int width, int height, Random rng, boolean hardOnly) {
        if (hardOnly) return;  // Edge shadows have no hard black areas

        // Vignette-style shadow, darker at corners/edges
        double intensity = 0.2 + rng.nextDouble() * 0.3;
        double cx = width / 2.0;
        double cy = height / 2.0;
        double maxDist = Math.sqrt(cx * cx + cy * cy);

        byte[] pixelData = new byte[3];
        byte[] maskData = new byte[1];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                mask.get(y, x, maskData);
                if ((maskData[0] & 0xFF) > 128) {
                    double dist = Math.sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
                    double darken = 1.0 - intensity * (dist / maxDist);

                    image.get(y, x, pixelData);
                    for (int c = 0; c < 3; c++) {
                        int val = (int) ((pixelData[c] & 0xFF) * darken);
                        pixelData[c] = (byte) Math.max(0, Math.min(255, val));
                    }
                    image.put(y, x, pixelData);
                }
            }
        }
    }

    private void applyCastShadow(Mat image, Mat mask, int width, int height, Random rng, boolean hardOnly) {
        // Simulate a shadow cast by an external object (like a person's hand or overhead object)
        double angle = rng.nextDouble() * Math.PI;  // Shadow angle
        double shadowWidth = 0.1 + rng.nextDouble() * 0.3;  // Shadow width (10-40% of image)
        double shadowPos = rng.nextDouble();  // Where the shadow starts

        // 30% chance of completely black shadow (total occlusion)
        boolean hardShadow = rng.nextDouble() < 0.3;
        double intensity = hardShadow ? 1.0 : (0.3 + rng.nextDouble() * 0.4);

        // If hardOnly mode and this isn't a hard shadow, skip
        if (hardOnly && !hardShadow) return;

        double cosA = Math.cos(angle);
        double sinA = Math.sin(angle);

        byte[] pixelData = new byte[3];
        byte[] maskData = new byte[1];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                mask.get(y, x, maskData);
                if ((maskData[0] & 0xFF) > 128) {
                    // Project point onto shadow line
                    double nx = (double) x / width;
                    double ny = (double) y / height;
                    double proj = nx * cosA + ny * sinA;

                    // Check if point is in shadow band
                    double distFromShadowCenter = Math.abs(proj - shadowPos);
                    if (distFromShadowCenter < shadowWidth / 2) {
                        double darken;
                        if (hardShadow) {
                            // Hard shadow: black in center, slight soft edge
                            double edgeFactor = distFromShadowCenter / (shadowWidth / 2);
                            darken = edgeFactor < 0.8 ? 0.0 : (edgeFactor - 0.8) / 0.2;
                        } else {
                            // Soft edge shadow
                            double edgeFactor = distFromShadowCenter / (shadowWidth / 2);
                            darken = 1.0 - intensity * (1.0 - edgeFactor * edgeFactor);
                        }

                        image.get(y, x, pixelData);
                        for (int c = 0; c < 3; c++) {
                            int val = (int) ((pixelData[c] & 0xFF) * darken);
                            pixelData[c] = (byte) Math.max(0, Math.min(255, val));
                        }
                        image.put(y, x, pixelData);
                    }
                }
            }
        }
    }

    private void applyCornerShadow(Mat image, Mat mask, int width, int height, Random rng, boolean hardOnly) {
        // Black out one or two corners completely - simulates hand/object covering part of document
        int numCorners = 1 + rng.nextInt(2);  // 1 or 2 corners
        double coverageRadius = 0.2 + rng.nextDouble() * 0.3;  // 20-50% of image diagonal

        byte[] pixelData = new byte[3];
        byte[] maskData = new byte[1];

        // Corner positions: 0=TL, 1=TR, 2=BR, 3=BL
        int[] corners = new int[]{0, 1, 2, 3};
        // Shuffle to pick random corners
        for (int i = corners.length - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int temp = corners[i];
            corners[i] = corners[j];
            corners[j] = temp;
        }

        double diagonal = Math.sqrt(width * width + height * height);
        double maxDist = diagonal * coverageRadius;

        for (int c = 0; c < numCorners; c++) {
            double cx, cy;
            switch (corners[c]) {
                case 0: cx = 0; cy = 0; break;           // TL
                case 1: cx = width; cy = 0; break;       // TR
                case 2: cx = width; cy = height; break;  // BR
                default: cx = 0; cy = height; break;     // BL
            }

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    mask.get(y, x, maskData);
                    if ((maskData[0] & 0xFF) > 128) {
                        double dist = Math.sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
                        if (dist < maxDist) {
                            // Soft edge at boundary, black inside
                            double edgeFactor = dist / maxDist;
                            double darken = edgeFactor < 0.7 ? 0.0 : (edgeFactor - 0.7) / 0.3;

                            image.get(y, x, pixelData);
                            for (int ch = 0; ch < 3; ch++) {
                                int val = (int) ((pixelData[ch] & 0xFF) * darken);
                                pixelData[ch] = (byte) Math.max(0, Math.min(255, val));
                            }
                            image.put(y, x, pixelData);
                        }
                    }
                }
            }
        }
    }

    /**
     * Apply dark mode CSS to the current page via JavaScript.
     * This inverts colors and adjusts images to simulate dark mode.
     */
    private void applyDarkMode() {
        String darkModeJS = """
            (function() {
                var style = document.createElement('style');
                style.textContent = `
                    html {
                        background-color: #1a1a2e !important;
                    }
                    body {
                        background-color: #1a1a2e !important;
                        color: #e8e8e8 !important;
                    }
                    * {
                        background-color: inherit !important;
                        color: inherit !important;
                        border-color: #444 !important;
                    }
                    a {
                        color: #6eb5ff !important;
                    }
                    img {
                        opacity: 0.9;
                    }
                    /* Wikipedia specific */
                    .mw-body, .mw-page-container, #content, #bodyContent {
                        background-color: #1a1a2e !important;
                    }
                    .infobox, .navbox, .sidebar, table {
                        background-color: #252540 !important;
                    }
                    code, pre, .mw-code {
                        background-color: #2d2d44 !important;
                    }
                `;
                document.head.appendChild(style);
            })();
            """;
        webEngine.executeScript(darkModeJS);
    }

    /**
     * Update the preview window with the distorted image.
     * Uses CountDownLatch to ensure the update is visible before continuing.
     */
    private void updatePreview(Mat mat, int sampleIndex) {
        if (previewImageView == null) return;

        // Convert BGR Mat to BufferedImage
        int width = mat.cols();
        int height = mat.rows();
        BufferedImage bufferedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        // Copy pixel by pixel (BGR to RGB)
        byte[] data = new byte[width * height * 3];
        mat.get(0, 0, data);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = (y * width + x) * 3;
                int b = data[idx] & 0xFF;
                int g = data[idx + 1] & 0xFF;
                int r = data[idx + 2] & 0xFF;
                bufferedImage.setRGB(x, y, (r << 16) | (g << 8) | b);
            }
        }

        // Convert to JavaFX Image and update on FX thread
        javafx.scene.image.Image fxImage = SwingFXUtils.toFXImage(bufferedImage, null);

        // Use CountDownLatch to wait for the UI update to complete
        java.util.concurrent.CountDownLatch latch = new java.util.concurrent.CountDownLatch(1);
        Platform.runLater(() -> {
            previewImageView.setImage(fxImage);
            // Update title to show sample progress
            String title = String.format("Sample %d/%d (page %d) %s",
                sampleIndex + 1, samplesPerPage, currentUrlIndex + 1,
                paused ? "[PAUSED]" : "[SPACE=pause]");
            previewStage.setTitle(title);
            latch.countDown();
        });

        try {
            // Wait for UI update, with timeout
            latch.await(100, java.util.concurrent.TimeUnit.MILLISECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    private void saveLabelsJson(Mat H, Mat H_inv, Point[] corners, String path) {
        try (PrintWriter writer = new PrintWriter(path)) {
            // Extract 9 values from each 3x3 matrix
            double[] hValues = new double[9];
            double[] hInvValues = new double[9];
            H.get(0, 0, hValues);
            H_inv.get(0, 0, hInvValues);

            writer.println("{");

            // Image dimensions for reference
            writer.println("  \"image_width\": " + imageWidth + ",");
            writer.println("  \"image_height\": " + imageHeight + ",");

            // Raw homography matrix (9 values, row-major)
            writer.println("  \"homography\": [");
            for (int i = 0; i < 9; i++) {
                writer.print("    " + hValues[i]);
                writer.println(i < 8 ? "," : "");
            }
            writer.println("  ],");

            // Inverse homography matrix (9 values, row-major)
            writer.println("  \"inverse\": [");
            for (int i = 0; i < 9; i++) {
                writer.print("    " + hInvValues[i]);
                writer.println(i < 8 ? "," : "");
            }
            writer.println("  ],");

            // Raw corners in pixels: [TL_x, TL_y, TR_x, TR_y, BR_x, BR_y, BL_x, BL_y]
            writer.println("  \"corners\": [");
            for (int i = 0; i < 4; i++) {
                writer.print("    " + corners[i].x + ", " + corners[i].y);
                writer.println(i < 3 ? "," : "");
            }
            writer.println("  ],");

            // Normalized corners in [-1, 1] range (centered on image)
            // x_norm = (x - width/2) / (width/2), y_norm = (y - height/2) / (height/2)
            writer.println("  \"corners_normalized\": [");
            for (int i = 0; i < 4; i++) {
                double xNorm = (corners[i].x - imageWidth / 2.0) / (imageWidth / 2.0);
                double yNorm = (corners[i].y - imageHeight / 2.0) / (imageHeight / 2.0);
                writer.print("    " + xNorm + ", " + yNorm);
                writer.println(i < 3 ? "," : "");
            }
            writer.println("  ]");
            writer.println("}");
        } catch (IOException e) {
            System.err.println("Failed to save labels JSON: " + e.getMessage());
        }
    }

    private Mat bufferedImageToMat(BufferedImage image) {
        // Convert BufferedImage to BGR Mat
        int width = image.getWidth();
        int height = image.getHeight();

        // Ensure image is in correct format
        BufferedImage converted = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
        converted.getGraphics().drawImage(image, 0, 0, null);

        byte[] pixels = ((java.awt.image.DataBufferByte) converted.getRaster().getDataBuffer()).getData();
        Mat mat = new Mat(height, width, CvType.CV_8UC3);
        mat.put(0, 0, pixels);

        return mat;
    }

    /**
     * Helper to open a video writer with error handling.
     */
    private VideoWriter openVideoWriter(String path, int fourcc, int fps, Size size) {
        System.out.println("Opening video writer: " + path);
        VideoWriter writer = new VideoWriter(path, fourcc, fps, size);
        if (!writer.isOpened()) {
            System.err.println("Warning: Could not open video writer: " + path);
            return null;
        }
        return writer;
    }

    /**
     * Helper to close a video writer safely.
     */
    private void closeVideoWriter(VideoWriter writer, String description) {
        if (writer != null) {
            try {
                System.out.println("  Releasing " + description + "...");
                writer.release();
                System.out.println("  " + description + " closed.");
            } catch (Exception e) {
                System.err.println("Error closing " + description + ": " + e.getMessage());
            }
        }
    }

    /**
     * Build JSON label string for a frame.
     */
    private String buildLabelJson(String sampleId, Mat H, Mat H_inv, Point[] dstCorners) {
        double[] hValues = new double[9];
        double[] hInvValues = new double[9];
        H.get(0, 0, hValues);
        H_inv.get(0, 0, hInvValues);

        StringBuilder sb = new StringBuilder();
        sb.append("  {\"frame\": ").append(totalSamplesGenerated);
        sb.append(", \"id\": \"").append(sampleId).append("\"");
        sb.append(", \"h\": [");
        for (int i = 0; i < 9; i++) {
            sb.append(hValues[i]);
            if (i < 8) sb.append(", ");
        }
        sb.append("], \"inv\": [");
        for (int i = 0; i < 9; i++) {
            sb.append(hInvValues[i]);
            if (i < 8) sb.append(", ");
        }
        sb.append("], \"corners\": [");
        for (int i = 0; i < 4; i++) {
            sb.append(dstCorners[i].x).append(", ").append(dstCorners[i].y);
            if (i < 3) sb.append(", ");
        }
        sb.append("], \"dark_mode\": ").append(currentPageDarkMode);
        sb.append("}");
        return sb.toString();
    }

    /**
     * Flush buffered frames to videos:
     * - Human videos: sorted by rotation angle (smooth viewing)
     * - Training videos: original random order (no temporal patterns)
     */
    private void flushPageFrameBuffer() {
        synchronized (bufferLock) {
            if (pageFrameBuffer.isEmpty()) return;

            // Write training videos first (original random order)
            for (FrameData frame : pageFrameBuffer) {
                if (trainingContentWriter != null) {
                    trainingContentWriter.write(frame.contentFrame);
                }
                if (trainingWhiteWriter != null) {
                    trainingWhiteWriter.write(frame.whiteFrame);
                }
                // Write labels in training order
                if (videoLabelsWriter != null) {
                    videoLabelsWriter.println(frame.labelJson + ",");
                }
            }

            // Sort by rotation angle for human videos
            List<FrameData> sortedFrames = new ArrayList<>(pageFrameBuffer);
            sortedFrames.sort((a, b) -> Double.compare(a.paperRotation, b.paperRotation));

            // Write human videos (sorted order)
            for (FrameData frame : sortedFrames) {
                if (humanContentWriter != null) {
                    humanContentWriter.write(frame.contentFrame);
                }
                if (humanWhiteWriter != null) {
                    humanWhiteWriter.write(frame.whiteFrame);
                }
            }

            // Release all frames and clear buffer
            for (FrameData frame : pageFrameBuffer) {
                frame.release();
            }
            pageFrameBuffer.clear();
        }
    }

    private void saveMatrix(Mat matrix, String path) {
        try (PrintWriter writer = new PrintWriter(path)) {
            for (int row = 0; row < matrix.rows(); row++) {
                for (int col = 0; col < matrix.cols(); col++) {
                    double[] val = matrix.get(row, col);
                    writer.print(val[0]);
                    if (col < matrix.cols() - 1) writer.print(" ");
                }
                writer.println();
            }
        } catch (IOException e) {
            System.err.println("Failed to save matrix: " + e.getMessage());
        }
    }

    private void writeManifest() {
        String manifestPath = outputDir + "/manifests/manifest.json";
        if (workerId > 0) {
            // Workers write their own manifest with worker suffix
            manifestPath = outputDir + "/manifests/manifest_w" + workerId + ".json";
        }
        
        try (PrintWriter writer = new PrintWriter(manifestPath)) {
            writer.println("{");
            writer.println("  \"generated\": \"" + new java.util.Date() + "\",");
            if (workerId > 0) {
                writer.println("  \"worker_id\": " + workerId + ",");
            }
            writer.println("  \"imageWidth\": " + imageWidth + ",");
            writer.println("  \"imageHeight\": " + imageHeight + ",");
            writer.println("  \"totalSamples\": " + generatedSampleIds.size() + ",");
            writer.println("  \"samplesPerPage\": " + samplesPerPage + ",");
            writer.println("  \"description\": \"Each label contains 9 homography values and 9 inverse values\",");
            writer.println("  \"samples\": [");

            int sampleNum = 0;
            for (String sampleId : generatedSampleIds) {
                writer.print("    {\"id\": \"" + sampleId + "\", ");
                writer.print("\"image\": \"images/" + sampleId + ".jpg\", ");
                writer.print("\"labels\": \"labels/" + sampleId + ".json\"}");
                sampleNum++;
                if (sampleNum < generatedSampleIds.size()) {
                    writer.println(",");
                } else {
                    writer.println();
                }
            }
            writer.println("  ]");
            writer.println("}");
        } catch (IOException e) {
            System.err.println("Failed to write manifest: " + e.getMessage());
        }
    }

    /**
     * Extract Wikipedia page title from URL and make it filesystem-safe.
     * E.g., "https://en.wikipedia.org/wiki/Quantum_mechanics" -> "Quantum_mechanics"
     * E.g., "https://en.wikipedia.org/wiki/Albert_Einstein" -> "Albert_Einstein"
     */
    private static String extractPageTitle(String url) {
        if (url == null || url.isEmpty()) {
            return "unknown";
        }
        try {
            // Extract the part after /wiki/
            int wikiIdx = url.indexOf("/wiki/");
            if (wikiIdx >= 0) {
                String title = url.substring(wikiIdx + 6);
                // Remove any query parameters or fragments
                int queryIdx = title.indexOf('?');
                if (queryIdx >= 0) title = title.substring(0, queryIdx);
                int fragIdx = title.indexOf('#');
                if (fragIdx >= 0) title = title.substring(0, fragIdx);
                // URL decode (e.g., %27 -> ')
                title = java.net.URLDecoder.decode(title, "UTF-8");
                // Make filesystem-safe: replace problematic chars with underscore
                title = title.replaceAll("[^a-zA-Z0-9_\\-]", "_");
                // Collapse multiple underscores
                title = title.replaceAll("_+", "_");
                // Trim underscores from ends
                title = title.replaceAll("^_+|_+$", "");
                // Limit length to avoid filesystem issues
                if (title.length() > 80) {
                    title = title.substring(0, 80);
                }
                return title.isEmpty() ? "unknown" : title;
            }
        } catch (Exception e) {
            // Fall through to default
        }
        return "unknown";
    }

    /**
     * Spawn multiple worker processes for parallel generation.
     * Each worker runs as a separate JVM with its own WebView.
     * The coordinator waits for all workers to complete.
     */
    private static void spawnWorkers(String[] originalArgs) {
        long startTime = System.currentTimeMillis();

        System.out.println("Homography Training Data Generator - Multi-Worker Mode");
        System.out.println("======================================================");
        System.out.println("Spawning " + numWorkers + " worker processes...");
        if (maxFrames > 0) System.out.println("Frame limit: " + maxFrames);
        if (maxSeconds > 0) System.out.println("Time limit: " + maxSeconds + " seconds");
        System.out.println();

        // Make output directory absolute so workers can find it
        java.io.File outDirFile = new java.io.File(outputDir);
        if (!outDirFile.isAbsolute()) {
            outputDir = outDirFile.getAbsolutePath();
        }

        // Clear existing training data if requested (coordinator handles this for all workers)
        if (clearAllData) {
            clearTrainingData();
        }

        // Create output directory before spawning workers
        try {
            Files.createDirectories(Paths.get(outputDir));
            Files.createDirectories(Paths.get(outputDir, "images"));
            Files.createDirectories(Paths.get(outputDir, "labels"));
            Files.createDirectories(Paths.get(outputDir, "videos"));
            Files.createDirectories(Paths.get(outputDir, "manifests"));
        } catch (IOException e) {
            System.err.println("Failed to create output directory: " + e.getMessage());
            return;
        }
        System.out.println("Output directory: " + outputDir);

        // Build the base command for workers
        // Get the directory where this class file is located to find the r2 script
        String scriptPath;
        try {
            // Get the path to the class file, then find the experiments directory
            String classPath = HomographyTrainingDataGenerator.class.getProtectionDomain()
                    .getCodeSource().getLocation().toURI().getPath();
            java.io.File classDir = new java.io.File(classPath).getParentFile();
            // Look for r2 script - might be in experiments/ or current dir
            java.io.File r2InExperiments = new java.io.File(classDir, "experiments/r2");
            java.io.File r2InParent = new java.io.File(classDir.getParentFile(), "experiments/r2");
            java.io.File r2InCurrent = new java.io.File("experiments/r2");
            java.io.File r2Here = new java.io.File("r2");

            if (r2InExperiments.exists()) {
                scriptPath = r2InExperiments.getAbsolutePath();
            } else if (r2InParent.exists()) {
                scriptPath = r2InParent.getAbsolutePath();
            } else if (r2InCurrent.exists()) {
                scriptPath = r2InCurrent.getAbsolutePath();
            } else if (r2Here.exists()) {
                scriptPath = r2Here.getAbsolutePath();
            } else {
                // Fallback: assume we're in the project root
                scriptPath = "experiments/r2";
            }
        } catch (Exception e) {
            scriptPath = "experiments/r2";  // Fallback
        }
        System.out.println("Using script: " + scriptPath);

        List<Process> workers = new ArrayList<>();
        List<Thread> outputThreads = new ArrayList<>();

        for (int i = 1; i <= numWorkers; i++) {
            final int workerIdLocal = i;

            // Build worker command - pass through all args but replace --workers with --worker-id
            // and add --headless for non-first workers
            List<String> cmd = new ArrayList<>();
            cmd.add(scriptPath);

            // Pass through original args, skipping coordinator-only options
            for (int j = 0; j < originalArgs.length; j++) {
                String arg = originalArgs[j];
                // Skip --workers N and --output DIR (coordinator handles these)
                if (arg.equals("--workers") || arg.equals("--output")) {
                    j++;  // Skip the value too
                    continue;
                }
                // Skip --clear (coordinator already cleared data)
                if (arg.equals("--clear_all_generated_data")) {
                    continue;
                }
                cmd.add(arg);
            }

            // Add absolute output directory so workers can find it regardless of working directory
            cmd.add("--output");
            cmd.add(outputDir);

            // Add worker-specific args
            cmd.add("--worker-id");
            cmd.add(String.valueOf(workerIdLocal));
            cmd.add("--headless");  // All workers run headless

            System.out.println("[Worker " + workerIdLocal + "] Starting: " + String.join(" ", cmd));

            try {
                ProcessBuilder pb = new ProcessBuilder(cmd);
                pb.redirectErrorStream(true);  // Merge stderr into stdout
                Process process = pb.start();
                workers.add(process);

                // Create thread to read and prefix worker output
                Thread outputThread = new Thread(() -> {
                    try (BufferedReader reader = new BufferedReader(
                            new InputStreamReader(process.getInputStream()))) {
                        String line;
                        while ((line = reader.readLine()) != null) {
                            System.out.println("[W" + workerIdLocal + "] " + line);
                        }
                    } catch (IOException e) {
                        System.err.println("[W" + workerIdLocal + "] Output read error: " + e.getMessage());
                    }
                });
                outputThread.setDaemon(true);
                outputThread.start();
                outputThreads.add(outputThread);

                // Small delay between spawns to avoid race conditions
                Thread.sleep(500);

            } catch (Exception e) {
                System.err.println("[Worker " + workerIdLocal + "] Failed to start: " + e.getMessage());
            }
        }

        System.out.println();
        System.out.println("All " + workers.size() + " workers started. Waiting for completion...");
        System.out.println("Press Ctrl+C to stop all workers.");
        System.out.println();

        // Start a monitoring thread to periodically print aggregated stats and check limits
        final long monitorStartTime = startTime;
        final List<Process> workersForMonitor = workers;
        Thread monitorThread = new Thread(() -> {
            while (!Thread.currentThread().isInterrupted()) {
                try {
                    Thread.sleep(2000);  // Check every 2 seconds
                    long totalFrames = getTotalWorkerFrames();
                    long elapsedSeconds = (System.currentTimeMillis() - monitorStartTime) / 1000;
                    double fps = elapsedSeconds > 0 ? totalFrames / (double) elapsedSeconds : 0;

                    // Print stats
                    System.out.printf("[STATS] Total: %,d frames, %.1f fps, %ds elapsed (limit: %d)%n",
                                      totalFrames, fps, elapsedSeconds, maxFrames);
                    System.out.flush();

                    // Check frame limit
                    if (maxFrames > 0 && totalFrames >= maxFrames) {
                        System.out.printf("%n[COORDINATOR] Frame limit reached: %,d frames. Stopping workers...%n", totalFrames);
                        for (Process p : workersForMonitor) {
                            if (p.isAlive()) p.destroy();
                        }
                        break;
                    }

                    // Check time limit
                    if (maxSeconds > 0 && elapsedSeconds >= maxSeconds) {
                        System.out.printf("%n[COORDINATOR] Time limit reached: %d seconds. Stopping workers...%n", elapsedSeconds);
                        for (Process p : workersForMonitor) {
                            if (p.isAlive()) p.destroy();
                        }
                        break;
                    }
                } catch (InterruptedException e) {
                    break;
                }
            }
        });
        monitorThread.setDaemon(true);
        monitorThread.start();

        // Add shutdown hook to gracefully stop all workers on Ctrl+C
        final List<Process> workersFinal = workers;
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("\nCoordinator: Shutting down all workers...");

            // Send SIGTERM to all workers (triggers their shutdown hooks)
            for (Process p : workersFinal) {
                if (p.isAlive()) {
                    p.destroy();  // SIGTERM - allows graceful shutdown
                }
            }

            // Wait for graceful shutdown (max 10 seconds total to allow stats writing)
            System.out.println("Coordinator: Waiting for workers to write stats (up to 10 seconds)...");
            long deadline = System.currentTimeMillis() + 10000;
            while (System.currentTimeMillis() < deadline) {
                boolean anyAlive = false;
                for (Process p : workersFinal) {
                    if (p.isAlive()) {
                        anyAlive = true;
                        break;
                    }
                }
                if (!anyAlive) break;  // All done, no need to wait
                try {
                    Thread.sleep(200);  // Check every 200ms
                } catch (InterruptedException e) {
                    break;
                }
            }

            // Force kill any stragglers
            for (Process p : workersFinal) {
                if (p.isAlive()) {
                    System.out.println("Coordinator: Force-killing unresponsive worker...");
                    p.destroyForcibly();
                }
            }
            
            System.out.println("Coordinator: All workers stopped.");
            
            // Now aggregate stats
            aggregateWorkerStats(startTime);
        }));

        // Wait for all workers to complete
        int exitedCount = 0;
        for (int i = 0; i < workers.size(); i++) {
            try {
                int exitCode = workers.get(i).waitFor();
                exitedCount++;
                System.out.println("[Worker " + (i + 1) + "] Exited with code " + exitCode +
                                   " (" + exitedCount + "/" + workers.size() + " complete)");
            } catch (InterruptedException e) {
                System.err.println("[Worker " + (i + 1) + "] Interrupted");
            }
        }

        // Wait for output threads to finish
        for (Thread t : outputThreads) {
            try {
                t.join(1000);
            } catch (InterruptedException e) {
                // Ignore
            }
        }

        System.out.println();

        // Aggregate worker stats (normal completion, not Ctrl+C)
        aggregateWorkerStats(startTime);
    }

    /**
     * Get total frames generated across all workers by reading their stats files.
     */
    private static long getTotalWorkerFrames() {
        long totalFrames = 0;
        for (int i = 1; i <= numWorkers; i++) {
            String statsFile = outputDir + "/" + WORKER_STATS_PREFIX + i + ".json";
            try {
                Path statsPath = Paths.get(statsFile);
                if (!Files.exists(statsPath)) continue;
                String content = Files.readString(statsPath);
                totalFrames += parseJsonLong(content, "frames", 0);
            } catch (Exception e) {
                // Ignore read errors
            }
        }
        return totalFrames;
    }

    /**
     * Aggregate stats from all worker JSON files and print summary.
     */
    private static volatile boolean statsAggregated = false;
    private static void aggregateWorkerStats(long coordStartTime) {
        // Prevent double-calling (shutdown hook + normal flow)
        synchronized (HomographyTrainingDataGenerator.class) {
            if (statsAggregated) {
                return;
            }
            statsAggregated = true;
        }
        
        long totalFrames = 0;
        long totalPages = 0;
        long totalDarkPages = 0;
        long totalLightPages = 0;
        long maxRuntimeMs = 0;
        int workersFound = 0;

        System.out.println("=".repeat(70));
        System.out.println("AGGREGATED STATS");
        System.out.println("=".repeat(70));
        System.out.println();
        System.out.println("Per-worker breakdown:");

        for (int i = 1; i <= numWorkers; i++) {
            String statsFile = outputDir + "/" + WORKER_STATS_PREFIX + i + ".json";
            try {
                // Check if file exists first
                Path statsPath = Paths.get(statsFile);
                if (!Files.exists(statsPath)) {
                    System.out.printf("  Worker %2d: (stats file not found: %s)%n", i, statsFile);
                    continue;
                }
                
                String content = Files.readString(statsPath);
                long frames = parseJsonLong(content, "frames", 0);
                long pages = parseJsonLong(content, "pages", 0);
                long runtimeMs = parseJsonLong(content, "runtime_ms", 0);
                long darkPages = parseJsonLong(content, "dark_mode_pages", 0);
                long lightPages = parseJsonLong(content, "light_mode_pages", 0);

                totalFrames += frames;
                totalPages += pages;
                totalDarkPages += darkPages;
                totalLightPages += lightPages;
                maxRuntimeMs = Math.max(maxRuntimeMs, runtimeMs);
                workersFound++;

                double fps = runtimeMs > 0 ? frames / (runtimeMs / 1000.0) : 0;
                System.out.printf("  Worker %2d: %,6d frames, %,4d pages, %.1f fps%n", i, frames, pages, fps);

                // Stats files are kept for debugging - they get overwritten on next run
            } catch (Exception e) {
                System.out.printf("  Worker %2d: (error reading stats: %s)%n", i, e.getMessage());
            }
        }

        long coordRuntimeMs = System.currentTimeMillis() - coordStartTime;
        double coordRuntimeSec = coordRuntimeMs / 1000.0;
        double coordRuntimeMin = coordRuntimeSec / 60.0;
        double aggregateFps = coordRuntimeSec > 0 ? totalFrames / coordRuntimeSec : 0;

        System.out.println();
        System.out.println("Totals:");
        System.out.printf("  Workers reporting:  %d / %d%n", workersFound, numWorkers);
        System.out.printf("  Total frames:       %,d%n", totalFrames);
        System.out.printf("  Total pages:        %,d%n", totalPages);
        System.out.printf("  Dark mode pages:    %,d%n", totalDarkPages);
        System.out.printf("  Light mode pages:   %,d%n", totalLightPages);
        System.out.println();
        System.out.println("Performance:");
        if (coordRuntimeMin >= 1.0) {
            System.out.printf("  Wall clock time:    %.1f minutes (%.0f seconds)%n", coordRuntimeMin, coordRuntimeSec);
        } else {
            System.out.printf("  Wall clock time:    %.1f seconds%n", coordRuntimeSec);
        }
        System.out.printf("  Aggregate throughput: %.1f frames/sec%n", aggregateFps);
        System.out.printf("  Projected hourly:   %,.0f frames/hour%n", aggregateFps * 3600);
        System.out.printf("  Projected daily:    %,.0f frames/day%n", aggregateFps * 86400);
        System.out.println();
        System.out.println("Output:");
        System.out.println("  " + outputDir + "/");
        System.out.println("    - " + numWorkers + " sets of 4 video files (human/training x content/white)");
        System.out.println("    - images/*.jpg (all workers)");
        System.out.println("    - labels/*.json (all workers)");
        System.out.println("=".repeat(70));
    }
}

