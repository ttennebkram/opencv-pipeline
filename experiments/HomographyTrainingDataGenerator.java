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
    private static int renderWidth = 1280;    // Render at higher res for content variety
    private static int renderHeight = 960;
    private static int samplesPerPage = 30;  // How many distortions per source image
    private static int totalPages = -1;       // How many pages to render (-1 = continuous until Ctrl+C)
    private static String outputDir = "training_data/homography";
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
    private Random random = new Random(42);  // Seed for reproducibility

    // Video output - 4 videos: human (sorted) and training (random) for content and white
    private VideoWriter humanContentWriter;   // Content video sorted by rotation (for humans)
    private VideoWriter humanWhiteWriter;     // White video sorted by rotation (for humans)
    private VideoWriter trainingContentWriter; // Content video in random order (for training)
    private VideoWriter trainingWhiteWriter;   // White video in random order (for training)
    private PrintWriter videoLabelsWriter;
    private static int fps = 30;  // Video fps (default 30), also controls display delay if > 0
    private static int progressInterval = 100;  // Print dot every N samples
    private static boolean estimateOnly = false;  // Just show estimates, don't run
    private static final String STATS_CACHE_FILE = System.getProperty("user.home") + "/.homography_generator_stats.json";

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

    // Dark mode - randomly chosen per page for training variety
    private boolean currentPageDarkMode = false;

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
                case "--help":
                    printUsage();
                    return;
            }
        }

        // Load cached stats from previous runs
        loadCachedStats();

        // Generate list of Wikipedia URLs
        generateUrlList();

        System.out.println("Homography Training Data Generator");
        System.out.println("===================================");
        System.out.println("Image size: " + imageWidth + "x" + imageHeight);
        System.out.println("Samples per page: " + samplesPerPage);
        if (totalPages < 0) {
            System.out.println("Mode: CONTINUOUS (Ctrl+C to stop)");
            System.out.println("WARNING: Disk usage grows without limit! (~200 KB/frame)");
        } else {
            int totalSamples = samplesPerPage * totalPages;
            long estimatedMB = (totalSamples * 200L) / 1024;  // ~200KB per frame
            System.out.println("Total pages: " + totalPages);
            System.out.printf("Total samples: %,d (estimated disk: ~%,d MB)%n", totalSamples, estimatedMB);
        }
        System.out.println("Output directory: " + outputDir);
        System.out.println();

        // Show projections based on cached stats
        printProjections();

        // If estimate-only mode, exit now
        if (estimateOnly) {
            System.out.println("(--estimate mode: exiting without generating data)");
            System.exit(0);  // Force exit - OpenCV native library may leave threads running
        }

        // Create output directories
        try {
            Files.createDirectories(Paths.get(outputDir));
            Files.createDirectories(Paths.get(outputDir, "images"));
            Files.createDirectories(Paths.get(outputDir, "labels"));
        } catch (IOException e) {
            System.err.println("Failed to create output directory: " + e.getMessage());
            return;
        }

        // Add shutdown hook for Ctrl-C
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("\nShutdown requested, cleaning up...");
            if (instance != null) {
                instance.cleanup();
            }
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

        System.out.println("\nClosing resources...");

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

        // Save stats to cache file for future estimates
        saveCachedStats();

        // Print summary of output files
        printOutputSummary();
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
        String[] videoFiles = {
            "human_content.avi    - Human review only: content sorted by rotation",
            "human_white.avi      - Human review only: geometry sorted by rotation",
            "training_content.avi - For training: content in random order",
            "training_white.avi   - For training: geometry in random order"
        };
        for (String video : videoFiles) {
            String filename = video.split(" ")[0].trim();
            if (isRelative) {
                System.out.println("  " + outputDir + "/" + filename);
                System.out.println("    -> " + absolutePath + "/" + filename);
            } else {
                System.out.println("  " + absolutePath + "/" + filename);
            }
            System.out.println("       " + video.substring(video.indexOf("-")));
        }
        System.out.println();

        System.out.println("LABELS:");
        if (isRelative) {
            System.out.println("  " + outputDir + "/training_labels.json");
            System.out.println("    -> " + absolutePath + "/training_labels.json");
        } else {
            System.out.println("  " + absolutePath + "/training_labels.json");
        }
        System.out.println("       Frame-by-frame labels (homography, inverse, corners)");
        System.out.println();

        System.out.println("INDIVIDUAL SAMPLES:");
        System.out.println("  images/NNNNN_SS.png       - Distorted content image");
        System.out.println("  images/NNNNN_SS_white.png - White page version");
        System.out.println("  labels/NNNNN_SS.json      - Full labels for each sample");
        System.out.println();

        System.out.println("METADATA:");
        System.out.println("  manifest.json             - Sample list and generation metadata");
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
        System.out.println("  --output DIR         Output directory (default: training_data/homography)");
        System.out.println("  --fps N              Video fps and display delay (default: 30)");
        System.out.println("  --estimate           Show projections based on previous runs, don't generate");
        System.out.println("  --help               Show this help");
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

        // Initialize video writers (AVI with MJPG) - 4 videos: human/training x content/white
        int fourcc = VideoWriter.fourcc('M', 'J', 'P', 'G');
        int videoFps = fps > 0 ? fps : 30;  // Use configured fps (default 30)
        Size videoSize = new Size(imageWidth, imageHeight);

        // Human videos (sorted by rotation angle for easy viewing)
        humanContentWriter = openVideoWriter(outputDir + "/human_content.avi", fourcc, videoFps, videoSize);
        humanWhiteWriter = openVideoWriter(outputDir + "/human_white.avi", fourcc, videoFps, videoSize);

        // Training videos (random order to prevent temporal learning)
        trainingContentWriter = openVideoWriter(outputDir + "/training_content.avi", fourcc, videoFps, videoSize);
        trainingWhiteWriter = openVideoWriter(outputDir + "/training_white.avi", fourcc, videoFps, videoSize);

        // Initialize video labels file (JSON array, one entry per frame)
        try {
            videoLabelsWriter = new PrintWriter(outputDir + "/training_labels.json");
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
        primaryStage.setX(100);
        primaryStage.setY(100);
        primaryStage.show();

        // Create preview window for distorted output (positioned to the right)
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

        // Start loading pages
        loadNextPage();
    }

    private void loadNextPage() {
        if (shuttingDown) return;

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

    /**
     * Capture snapshot on FX thread, then hand off to background thread for processing.
     */
    private void captureSnapshot() {
        if (shuttingDown) return;

        try {
            // Capture the WebView as an image (must be on FX thread)
            WritableImage fxImage = webView.snapshot(new SnapshotParameters(), null);
            BufferedImage bufferedImage = SwingFXUtils.fromFXImage(fxImage, null);

            // Process on background thread to keep UI responsive
            final int pageIdx = currentUrlIndex;
            new Thread(() -> processCapture(bufferedImage, pageIdx)).start();

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
    private void processCapture(BufferedImage bufferedImage, int pageIdx) {
        if (shuttingDown) return;

        try {
            // Convert to OpenCV Mat
            Mat originalMat = bufferedImageToMat(bufferedImage);

            // Generate multiple distorted versions
            for (int i = 0; i < samplesPerPage && !shuttingDown; i++) {
                generateSample(originalMat, pageIdx, i);
                currentSampleIndex++;
                totalSamplesGenerated++;
            }

            originalMat.release();

            // Flush buffered frames to videos (sorted for human, random for training)
            flushPageFrameBuffer();

            // Print sample count for this page
            System.out.println(samplesPerPage + " samples -> " + totalSamplesGenerated + " total");

        } catch (Exception e) {
            System.err.println("Error processing page: " + e.getMessage());
            e.printStackTrace();
        }

        // Move to next page (must be on FX thread)
        currentUrlIndex++;
        Platform.runLater(this::loadNextPage);
    }

    private void generateSample(Mat original, int pageIndex, int sampleIndex) {
        String sampleId = String.format("%05d_%02d", pageIndex, sampleIndex);

        // Source image is renderWidth x renderHeight, output is imageWidth x imageHeight
        // We'll pick a random quadrilateral from the source and map it to the output

        // Source: a rectangle from the rendered webpage (we'll crop/sample from this)
        // Pick a random region within the source image with some margin
        double margin = 0.1;
        double srcX1 = renderWidth * (margin + random.nextDouble() * (0.4 - margin));
        double srcY1 = renderHeight * (margin + random.nextDouble() * (0.4 - margin));
        double srcX2 = renderWidth * (0.6 + random.nextDouble() * (0.4 - margin));
        double srcY2 = renderHeight * (0.6 + random.nextDouble() * (0.4 - margin));

        // Source corners - just a rectangle from the webpage
        Point[] srcCorners = {
            new Point(srcX1, srcY1),  // TL
            new Point(srcX2, srcY1),  // TR
            new Point(srcX2, srcY2),  // BR
            new Point(srcX1, srcY2)   // BL
        };

        // Destination: simulate paper on a table viewed from a tilted camera
        // The paper appears as a distorted quadrilateral in the output image

        double paperRotation = random.nextDouble() * 2 * Math.PI;  // Paper orientation 0-360°
        double cameraTiltX = (random.nextDouble() - 0.5) * 0.6;    // Camera tilt left/right (±0.3)
        double cameraTiltY = (random.nextDouble() - 0.5) * 0.6;    // Camera tilt forward/back (±0.3)
        // Use squared random to bias toward smaller scales (more distant papers)
        double scaleRandom = random.nextDouble();
        scaleRandom = scaleRandom * scaleRandom;  // Square it: more samples near 0
        double paperScale = 0.05 + scaleRandom * 0.75;             // Paper size 5-80% of frame (biased toward small/distant)

        // Random position - paper can be anywhere, including partially off-frame
        // Range allows paper center from -20% to 120% of image dimensions
        double centerX = imageWidth * (-0.2 + random.nextDouble() * 1.4);
        double centerY = imageHeight * (-0.2 + random.nextDouble() * 1.4);

        // Base paper size in output
        double halfW = imageWidth * paperScale / 2.0;
        double halfH = imageHeight * paperScale / 2.0;

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
            double distFactorX = 1.0 + cameraTiltX * (rx / halfW);
            double distFactorY = 1.0 + cameraTiltY * (ry / halfH);
            double perspScale = distFactorX * distFactorY;
            rx *= perspScale;
            ry *= perspScale;

            // 3. Translate to position in output image
            dstCorners[i] = new Point(centerX + rx, centerY + ry);
        }

        // getPerspectiveTransform: maps srcCorners -> dstCorners
        // warpPerspective uses inverse: for each dst pixel, find where it comes from in src
        MatOfPoint2f srcMat = new MatOfPoint2f(srcCorners);
        MatOfPoint2f dstMat = new MatOfPoint2f(dstCorners);

        // Compute homography
        Mat H = Imgproc.getPerspectiveTransform(srcMat, dstMat);
        Mat H_inv = H.inv();

        // Create mask for warping
        Mat mask = Mat.ones(original.size(), CvType.CV_8UC1);
        mask.setTo(new Scalar(255));

        // Create random noise background to avoid the network learning border patterns
        Mat distorted = new Mat(imageHeight, imageWidth, CvType.CV_8UC3);
        Core.randu(distorted, 0, 256);

        // Warp original onto noise background
        Mat warped = new Mat();
        Mat warpedMask = new Mat();
        Imgproc.warpPerspective(original, warped, H, new Size(imageWidth, imageHeight));
        Imgproc.warpPerspective(mask, warpedMask, H, new Size(imageWidth, imageHeight));

        // Randomly apply shadows (50% chance) - save parameters for white page
        int shadowType = -1;  // -1 = no shadow
        long shadowSeed = random.nextLong();  // Save seed to reproduce same shadow
        if (random.nextBoolean()) {
            shadowType = random.nextInt(4);
            Random shadowRandom = new Random(shadowSeed);
            applyShadowWithParams(warped, warpedMask, shadowType, shadowRandom);
        }

        warped.copyTo(distorted, warpedMask);
        warped.release();
        warpedMask.release();
        mask.release();

        // Save distorted image
        String distortedPath = outputDir + "/images/" + sampleId + ".png";
        Imgcodecs.imwrite(distortedPath, distorted);

        // Create and save white page version (same transform, white rectangle on noise)
        Mat whitePage = new Mat(original.size(), CvType.CV_8UC3, new Scalar(255, 255, 255));
        Mat whiteMask = Mat.ones(original.size(), CvType.CV_8UC1);
        whiteMask.setTo(new Scalar(255));
        Mat whiteDistorted = new Mat(imageHeight, imageWidth, CvType.CV_8UC3);
        Core.randu(whiteDistorted, 0, 256);  // Fresh random noise background
        Mat whiteWarped = new Mat();
        Mat whiteMaskWarped = new Mat();
        Imgproc.warpPerspective(whitePage, whiteWarped, H, new Size(imageWidth, imageHeight));
        Imgproc.warpPerspective(whiteMask, whiteMaskWarped, H, new Size(imageWidth, imageHeight));

        // Apply same hard shadow to white page (only the 100% black parts)
        if (shadowType >= 0) {
            Random shadowRandom = new Random(shadowSeed);
            applyHardShadowOnly(whiteWarped, whiteMaskWarped, shadowType, shadowRandom);
        }

        whiteWarped.copyTo(whiteDistorted, whiteMaskWarped);
        String whitePath = outputDir + "/images/" + sampleId + "_white.png";
        Imgcodecs.imwrite(whitePath, whiteDistorted);
        whitePage.release();
        whiteMask.release();
        whiteWarped.release();
        whiteMaskWarped.release();

        // Save labels as JSON (homography, inverse, and 4 corners)
        String labelsPath = outputDir + "/labels/" + sampleId + ".json";
        saveLabelsJson(H, H_inv, dstCorners, labelsPath);

        // Build label JSON for video labels file
        String labelJson = buildLabelJson(sampleId, H, H_inv, dstCorners);

        // Buffer frames for deferred video writing (sorted vs random order)
        synchronized (bufferLock) {
            pageFrameBuffer.add(new FrameData(distorted, whiteDistorted, paperRotation, labelJson));
        }

        // Release whiteDistorted - we cloned it in FrameData
        whiteDistorted.release();

        // Print progress dot
        if (totalSamplesGenerated > 0 && totalSamplesGenerated % progressInterval == 0) {
            System.out.print(".");
            System.out.flush();
        }

        // Update preview window with distorted image
        updatePreview(distorted, sampleIndex);

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
                writer.release();
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
        try (PrintWriter writer = new PrintWriter(outputDir + "/manifest.json")) {
            writer.println("{");
            writer.println("  \"generated\": \"" + new java.util.Date() + "\",");
            writer.println("  \"imageWidth\": " + imageWidth + ",");
            writer.println("  \"imageHeight\": " + imageHeight + ",");
            writer.println("  \"totalSamples\": " + totalSamplesGenerated + ",");
            writer.println("  \"samplesPerPage\": " + samplesPerPage + ",");
            writer.println("  \"description\": \"Each label contains 9 homography values and 9 inverse values\",");
            writer.println("  \"samples\": [");

            int sampleNum = 0;
            for (int pageIdx = 0; pageIdx < currentUrlIndex; pageIdx++) {
                for (int sampleIdx = 0; sampleIdx < samplesPerPage; sampleIdx++) {
                    String sampleId = String.format("%05d_%02d", pageIdx, sampleIdx);
                    writer.print("    {\"id\": \"" + sampleId + "\", ");
                    writer.print("\"image\": \"images/" + sampleId + ".png\", ");
                    writer.print("\"labels\": \"labels/" + sampleId + ".json\"}");
                    sampleNum++;
                    if (sampleNum < totalSamplesGenerated) {
                        writer.println(",");
                    } else {
                        writer.println();
                    }
                }
            }
            writer.println("  ]");
            writer.println("}");
        } catch (IOException e) {
            System.err.println("Failed to write manifest: " + e.getMessage());
        }
    }
}
