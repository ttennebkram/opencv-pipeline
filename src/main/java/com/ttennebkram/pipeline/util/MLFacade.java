package com.ttennebkram.pipeline.util;

import java.io.*;
import java.nio.file.*;
import java.util.function.Consumer;

/**
 * Unified ML facade that automatically chooses the best available backend:
 *
 * 1. Python + GPU (MPS/CUDA) - fastest for training
 * 2. Python + CPU - if Python installed but no GPU
 * 3. Pure Java (DJL) - fallback when Python not available
 *
 * Usage:
 *   MLFacade ml = new MLFacade();
 *   ml.initialize(System.out::println);
 *   ml.createModel(10);
 *   ml.loadMnist(128);
 *   ml.configureTraining(0.001f);
 *   for (int i = 0; i < epochs; i++) {
 *       TrainResult r = ml.trainEpoch();
 *   }
 *   ml.exportWeights("model.weights");
 *   ml.shutdown();
 *
 *   // Later, for inference (no Python needed):
 *   JavaCNNInference cnn = new JavaCNNInference();
 *   cnn.loadWeights("model.weights");
 *   int digit = cnn.predictClass(pixels, 28, 28);
 */
public class MLFacade {

    public enum Backend {
        PYTHON_GPU,    // Python + MPS or CUDA
        PYTHON_CPU,    // Python but no GPU
        JAVA_DJL       // Pure Java with DJL
    }

    private Backend backend;
    private PythonMLBridge pythonBridge;
    private DJLTrainer djlTrainer;
    private Consumer<String> logger;
    private boolean initialized = false;
    boolean forceDJL = false;  // For testing: skip Python detection

    /**
     * Initialize the ML facade, auto-detecting the best backend.
     */
    public void initialize(Consumer<String> logger) throws Exception {
        this.logger = logger != null ? logger : s -> {};

        log("╔═══════════════════════════════════════════════════════════╗");
        log("║              ML Facade - Auto Backend Selection           ║");
        log("╚═══════════════════════════════════════════════════════════╝");
        log("");

        // Try Python first (unless forced to use DJL)
        if (!forceDJL && tryPythonBackend()) {
            initialized = true;
            return;
        }

        // Fall back to Java DJL
        log("┌───────────────────────────────────────────────────────────┐");
        log("│ Falling back to Pure Java (DJL)                          │");
        log("├───────────────────────────────────────────────────────────┤");
        log("│ Training will be slower but works without Python.        │");
        log("└───────────────────────────────────────────────────────────┘");
        log("");

        backend = Backend.JAVA_DJL;
        djlTrainer = new DJLTrainer();
        log("  [Java] Backend: DJL (CPU)");
        log("");
        initialized = true;
    }

    private boolean tryPythonBackend() {
        log("  [Check] Looking for Python + PyTorch...");

        try {
            pythonBridge = new PythonMLBridge();
            pythonBridge.initialize(msg -> log("  [Python] " + msg));

            String device = pythonBridge.getDeviceType();
            if ("mps".equals(device) || "cuda".equals(device)) {
                backend = Backend.PYTHON_GPU;
                String gpuName = "mps".equals(device) ? "Apple Silicon (Metal)" : "NVIDIA (CUDA)";
                log("");
                log("┌───────────────────────────────────────────────────────────┐");
                log("│ Using Python + GPU                                        │");
                log("├───────────────────────────────────────────────────────────┤");
                log("│ GPU: " + padRight(gpuName, 51) + " │");
                log("│ This is the fastest option for training.                 │");
                log("└───────────────────────────────────────────────────────────┘");
            } else {
                backend = Backend.PYTHON_CPU;
                log("");
                log("┌───────────────────────────────────────────────────────────┐");
                log("│ Using Python + CPU                                        │");
                log("├───────────────────────────────────────────────────────────┤");
                log("│ No GPU detected. Training will be slower.                │");
                log("└───────────────────────────────────────────────────────────┘");
            }
            log("");
            return true;

        } catch (Exception e) {
            log("  [Check] Python not available: " + e.getMessage());
            log("");
            if (pythonBridge != null) {
                pythonBridge.shutdown();
                pythonBridge = null;
            }
            return false;
        }
    }

    /**
     * Get the active backend.
     */
    public Backend getBackend() {
        return backend;
    }

    /**
     * Create a CNN model.
     */
    public void createModel(int numClasses) throws Exception {
        checkInitialized();
        log("  [" + backendName() + "] Creating CNN model with " + numClasses + " classes...");

        if (backend == Backend.JAVA_DJL) {
            djlTrainer.createModel(numClasses);
        } else {
            pythonBridge.createModel("cnn", numClasses);
        }
        log("  [" + backendName() + "] Model created");
    }

    /**
     * Load MNIST dataset.
     */
    public DatasetInfo loadMnist(int batchSize) throws Exception {
        checkInitialized();
        log("  [" + backendName() + "] Loading MNIST dataset (batch size " + batchSize + ")...");

        DatasetInfo info;
        if (backend == Backend.JAVA_DJL) {
            DJLTrainer.DatasetInfo djlInfo = djlTrainer.loadMnist(batchSize);
            info = new DatasetInfo(djlInfo.trainSize, djlInfo.testSize);
        } else {
            PythonMLBridge.DatasetInfo pInfo = pythonBridge.loadDataset("mnist", batchSize);
            info = new DatasetInfo(pInfo.trainSize, pInfo.testSize);
        }

        log("  [" + backendName() + "] Loaded: " + info.trainSize + " train, " + info.testSize + " test");
        return info;
    }

    /**
     * Configure training parameters.
     */
    public void configureTraining(float learningRate) throws Exception {
        checkInitialized();
        log("  [" + backendName() + "] Configuring training (lr=" + learningRate + ")...");

        if (backend == Backend.JAVA_DJL) {
            djlTrainer.configureTraining(learningRate);
        } else {
            pythonBridge.configureTraining(learningRate);
        }
    }

    /**
     * Train for one epoch.
     */
    public TrainResult trainEpoch() throws Exception {
        checkInitialized();

        TrainResult result;
        if (backend == Backend.JAVA_DJL) {
            DJLTrainer.TrainResult djlResult = djlTrainer.trainEpoch();
            result = new TrainResult(djlResult.loss, djlResult.accuracy, djlResult.timeMs);
        } else {
            PythonMLBridge.TrainResult pyResult = pythonBridge.trainEpoch();
            result = new TrainResult(pyResult.loss, pyResult.accuracy, pyResult.timeMs);
        }

        log("  [" + backendName() + "] " + result);
        return result;
    }

    /**
     * Evaluate on test set.
     */
    public EvalResult evaluate() throws Exception {
        checkInitialized();
        log("  [" + backendName() + "] Evaluating on test set...");

        EvalResult result;
        if (backend == Backend.JAVA_DJL) {
            DJLTrainer.EvalResult djlResult = djlTrainer.evaluate();
            result = new EvalResult(djlResult.loss, djlResult.accuracy);
        } else {
            PythonMLBridge.EvalResult pyResult = pythonBridge.evaluate();
            result = new EvalResult(pyResult.loss, pyResult.accuracy);
        }

        log("  [" + backendName() + "] " + result);
        return result;
    }

    /**
     * Export weights for Java inference.
     */
    public void exportWeights(String path) throws Exception {
        checkInitialized();
        log("  [" + backendName() + "] Exporting weights to: " + path);

        if (backend == Backend.JAVA_DJL) {
            djlTrainer.exportWeightsForJava(path);
        } else {
            pythonBridge.exportWeightsForJava(path);
        }

        long size = Files.size(Path.of(path));
        log("  [" + backendName() + "] Exported (" + (size / 1024) + " KB)");
    }

    /**
     * Shutdown the backend.
     */
    public void shutdown() {
        if (pythonBridge != null) {
            pythonBridge.shutdown();
            pythonBridge = null;
        }
        if (djlTrainer != null) {
            djlTrainer.close();
            djlTrainer = null;
        }
        initialized = false;
        log("  [" + backendName() + "] Shutdown complete");
    }

    // ========== Helpers ==========

    private void checkInitialized() {
        if (!initialized) {
            throw new IllegalStateException("Not initialized. Call initialize() first.");
        }
    }

    private String backendName() {
        if (backend == null) return "?";
        return switch (backend) {
            case PYTHON_GPU -> "Python/GPU";
            case PYTHON_CPU -> "Python/CPU";
            case JAVA_DJL -> "Java/DJL";
        };
    }

    private void log(String msg) {
        if (logger != null) logger.accept(msg);
    }

    private String padRight(String s, int width) {
        if (s.length() >= width) return s.substring(0, width);
        return s + " ".repeat(width - s.length());
    }

    // ========== Result Classes ==========

    public static class DatasetInfo {
        public final int trainSize;
        public final int testSize;

        public DatasetInfo(int trainSize, int testSize) {
            this.trainSize = trainSize;
            this.testSize = testSize;
        }
    }

    public static class TrainResult {
        public final float loss;
        public final float accuracy;
        public final long timeMs;

        public TrainResult(float loss, float accuracy, long timeMs) {
            this.loss = loss;
            this.accuracy = accuracy;
            this.timeMs = timeMs;
        }

        @Override
        public String toString() {
            return String.format("Loss: %.4f, Accuracy: %.2f%%, Time: %dms", loss, accuracy * 100, timeMs);
        }
    }

    public static class EvalResult {
        public final float loss;
        public final float accuracy;

        public EvalResult(float loss, float accuracy) {
            this.loss = loss;
            this.accuracy = accuracy;
        }

        @Override
        public String toString() {
            return String.format("Loss: %.4f, Accuracy: %.2f%%", loss, accuracy * 100);
        }
    }

    // ========== Main for testing ==========

    public static void main(String[] args) throws Exception {
        boolean forceDJL = args.length > 0 && args[0].equals("--djl");

        System.out.println("MLFacade Demo - Automatic Backend Selection");
        if (forceDJL) {
            System.out.println("(Forcing DJL backend for testing)");
        }
        System.out.println();

        MLFacade ml = new MLFacade();
        if (forceDJL) {
            ml.forceDJL = true;
        }

        try {
            ml.initialize(System.out::println);

            System.out.println("\n--- Training ---\n");

            ml.createModel(10);
            ml.loadMnist(128);
            ml.configureTraining(0.001f);

            int epochs = 3;
            for (int epoch = 1; epoch <= epochs; epoch++) {
                System.out.println("\n  Epoch " + epoch + "/" + epochs + ":");
                ml.trainEpoch();
            }

            System.out.println("\n--- Evaluation ---\n");
            ml.evaluate();

            System.out.println("\n--- Export ---\n");
            String weightsPath = "/tmp/ml_facade_model.weights";
            ml.exportWeights(weightsPath);

            System.out.println("\n--- Java Inference Test ---\n");
            JavaCNNInference cnn = new JavaCNNInference();
            cnn.loadWeights(weightsPath);

            // Quick test
            int[] testImage = new int[28 * 28];
            // Draw a "1" pattern
            for (int y = 5; y < 23; y++) {
                testImage[y * 28 + 14] = 255;
            }

            JavaCNNInference.PredictResult result = cnn.predictTimed(testImage, 28, 28);
            System.out.println("  Java inference: " + result);

        } finally {
            System.out.println("\n--- Shutdown ---\n");
            ml.shutdown();
        }

        System.out.println("\nDone!");
    }
}
