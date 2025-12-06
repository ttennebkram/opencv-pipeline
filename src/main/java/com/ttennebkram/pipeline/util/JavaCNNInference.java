package com.ttennebkram.pipeline.util;

import java.io.*;
import java.nio.*;
import java.nio.file.*;
import java.util.*;

/**
 * Pure Java CNN inference for MNIST-style models.
 * No Python required - runs entirely on CPU.
 *
 * For single-image inference, CPU is fast enough (~1-5ms per image).
 * Use Python/GPU for training and batch inference.
 *
 * Usage:
 *   JavaCNNInference cnn = new JavaCNNInference();
 *   cnn.loadWeights("model.weights");  // Exported from Python
 *   float[] probs = cnn.predict(imageData, 28, 28);
 *   int digit = cnn.predictClass(imageData, 28, 28);
 */
public class JavaCNNInference {

    // Network architecture (must match Python model)
    // Conv1: 1 -> 8 filters, 3x3 kernel
    // Pool1: 2x2 max pool
    // Conv2: 8 -> 16 filters, 3x3 kernel
    // Pool2: 2x2 max pool
    // FC1: 400 -> 64
    // FC2: 64 -> 10

    private float[][][][] conv1Weights;  // [8][1][3][3]
    private float[] conv1Bias;           // [8]
    private float[][][][] conv2Weights;  // [16][8][3][3]
    private float[] conv2Bias;           // [16]
    private float[][] fc1Weights;        // [64][400]
    private float[] fc1Bias;             // [64]
    private float[][] fc2Weights;        // [10][64]
    private float[] fc2Bias;             // [10]

    private boolean loaded = false;

    // MNIST normalization constants
    private static final float MNIST_MEAN = 0.1307f;
    private static final float MNIST_STD = 0.3081f;

    /**
     * Load weights from a file exported by Python.
     */
    public void loadWeights(String path) throws IOException {
        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(path)))) {
            // Read magic number and version
            int magic = dis.readInt();
            if (magic != 0x4A434E4E) {  // "JCNN"
                throw new IOException("Invalid weight file format");
            }
            int version = dis.readInt();
            if (version != 1) {
                throw new IOException("Unsupported weight file version: " + version);
            }

            // Conv1: [8][1][3][3]
            conv1Weights = new float[8][1][3][3];
            conv1Bias = new float[8];
            for (int f = 0; f < 8; f++) {
                for (int c = 0; c < 1; c++) {
                    for (int h = 0; h < 3; h++) {
                        for (int w = 0; w < 3; w++) {
                            conv1Weights[f][c][h][w] = dis.readFloat();
                        }
                    }
                }
            }
            for (int f = 0; f < 8; f++) {
                conv1Bias[f] = dis.readFloat();
            }

            // Conv2: [16][8][3][3]
            conv2Weights = new float[16][8][3][3];
            conv2Bias = new float[16];
            for (int f = 0; f < 16; f++) {
                for (int c = 0; c < 8; c++) {
                    for (int h = 0; h < 3; h++) {
                        for (int w = 0; w < 3; w++) {
                            conv2Weights[f][c][h][w] = dis.readFloat();
                        }
                    }
                }
            }
            for (int f = 0; f < 16; f++) {
                conv2Bias[f] = dis.readFloat();
            }

            // FC1: [64][400]
            fc1Weights = new float[64][400];
            fc1Bias = new float[64];
            for (int o = 0; o < 64; o++) {
                for (int i = 0; i < 400; i++) {
                    fc1Weights[o][i] = dis.readFloat();
                }
            }
            for (int o = 0; o < 64; o++) {
                fc1Bias[o] = dis.readFloat();
            }

            // FC2: [10][64]
            fc2Weights = new float[10][64];
            fc2Bias = new float[10];
            for (int o = 0; o < 10; o++) {
                for (int i = 0; i < 64; i++) {
                    fc2Weights[o][i] = dis.readFloat();
                }
            }
            for (int o = 0; o < 10; o++) {
                fc2Bias[o] = dis.readFloat();
            }

            loaded = true;
        }
    }

    /**
     * Check if weights are loaded.
     */
    public boolean isLoaded() {
        return loaded;
    }

    /**
     * Predict class probabilities for a grayscale image.
     * @param pixels grayscale pixel values (0-255), row-major order
     * @param width image width (should be 28 for MNIST)
     * @param height image height (should be 28 for MNIST)
     * @return probability for each class (0-9)
     */
    public float[] predict(int[] pixels, int width, int height) {
        if (!loaded) {
            throw new IllegalStateException("Weights not loaded. Call loadWeights() first.");
        }

        // Convert to normalized float array [1][height][width]
        float[][] input = new float[height][width];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float pixel = pixels[y * width + x] / 255.0f;
                input[y][x] = (pixel - MNIST_MEAN) / MNIST_STD;
            }
        }

        // Forward pass
        // Conv1 + ReLU: [1][28][28] -> [8][26][26]
        float[][][] conv1Out = conv2d(new float[][][]{input}, conv1Weights, conv1Bias);
        relu3d(conv1Out);

        // Pool1: [8][26][26] -> [8][13][13]
        float[][][] pool1Out = maxPool2d(conv1Out, 2, 2);

        // Conv2 + ReLU: [8][13][13] -> [16][11][11]
        float[][][] conv2Out = conv2d(pool1Out, conv2Weights, conv2Bias);
        relu3d(conv2Out);

        // Pool2: [16][11][11] -> [16][5][5]
        float[][][] pool2Out = maxPool2d(conv2Out, 2, 2);

        // Flatten: [16][5][5] -> [400]
        float[] flat = flatten(pool2Out);

        // FC1 + ReLU: [400] -> [64]
        float[] fc1Out = linear(flat, fc1Weights, fc1Bias);
        relu1d(fc1Out);

        // FC2: [64] -> [10]
        float[] fc2Out = linear(fc1Out, fc2Weights, fc2Bias);

        // Softmax
        return softmax(fc2Out);
    }

    /**
     * Predict the most likely class.
     */
    public int predictClass(int[] pixels, int width, int height) {
        float[] probs = predict(pixels, width, height);
        int maxIdx = 0;
        for (int i = 1; i < probs.length; i++) {
            if (probs[i] > probs[maxIdx]) {
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    /**
     * Predict with timing info.
     */
    public PredictResult predictTimed(int[] pixels, int width, int height) {
        long start = System.nanoTime();
        float[] probs = predict(pixels, width, height);
        long elapsed = System.nanoTime() - start;

        int maxIdx = 0;
        for (int i = 1; i < probs.length; i++) {
            if (probs[i] > probs[maxIdx]) {
                maxIdx = i;
            }
        }

        return new PredictResult(maxIdx, probs[maxIdx], probs, elapsed / 1_000_000.0);
    }

    // ========== Neural Network Operations ==========

    /**
     * 2D Convolution (no padding, stride 1)
     */
    private float[][][] conv2d(float[][][] input, float[][][][] weights, float[] bias) {
        int inChannels = input.length;
        int inHeight = input[0].length;
        int inWidth = input[0][0].length;
        int outChannels = weights.length;
        int kernelH = weights[0][0].length;
        int kernelW = weights[0][0][0].length;
        int outHeight = inHeight - kernelH + 1;
        int outWidth = inWidth - kernelW + 1;

        float[][][] output = new float[outChannels][outHeight][outWidth];

        for (int oc = 0; oc < outChannels; oc++) {
            for (int oh = 0; oh < outHeight; oh++) {
                for (int ow = 0; ow < outWidth; ow++) {
                    float sum = bias[oc];
                    for (int ic = 0; ic < inChannels; ic++) {
                        for (int kh = 0; kh < kernelH; kh++) {
                            for (int kw = 0; kw < kernelW; kw++) {
                                sum += input[ic][oh + kh][ow + kw] * weights[oc][ic][kh][kw];
                            }
                        }
                    }
                    output[oc][oh][ow] = sum;
                }
            }
        }

        return output;
    }

    /**
     * 2D Max Pooling
     */
    private float[][][] maxPool2d(float[][][] input, int poolH, int poolW) {
        int channels = input.length;
        int inHeight = input[0].length;
        int inWidth = input[0][0].length;
        int outHeight = inHeight / poolH;
        int outWidth = inWidth / poolW;

        float[][][] output = new float[channels][outHeight][outWidth];

        for (int c = 0; c < channels; c++) {
            for (int oh = 0; oh < outHeight; oh++) {
                for (int ow = 0; ow < outWidth; ow++) {
                    float maxVal = Float.NEGATIVE_INFINITY;
                    for (int ph = 0; ph < poolH; ph++) {
                        for (int pw = 0; pw < poolW; pw++) {
                            float val = input[c][oh * poolH + ph][ow * poolW + pw];
                            if (val > maxVal) maxVal = val;
                        }
                    }
                    output[c][oh][ow] = maxVal;
                }
            }
        }

        return output;
    }

    /**
     * Linear (fully connected) layer
     */
    private float[] linear(float[] input, float[][] weights, float[] bias) {
        int outSize = weights.length;
        int inSize = weights[0].length;
        float[] output = new float[outSize];

        for (int o = 0; o < outSize; o++) {
            float sum = bias[o];
            for (int i = 0; i < inSize; i++) {
                sum += input[i] * weights[o][i];
            }
            output[o] = sum;
        }

        return output;
    }

    /**
     * Flatten 3D array to 1D
     */
    private float[] flatten(float[][][] input) {
        int channels = input.length;
        int height = input[0].length;
        int width = input[0][0].length;
        float[] output = new float[channels * height * width];

        int idx = 0;
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    output[idx++] = input[c][h][w];
                }
            }
        }

        return output;
    }

    /**
     * ReLU activation (in-place, 3D)
     */
    private void relu3d(float[][][] arr) {
        for (int c = 0; c < arr.length; c++) {
            for (int h = 0; h < arr[0].length; h++) {
                for (int w = 0; w < arr[0][0].length; w++) {
                    if (arr[c][h][w] < 0) arr[c][h][w] = 0;
                }
            }
        }
    }

    /**
     * ReLU activation (in-place, 1D)
     */
    private void relu1d(float[] arr) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] < 0) arr[i] = 0;
        }
    }

    /**
     * Softmax activation
     */
    private float[] softmax(float[] input) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : input) if (v > max) max = v;

        float sum = 0;
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = (float) Math.exp(input[i] - max);
            sum += output[i];
        }
        for (int i = 0; i < output.length; i++) {
            output[i] /= sum;
        }

        return output;
    }

    // ========== Result Class ==========

    public static class PredictResult {
        public final int predictedClass;
        public final float confidence;
        public final float[] probabilities;
        public final double inferenceTimeMs;

        public PredictResult(int predictedClass, float confidence, float[] probabilities, double inferenceTimeMs) {
            this.predictedClass = predictedClass;
            this.confidence = confidence;
            this.probabilities = probabilities;
            this.inferenceTimeMs = inferenceTimeMs;
        }

        @Override
        public String toString() {
            return String.format("Predicted: %d (%.2f%% confidence) in %.3fms",
                    predictedClass, confidence * 100, inferenceTimeMs);
        }
    }

    // ========== Python Weight Export Script ==========

    /**
     * Get Python code to export model weights for Java inference.
     */
    public static String getExportScript() {
        return """
            import struct
            import torch

            def export_for_java(model, path):
                \"\"\"Export CNN weights in format readable by JavaCNNInference.\"\"\"
                with open(path, 'wb') as f:
                    # Magic number "JCNN" and version
                    f.write(struct.pack('>I', 0x4A434E4E))
                    f.write(struct.pack('>I', 1))

                    # Conv1 weights [8][1][3][3] and bias [8]
                    conv1_w = model.conv1.weight.detach().cpu().numpy()
                    conv1_b = model.conv1.bias.detach().cpu().numpy()
                    for f_idx in range(8):
                        for c in range(1):
                            for h in range(3):
                                for w in range(3):
                                    f.write(struct.pack('>f', conv1_w[f_idx, c, h, w]))
                    for f_idx in range(8):
                        f.write(struct.pack('>f', conv1_b[f_idx]))

                    # Conv2 weights [16][8][3][3] and bias [16]
                    conv2_w = model.conv2.weight.detach().cpu().numpy()
                    conv2_b = model.conv2.bias.detach().cpu().numpy()
                    for f_idx in range(16):
                        for c in range(8):
                            for h in range(3):
                                for w in range(3):
                                    f.write(struct.pack('>f', conv2_w[f_idx, c, h, w]))
                    for f_idx in range(16):
                        f.write(struct.pack('>f', conv2_b[f_idx]))

                    # FC1 weights [64][400] and bias [64]
                    fc1_w = model.fc1.weight.detach().cpu().numpy()
                    fc1_b = model.fc1.bias.detach().cpu().numpy()
                    for o in range(64):
                        for i in range(400):
                            f.write(struct.pack('>f', fc1_w[o, i]))
                    for o in range(64):
                        f.write(struct.pack('>f', fc1_b[o]))

                    # FC2 weights [10][64] and bias [10]
                    fc2_w = model.fc2.weight.detach().cpu().numpy()
                    fc2_b = model.fc2.bias.detach().cpu().numpy()
                    for o in range(10):
                        for i in range(64):
                            f.write(struct.pack('>f', fc2_w[o, i]))
                    for o in range(10):
                        f.write(struct.pack('>f', fc2_b[o]))

                print(f"Weights exported to {path}")
                print(f"  Conv1: {conv1_w.shape}, bias: {conv1_b.shape}")
                print(f"  Conv2: {conv2_w.shape}, bias: {conv2_b.shape}")
                print(f"  FC1: {fc1_w.shape}, bias: {fc1_b.shape}")
                print(f"  FC2: {fc2_w.shape}, bias: {fc2_b.shape}")
            """;
    }

    // ========== Main for testing ==========

    public static void main(String[] args) throws Exception {
        System.out.println("╔══════════════════════════════════════════════════════════════════╗");
        System.out.println("║         JavaCNNInference - Pure Java Neural Network             ║");
        System.out.println("╚══════════════════════════════════════════════════════════════════╝\n");

        System.out.println("This demo trains a model in Python (GPU), exports weights,");
        System.out.println("then runs inference in pure Java (CPU).\n");

        // Step 1: Train in Python and export weights
        System.out.println("┌─────────────────────────────────────────────────────────────────┐");
        System.out.println("│ STEP 1: Train model in Python and export weights               │");
        System.out.println("└─────────────────────────────────────────────────────────────────┘\n");

        PythonMLBridge bridge = new PythonMLBridge();
        String weightsPath = "/tmp/mnist_model.weights";

        try {
            System.out.println("  [Java]   → Starting Python bridge...");
            bridge.initialize(msg -> System.out.println("  [Python] ← " + msg));
            System.out.println("  [Python] ← Device: " + bridge.getDeviceType() + "\n");

            System.out.println("  [Java]   → Creating and training model...");
            bridge.createModel("cnn", 10);
            bridge.loadDataset("mnist", 128);
            bridge.configureTraining(0.001f);

            for (int epoch = 1; epoch <= 2; epoch++) {
                PythonMLBridge.TrainResult result = bridge.trainEpoch();
                System.out.println("  [Python] ← Epoch " + epoch + ": " + result);
            }

            System.out.println("\n  [Java]   → Exporting weights to: " + weightsPath);
            bridge.exportWeightsForJava(weightsPath);
            System.out.println("  [Python] ← Weights exported\n");

        } finally {
            bridge.shutdown();
        }

        // Step 2: Load weights in Java
        System.out.println("┌─────────────────────────────────────────────────────────────────┐");
        System.out.println("│ STEP 2: Load weights in pure Java                               │");
        System.out.println("└─────────────────────────────────────────────────────────────────┘\n");

        JavaCNNInference cnn = new JavaCNNInference();
        System.out.println("  [Java]   → Loading weights from: " + weightsPath);
        cnn.loadWeights(weightsPath);
        System.out.println("  [Java]   ← Weights loaded successfully\n");

        // Step 3: Run inference in Java
        System.out.println("┌─────────────────────────────────────────────────────────────────┐");
        System.out.println("│ STEP 3: Run inference in pure Java (CPU)                        │");
        System.out.println("└─────────────────────────────────────────────────────────────────┘\n");

        // Create some test images (simple patterns representing digits)
        System.out.println("  Testing with synthetic digit patterns:\n");

        // Test multiple times to show consistent speed
        int[][] testDigits = {
            createDigitPattern(0),
            createDigitPattern(1),
            createDigitPattern(7),
        };
        int[] expectedLabels = {0, 1, 7};

        for (int i = 0; i < testDigits.length; i++) {
            PredictResult result = cnn.predictTimed(testDigits[i], 28, 28);
            System.out.printf("  [Java]   → Predicting digit pattern %d...%n", expectedLabels[i]);
            System.out.printf("  [Java]   ← %s%n%n", result);
        }

        // Benchmark
        System.out.println("┌─────────────────────────────────────────────────────────────────┐");
        System.out.println("│ BENCHMARK: 100 inferences                                       │");
        System.out.println("└─────────────────────────────────────────────────────────────────┘\n");

        int[] testImage = createDigitPattern(5);

        // Warmup
        for (int i = 0; i < 10; i++) {
            cnn.predict(testImage, 28, 28);
        }

        // Benchmark
        long start = System.nanoTime();
        int iterations = 100;
        for (int i = 0; i < iterations; i++) {
            cnn.predict(testImage, 28, 28);
        }
        long elapsed = System.nanoTime() - start;
        double avgMs = (elapsed / 1_000_000.0) / iterations;

        System.out.printf("  Total time: %.2fms for %d inferences%n", elapsed / 1_000_000.0, iterations);
        System.out.printf("  Average:    %.3fms per inference%n", avgMs);
        System.out.printf("  Throughput: %.0f inferences/second%n%n", 1000.0 / avgMs);

        System.out.println("┌─────────────────────────────────────────────────────────────────┐");
        System.out.println("│ SUMMARY                                                         │");
        System.out.println("├─────────────────────────────────────────────────────────────────┤");
        System.out.println("│ Training:  Python + PyTorch + GPU  (fast, ~3s/epoch)           │");
        System.out.println("│ Inference: Pure Java + CPU         (fast enough, ~1-2ms)       │");
        System.out.println("│                                                                 │");
        System.out.println("│ Benefits:                                                       │");
        System.out.println("│   • No Python needed at runtime for inference                  │");
        System.out.println("│   • Weights file is small (~100KB) and portable                │");
        System.out.println("│   • Works on any platform with Java                            │");
        System.out.println("│   • Easy to integrate into Java applications                   │");
        System.out.println("└─────────────────────────────────────────────────────────────────┘");
    }

    /**
     * Create a simple synthetic digit pattern for testing.
     */
    private static int[] createDigitPattern(int digit) {
        int[] pixels = new int[28 * 28];

        // Fill with white background
        java.util.Arrays.fill(pixels, 0);

        // Draw simple patterns
        switch (digit) {
            case 0 -> {
                // Draw oval
                for (int y = 6; y < 22; y++) {
                    for (int x = 8; x < 20; x++) {
                        double dx = (x - 14) / 5.0;
                        double dy = (y - 14) / 7.0;
                        double dist = dx * dx + dy * dy;
                        if (dist > 0.6 && dist < 1.2) {
                            pixels[y * 28 + x] = 255;
                        }
                    }
                }
            }
            case 1 -> {
                // Draw vertical line
                for (int y = 5; y < 23; y++) {
                    pixels[y * 28 + 14] = 255;
                    pixels[y * 28 + 15] = 255;
                }
            }
            case 7 -> {
                // Draw 7: horizontal top + diagonal
                for (int x = 8; x < 20; x++) {
                    pixels[6 * 28 + x] = 255;
                    pixels[7 * 28 + x] = 255;
                }
                for (int y = 6; y < 22; y++) {
                    int x = 18 - (y - 6) / 2;
                    pixels[y * 28 + x] = 255;
                    pixels[y * 28 + x + 1] = 255;
                }
            }
            default -> {
                // Draw X pattern
                for (int i = 0; i < 20; i++) {
                    pixels[(4 + i) * 28 + (4 + i)] = 255;
                    pixels[(4 + i) * 28 + (23 - i)] = 255;
                }
            }
        }

        return pixels;
    }
}
