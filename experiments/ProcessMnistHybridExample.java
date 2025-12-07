// ProcessMnistHybridExample.java
//
// JAVA + PYTHON HYBRID - Requires Python!
//
// Java spawns a Python subprocess to run PyTorch training (GPU-accelerated).
// After training, Python exports the weights and Java does inference.
//
// How it works:
//   Steps 1-7:  Python trains on GPU (MPS/CUDA)
//   Steps 8-10: Java loads weights and runs inference on CPU
//
// Performance:
//   Training:  ~3s/epoch (Python+GPU) vs ~7s/epoch (pure Java)
//   Inference: ~0.3ms/image (Java CPU) - fast enough for real-time
//
// Prerequisites:
//   pip install torch torchvision
//
// Run with:
//   mvn dependency:build-classpath -Dmdep.outputFile=/tmp/cp.txt -q
//   java -cp "$(cat /tmp/cp.txt):target/classes:experiments" ProcessMnistHybridExample
//

import java.io.*;
import java.nio.file.*;
import java.util.*;

public class ProcessMnistHybridExample {

    // MNIST normalization constants (must match Python)
    private static final float MNIST_MEAN = 0.1307f;
    private static final float MNIST_STD = 0.3081f;

    public static void main(String[] args) throws Exception {
        System.out.println("ProcessMnistHybridExample - Java orchestrates, Python trains, Java infers");
        System.out.println("=========================================================================\n");

        // Temp files for weight and test data exchange
        String weightsPath = "/tmp/mnist_hybrid.weights";
        String testDataPath = "/tmp/mnist_test_samples.bin";

        // ========================================================
        // Step 1: [JAVA] Prepare the Python training script
        //
        // Java holds the Python code as a string. This keeps
        // everything in one file for easy distribution.
        // ========================================================
        System.out.println("Step 1: [JAVA] Preparing Python training script...");

        String pythonScript = """
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import sys
import struct

# Unbuffered output
sys.stdout.reconfigure(line_buffering=True)

WEIGHTS_PATH = '%s'
TEST_DATA_PATH = '%s'

# ========================================================
# Step 2: [PYTHON] Detect GPU
# ========================================================
print("Step 2: [PYTHON] Detecting GPU...")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("  GPU: MPS (Apple Metal) - FAST!")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"  GPU: CUDA ({torch.cuda.get_device_name(0)}) - FAST!")
else:
    device = torch.device("cpu")
    print("  GPU: None - using CPU (slower)")

# ========================================================
# Step 3: [PYTHON] Define hyperparameters
# ========================================================
print("\\nStep 3: [PYTHON] Setting hyperparameters...")
batch_size = 128
epochs = 3
learning_rate = 0.001
print(f"  Batch size: {batch_size}")
print(f"  Epochs: {epochs}")
print(f"  Learning rate: {learning_rate}")

# ========================================================
# Step 4: [PYTHON] Load MNIST dataset
# ========================================================
print("\\nStep 4: [PYTHON] Loading MNIST dataset...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"  Training samples: {len(train_dataset)}")
print(f"  Test samples: {len(test_dataset)}")

# ========================================================
# Step 5: [PYTHON] Define CNN architecture
# ========================================================
print("\\nStep 5: [PYTHON] Defining CNN architecture...")
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Conv block 1: 1 -> 8 filters
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Conv block 2: 8 -> 16 filters
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Dense layers
        self.fc1 = nn.Linear(16 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN().to(device)
print("  Layer 0: Conv2D     1 -> 8 filters (3x3)")
print("  Layer 1: MaxPool    2x2")
print("  Layer 2: Conv2D     8 -> 16 filters (3x3)")
print("  Layer 3: MaxPool    2x2")
print("  Layer 4: Dense      400 -> 64")
print("  Layer 5: Output     64 -> 10")

# ========================================================
# Step 6: [PYTHON] Configure optimizer
# ========================================================
print("\\nStep 6: [PYTHON] Configuring optimizer...")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print("  Loss: CrossEntropyLoss")
print("  Optimizer: Adam")

# ========================================================
# Step 7: [PYTHON] Train the model (GPU-accelerated)
# ========================================================
print("\\nStep 7: [PYTHON] Training model on " + str(device) + "...")

total_start = time.time()

for epoch in range(epochs):
    epoch_start = time.time()
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    epoch_time = time.time() - epoch_start
    train_acc = 100. * correct / total
    avg_loss = running_loss / len(train_loader)
    print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Acc: {train_acc:.2f}%%, Time: {epoch_time:.2f}s")

total_time = time.time() - total_start
print(f"  Total training time: {total_time:.2f}s")

# ========================================================
# Step 7b: [PYTHON] Export weights for Java
# ========================================================
print("\\nStep 7b: [PYTHON] Exporting weights for Java inference...")

model.eval()
model.cpu()  # Move to CPU for export

with open(WEIGHTS_PATH, 'wb') as f:
    # Magic number "JCNN" and version
    f.write(struct.pack('>I', 0x4A434E4E))
    f.write(struct.pack('>I', 1))

    # Conv1 weights [8][1][3][3] and bias [8]
    conv1_w = model.conv1.weight.detach().numpy()
    conv1_b = model.conv1.bias.detach().numpy()
    for f_idx in range(8):
        for c in range(1):
            for h in range(3):
                for w in range(3):
                    f.write(struct.pack('>f', conv1_w[f_idx, c, h, w]))
    for f_idx in range(8):
        f.write(struct.pack('>f', conv1_b[f_idx]))

    # Conv2 weights [16][8][3][3] and bias [16]
    conv2_w = model.conv2.weight.detach().numpy()
    conv2_b = model.conv2.bias.detach().numpy()
    for f_idx in range(16):
        for c in range(8):
            for h in range(3):
                for w in range(3):
                    f.write(struct.pack('>f', conv2_w[f_idx, c, h, w]))
    for f_idx in range(16):
        f.write(struct.pack('>f', conv2_b[f_idx]))

    # FC1 weights [64][400] and bias [64]
    fc1_w = model.fc1.weight.detach().numpy()
    fc1_b = model.fc1.bias.detach().numpy()
    for o in range(64):
        for i in range(400):
            f.write(struct.pack('>f', fc1_w[o, i]))
    for o in range(64):
        f.write(struct.pack('>f', fc1_b[o]))

    # FC2 weights [10][64] and bias [10]
    fc2_w = model.fc2.weight.detach().numpy()
    fc2_b = model.fc2.bias.detach().numpy()
    for o in range(10):
        for i in range(64):
            f.write(struct.pack('>f', fc2_w[o, i]))
    for o in range(10):
        f.write(struct.pack('>f', fc2_b[o]))

print(f"  Weights saved to: {WEIGHTS_PATH}")

# ========================================================
# Step 7c: [PYTHON] Export test samples for Java
# ========================================================
print("\\nStep 7c: [PYTHON] Exporting test samples for Java...")

# Get test samples (raw pixels, not normalized - Java will normalize)
test_dataset_raw = datasets.MNIST('./data', train=False, transform=transforms.ToTensor())
test_loader_raw = DataLoader(test_dataset_raw, batch_size=100, shuffle=False)

with open(TEST_DATA_PATH, 'wb') as f:
    # Export 100 test samples
    data, labels = next(iter(test_loader_raw))

    # Number of samples
    f.write(struct.pack('>I', 100))

    for i in range(100):
        # Label
        f.write(struct.pack('>I', labels[i].item()))
        # Pixels (28x28 = 784 bytes, raw 0-255)
        pixels = (data[i, 0] * 255).byte().numpy()
        f.write(pixels.tobytes())

print(f"  Test samples saved to: {TEST_DATA_PATH}")
print("\\nPython training complete. Handing off to Java for inference...")
""".formatted(weightsPath, testDataPath);

        // Write script to temp file
        File tempScript = File.createTempFile("mnist_train", ".py");
        tempScript.deleteOnExit();
        try (PrintWriter writer = new PrintWriter(tempScript)) {
            writer.print(pythonScript);
        }
        System.out.println("  Script ready (" + pythonScript.length() + " chars)\n");

        // ========================================================
        // Step 1b: [JAVA] Launch Python subprocess for training
        //
        // Java spawns Python and streams its output in real-time.
        // Python runs Steps 2-7, then exports weights for Java.
        // ========================================================
        System.out.println("Step 1b: [JAVA] Launching Python subprocess for training...\n");

        ProcessBuilder pb = new ProcessBuilder("python3", tempScript.getAbsolutePath());
        pb.redirectErrorStream(true);
        pb.environment().put("PYTHONUNBUFFERED", "1");

        Process process = pb.start();

        // Stream output in real-time
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println(line);
            }
        }

        int exitCode = process.waitFor();
        if (exitCode != 0) {
            System.err.println("Python training failed with exit code: " + exitCode);
            return;
        }

        System.out.println();

        // ========================================================
        // Step 8: [JAVA] Load weights and evaluate on test set
        //
        // Now Java takes over. We load the weights Python exported
        // and run inference entirely in Java - no Python needed.
        // ========================================================
        System.out.println("Step 8: [JAVA] Loading weights and evaluating on test set...");

        // Load weights
        float[][][][] conv1Weights = new float[8][1][3][3];
        float[] conv1Bias = new float[8];
        float[][][][] conv2Weights = new float[16][8][3][3];
        float[] conv2Bias = new float[16];
        float[][] fc1Weights = new float[64][400];
        float[] fc1Bias = new float[64];
        float[][] fc2Weights = new float[10][64];
        float[] fc2Bias = new float[10];

        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(weightsPath)))) {
            int magic = dis.readInt();
            if (magic != 0x4A434E4E) {
                throw new IOException("Invalid weight file format");
            }
            int version = dis.readInt();

            // Conv1
            for (int f = 0; f < 8; f++)
                for (int c = 0; c < 1; c++)
                    for (int h = 0; h < 3; h++)
                        for (int w = 0; w < 3; w++)
                            conv1Weights[f][c][h][w] = dis.readFloat();
            for (int f = 0; f < 8; f++)
                conv1Bias[f] = dis.readFloat();

            // Conv2
            for (int f = 0; f < 16; f++)
                for (int c = 0; c < 8; c++)
                    for (int h = 0; h < 3; h++)
                        for (int w = 0; w < 3; w++)
                            conv2Weights[f][c][h][w] = dis.readFloat();
            for (int f = 0; f < 16; f++)
                conv2Bias[f] = dis.readFloat();

            // FC1
            for (int o = 0; o < 64; o++)
                for (int i = 0; i < 400; i++)
                    fc1Weights[o][i] = dis.readFloat();
            for (int o = 0; o < 64; o++)
                fc1Bias[o] = dis.readFloat();

            // FC2
            for (int o = 0; o < 10; o++)
                for (int i = 0; i < 64; i++)
                    fc2Weights[o][i] = dis.readFloat();
            for (int o = 0; o < 10; o++)
                fc2Bias[o] = dis.readFloat();
        }
        System.out.println("  Weights loaded from: " + weightsPath);

        // Load test samples
        int numSamples;
        int[] labels;
        int[][] pixels;

        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(testDataPath)))) {
            numSamples = dis.readInt();
            labels = new int[numSamples];
            pixels = new int[numSamples][784];

            for (int i = 0; i < numSamples; i++) {
                labels[i] = dis.readInt();
                byte[] rawPixels = new byte[784];
                dis.readFully(rawPixels);
                for (int j = 0; j < 784; j++) {
                    pixels[i][j] = rawPixels[j] & 0xFF;
                }
            }
        }
        System.out.println("  Test samples loaded: " + numSamples);

        // Evaluate on test set
        System.out.println("  Running inference on " + numSamples + " test samples...");
        long evalStart = System.nanoTime();
        int correct = 0;
        for (int i = 0; i < numSamples; i++) {
            int predicted = predict(pixels[i], conv1Weights, conv1Bias, conv2Weights, conv2Bias,
                                    fc1Weights, fc1Bias, fc2Weights, fc2Bias);
            if (predicted == labels[i]) {
                correct++;
            }
        }
        long evalTime = System.nanoTime() - evalStart;
        double accuracy = 100.0 * correct / numSamples;
        double avgInferenceMs = (evalTime / 1_000_000.0) / numSamples;

        System.out.println("  Test Accuracy: " + String.format("%.2f", accuracy) + "% (" + correct + "/" + numSamples + ")");
        System.out.println("  Avg inference time: " + String.format("%.3f", avgInferenceMs) + "ms per image");

        // ========================================================
        // Step 9: [JAVA] Single sample prediction
        // ========================================================
        System.out.println("\nStep 9: [JAVA] Single sample prediction...");

        int sampleIdx = 0;
        float[] probs = predictWithProbs(pixels[sampleIdx], conv1Weights, conv1Bias, conv2Weights, conv2Bias,
                                          fc1Weights, fc1Bias, fc2Weights, fc2Bias);
        int predicted = argmax(probs);

        System.out.println("  Actual: " + labels[sampleIdx]);
        System.out.println("  Predicted: " + predicted);
        System.out.println("  Confidence: " + String.format("%.2f", probs[predicted] * 100) + "%");

        // ========================================================
        // Step 10: [JAVA] Complete!
        // ========================================================
        System.out.println("\nStep 10: [JAVA] Complete!");
        System.out.println("  Summary:");
        System.out.println("    - Steps 2-7:  Python + PyTorch + GPU (training)");
        System.out.println("    - Steps 8-9:  Pure Java + CPU (inference)");
        System.out.println("    - Best of both worlds: fast training + portable inference");
    }

    // ========== Neural Network Operations (Pure Java) ==========

    private static int predict(int[] imagePixels,
                               float[][][][] conv1W, float[] conv1B,
                               float[][][][] conv2W, float[] conv2B,
                               float[][] fc1W, float[] fc1B,
                               float[][] fc2W, float[] fc2B) {
        float[] probs = predictWithProbs(imagePixels, conv1W, conv1B, conv2W, conv2B, fc1W, fc1B, fc2W, fc2B);
        return argmax(probs);
    }

    private static float[] predictWithProbs(int[] imagePixels,
                                            float[][][][] conv1W, float[] conv1B,
                                            float[][][][] conv2W, float[] conv2B,
                                            float[][] fc1W, float[] fc1B,
                                            float[][] fc2W, float[] fc2B) {
        // Normalize input
        float[][] input = new float[28][28];
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                float pixel = imagePixels[y * 28 + x] / 255.0f;
                input[y][x] = (pixel - MNIST_MEAN) / MNIST_STD;
            }
        }

        // Conv1 + ReLU: [1][28][28] -> [8][26][26]
        float[][][] conv1Out = conv2d(new float[][][]{input}, conv1W, conv1B);
        relu3d(conv1Out);

        // Pool1: [8][26][26] -> [8][13][13]
        float[][][] pool1Out = maxPool2d(conv1Out, 2, 2);

        // Conv2 + ReLU: [8][13][13] -> [16][11][11]
        float[][][] conv2Out = conv2d(pool1Out, conv2W, conv2B);
        relu3d(conv2Out);

        // Pool2: [16][11][11] -> [16][5][5]
        float[][][] pool2Out = maxPool2d(conv2Out, 2, 2);

        // Flatten: [16][5][5] -> [400]
        float[] flat = flatten(pool2Out);

        // FC1 + ReLU: [400] -> [64]
        float[] fc1Out = linear(flat, fc1W, fc1B);
        relu1d(fc1Out);

        // FC2: [64] -> [10]
        float[] fc2Out = linear(fc1Out, fc2W, fc2B);

        // Softmax
        return softmax(fc2Out);
    }

    private static float[][][] conv2d(float[][][] input, float[][][][] weights, float[] bias) {
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

    private static float[][][] maxPool2d(float[][][] input, int poolH, int poolW) {
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

    private static float[] linear(float[] input, float[][] weights, float[] bias) {
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

    private static float[] flatten(float[][][] input) {
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

    private static void relu3d(float[][][] arr) {
        for (int c = 0; c < arr.length; c++) {
            for (int h = 0; h < arr[0].length; h++) {
                for (int w = 0; w < arr[0][0].length; w++) {
                    if (arr[c][h][w] < 0) arr[c][h][w] = 0;
                }
            }
        }
    }

    private static void relu1d(float[] arr) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] < 0) arr[i] = 0;
        }
    }

    private static float[] softmax(float[] input) {
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

    private static int argmax(float[] arr) {
        int maxIdx = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[maxIdx]) {
                maxIdx = i;
            }
        }
        return maxIdx;
    }
}
