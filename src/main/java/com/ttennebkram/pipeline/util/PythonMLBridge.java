package com.ttennebkram.pipeline.util;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.function.Consumer;

/**
 * Bridge for calling Python ML operations step-by-step.
 * Keeps model state in Python between calls, orchestration in Java.
 *
 * Usage:
 *   PythonMLBridge bridge = new PythonMLBridge();
 *   bridge.initialize();
 *   bridge.createModel("cnn", 10);
 *   bridge.loadDataset("mnist", 128);
 *   for (int i = 0; i < epochs; i++) {
 *       TrainResult result = bridge.trainEpoch();
 *       System.out.println("Epoch " + i + ": " + result.accuracy);
 *   }
 *   float[] probs = bridge.predict(imageData);
 *   bridge.shutdown();
 */
public class PythonMLBridge {

    private Process pythonProcess;
    private BufferedWriter toProcess;
    private BufferedReader fromProcess;
    private boolean initialized = false;
    private String deviceType = "cpu";
    private Path workDir;

    // ========== Lifecycle ==========

    /**
     * Start the Python bridge process.
     */
    public void initialize() throws IOException {
        initialize(null);
    }

    public void initialize(Consumer<String> logHandler) throws IOException {
        if (initialized) {
            return;
        }

        // Create work directory for data exchange
        workDir = Files.createTempDirectory("python_ml_bridge_");

        // Write the Python bridge script
        Path scriptPath = workDir.resolve("bridge.py");
        Files.writeString(scriptPath, PYTHON_BRIDGE_SCRIPT);

        // Start Python process
        ProcessBuilder pb = new ProcessBuilder("python3", scriptPath.toString());
        pb.environment().put("PYTHONUNBUFFERED", "1");
        pb.redirectErrorStream(false);  // Keep stderr separate

        pythonProcess = pb.start();
        toProcess = new BufferedWriter(new OutputStreamWriter(pythonProcess.getOutputStream()));
        fromProcess = new BufferedReader(new InputStreamReader(pythonProcess.getInputStream()));

        // Start error reader thread
        BufferedReader errorReader = new BufferedReader(new InputStreamReader(pythonProcess.getErrorStream()));
        Thread errorThread = new Thread(() -> {
            try {
                String line;
                while ((line = errorReader.readLine()) != null) {
                    if (logHandler != null) {
                        logHandler.accept("[Python] " + line);
                    }
                }
            } catch (IOException e) {
                // Process ended
            }
        });
        errorThread.setDaemon(true);
        errorThread.start();

        // Wait for ready signal and get device type
        String response = sendCommand("INIT");
        if (response.startsWith("READY:")) {
            deviceType = response.substring(6).trim();
            initialized = true;
        } else {
            throw new IOException("Failed to initialize Python bridge: " + response);
        }
    }

    /**
     * Shutdown the Python bridge.
     */
    public void shutdown() {
        if (!initialized) return;

        try {
            sendCommand("QUIT");
        } catch (Exception e) {
            // Ignore
        }

        try {
            toProcess.close();
            fromProcess.close();
            pythonProcess.destroyForcibly();
        } catch (Exception e) {
            // Ignore
        }

        // Cleanup work directory
        try {
            Files.walk(workDir)
                .sorted(Comparator.reverseOrder())
                .forEach(p -> {
                    try { Files.delete(p); } catch (IOException e) {}
                });
        } catch (IOException e) {
            // Ignore
        }

        initialized = false;
    }

    public boolean isInitialized() {
        return initialized;
    }

    public String getDeviceType() {
        return deviceType;
    }

    // ========== Model Operations ==========

    /**
     * Create a CNN model.
     * @param architecture "cnn" for standard CNN, or custom definition
     * @param numClasses number of output classes
     */
    public void createModel(String architecture, int numClasses) throws IOException {
        checkInitialized();
        String response = sendCommand("CREATE_MODEL:" + architecture + ":" + numClasses);
        if (!response.equals("OK")) {
            throw new IOException("Failed to create model: " + response);
        }
    }

    /**
     * Save model to file.
     */
    public void saveModel(String path) throws IOException {
        checkInitialized();
        String response = sendCommand("SAVE_MODEL:" + path);
        if (!response.equals("OK")) {
            throw new IOException("Failed to save model: " + response);
        }
    }

    /**
     * Load model from file.
     */
    public void loadModel(String path) throws IOException {
        checkInitialized();
        String response = sendCommand("LOAD_MODEL:" + path);
        if (!response.equals("OK")) {
            throw new IOException("Failed to load model: " + response);
        }
    }

    /**
     * Export model weights in format readable by JavaCNNInference.
     */
    public void exportWeightsForJava(String path) throws IOException {
        checkInitialized();
        String response = sendCommand("EXPORT_JAVA:" + path);
        if (!response.equals("OK")) {
            throw new IOException("Failed to export weights: " + response);
        }
    }

    // ========== Dataset Operations ==========

    /**
     * Load a built-in dataset.
     * @param name "mnist", "cifar10", etc.
     * @param batchSize batch size for training
     */
    public DatasetInfo loadDataset(String name, int batchSize) throws IOException {
        checkInitialized();
        String response = sendCommand("LOAD_DATASET:" + name + ":" + batchSize);
        // Response: OK:trainSize:testSize
        String[] parts = response.split(":");
        if (parts[0].equals("OK")) {
            return new DatasetInfo(
                Integer.parseInt(parts[1]),
                Integer.parseInt(parts[2])
            );
        } else {
            throw new IOException("Failed to load dataset: " + response);
        }
    }

    // ========== Training Operations ==========

    /**
     * Configure training parameters.
     */
    public void configureTraining(float learningRate) throws IOException {
        checkInitialized();
        String response = sendCommand("CONFIGURE_TRAINING:" + learningRate);
        if (!response.equals("OK")) {
            throw new IOException("Failed to configure training: " + response);
        }
    }

    /**
     * Train for one epoch.
     * @return training results for this epoch
     */
    public TrainResult trainEpoch() throws IOException {
        checkInitialized();
        String response = sendCommand("TRAIN_EPOCH");
        // Response: OK:loss:accuracy:timeMs
        String[] parts = response.split(":");
        if (parts[0].equals("OK")) {
            return new TrainResult(
                Float.parseFloat(parts[1]),
                Float.parseFloat(parts[2]),
                Long.parseLong(parts[3])
            );
        } else {
            throw new IOException("Training failed: " + response);
        }
    }

    /**
     * Evaluate on test set.
     */
    public EvalResult evaluate() throws IOException {
        checkInitialized();
        String response = sendCommand("EVALUATE");
        // Response: OK:loss:accuracy
        String[] parts = response.split(":");
        if (parts[0].equals("OK")) {
            return new EvalResult(
                Float.parseFloat(parts[1]),
                Float.parseFloat(parts[2])
            );
        } else {
            throw new IOException("Evaluation failed: " + response);
        }
    }

    // ========== Inference Operations ==========

    /**
     * Predict class probabilities for a single image.
     * @param imageData flattened grayscale image data (0-255)
     * @param width image width
     * @param height image height
     * @return probability for each class
     */
    public float[] predict(int[] imageData, int width, int height) throws IOException {
        checkInitialized();

        // Write image data to temp file (more efficient than string encoding)
        Path dataFile = workDir.resolve("input.bin");
        try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(dataFile.toFile()))) {
            dos.writeInt(width);
            dos.writeInt(height);
            for (int pixel : imageData) {
                dos.writeByte(pixel);
            }
        }

        String response = sendCommand("PREDICT:" + dataFile.toString());
        // Response: OK:prob0,prob1,prob2,...
        String[] parts = response.split(":");
        if (parts[0].equals("OK")) {
            String[] probStrs = parts[1].split(",");
            float[] probs = new float[probStrs.length];
            for (int i = 0; i < probStrs.length; i++) {
                probs[i] = Float.parseFloat(probStrs[i]);
            }
            return probs;
        } else {
            throw new IOException("Prediction failed: " + response);
        }
    }

    /**
     * Predict class for a single image.
     * @return predicted class index
     */
    public int predictClass(int[] imageData, int width, int height) throws IOException {
        float[] probs = predict(imageData, width, height);
        int maxIdx = 0;
        for (int i = 1; i < probs.length; i++) {
            if (probs[i] > probs[maxIdx]) {
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    // ========== Helper Methods ==========

    private void checkInitialized() {
        if (!initialized) {
            throw new IllegalStateException("Bridge not initialized. Call initialize() first.");
        }
    }

    private synchronized String sendCommand(String command) throws IOException {
        toProcess.write(command);
        toProcess.newLine();
        toProcess.flush();
        return fromProcess.readLine();
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

    // ========== Python Bridge Script ==========

    private static final String PYTHON_BRIDGE_SCRIPT = """
import sys
import time
import struct

# Unbuffered
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Detect device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    device_name = "mps"
elif torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = "cuda"
else:
    device = torch.device("cpu")
    device_name = "cpu"

print(f"Python bridge using device: {device_name}", file=sys.stderr)

# Global state
model = None
optimizer = None
criterion = nn.CrossEntropyLoss()
train_loader = None
test_loader = None

# CNN Architecture
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def handle_command(cmd):
    global model, optimizer, train_loader, test_loader

    parts = cmd.strip().split(":")

    if parts[0] == "INIT":
        return f"READY:{device_name}"

    elif parts[0] == "QUIT":
        return "BYE"

    elif parts[0] == "CREATE_MODEL":
        arch = parts[1]
        num_classes = int(parts[2])
        if arch == "cnn":
            model = CNN(num_classes).to(device)
        else:
            return f"ERROR:Unknown architecture {arch}"
        return "OK"

    elif parts[0] == "SAVE_MODEL":
        path = parts[1]
        torch.save(model.state_dict(), path)
        return "OK"

    elif parts[0] == "LOAD_MODEL":
        path = parts[1]
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return "OK"

    elif parts[0] == "LOAD_DATASET":
        name = parts[1]
        batch_size = int(parts[2])

        if name == "mnist":
            from torchvision import datasets, transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST('./data', train=False, transform=transform)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            return f"OK:{len(train_dataset)}:{len(test_dataset)}"
        else:
            return f"ERROR:Unknown dataset {name}"

    elif parts[0] == "CONFIGURE_TRAINING":
        lr = float(parts[1])
        optimizer = optim.Adam(model.parameters(), lr=lr)
        return "OK"

    elif parts[0] == "TRAIN_EPOCH":
        model.train()
        start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        for data, target in train_loader:
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

        elapsed = int((time.time() - start_time) * 1000)
        avg_loss = running_loss / len(train_loader)
        accuracy = correct / total
        return f"OK:{avg_loss:.6f}:{accuracy:.6f}:{elapsed}"

    elif parts[0] == "EVALUATE":
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        avg_loss = test_loss / len(test_loader)
        accuracy = correct / total
        return f"OK:{avg_loss:.6f}:{accuracy:.6f}"

    elif parts[0] == "EXPORT_JAVA":
        path = parts[1]
        import struct
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
                            f.write(struct.pack('>f', float(conv1_w[f_idx, c, h, w])))
            for f_idx in range(8):
                f.write(struct.pack('>f', float(conv1_b[f_idx])))

            # Conv2 weights [16][8][3][3] and bias [16]
            conv2_w = model.conv2.weight.detach().cpu().numpy()
            conv2_b = model.conv2.bias.detach().cpu().numpy()
            for f_idx in range(16):
                for c in range(8):
                    for h in range(3):
                        for w in range(3):
                            f.write(struct.pack('>f', float(conv2_w[f_idx, c, h, w])))
            for f_idx in range(16):
                f.write(struct.pack('>f', float(conv2_b[f_idx])))

            # FC1 weights [64][400] and bias [64]
            fc1_w = model.fc1.weight.detach().cpu().numpy()
            fc1_b = model.fc1.bias.detach().cpu().numpy()
            for o in range(64):
                for i in range(400):
                    f.write(struct.pack('>f', float(fc1_w[o, i])))
            for o in range(64):
                f.write(struct.pack('>f', float(fc1_b[o])))

            # FC2 weights [10][64] and bias [10]
            fc2_w = model.fc2.weight.detach().cpu().numpy()
            fc2_b = model.fc2.bias.detach().cpu().numpy()
            for o in range(10):
                for i in range(64):
                    f.write(struct.pack('>f', float(fc2_w[o, i])))
            for o in range(10):
                f.write(struct.pack('>f', float(fc2_b[o])))
        return "OK"

    elif parts[0] == "PREDICT":
        path = parts[1]
        with open(path, 'rb') as f:
            width = struct.unpack('>i', f.read(4))[0]
            height = struct.unpack('>i', f.read(4))[0]
            pixels = list(f.read(width * height))

        # Convert to tensor
        import numpy as np
        img = np.array(pixels, dtype=np.float32).reshape(1, 1, height, width)
        img = (img / 255.0 - 0.1307) / 0.3081  # MNIST normalization
        tensor = torch.from_numpy(img).to(device)

        model.eval()
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)[0]

        prob_strs = [f"{p:.6f}" for p in probs.cpu().numpy()]
        return "OK:" + ",".join(prob_strs)

    else:
        return f"ERROR:Unknown command {parts[0]}"

# Main loop
for line in sys.stdin:
    try:
        result = handle_command(line)
        print(result, flush=True)
        if result == "BYE":
            break
    except Exception as e:
        print(f"ERROR:{str(e)}", flush=True)
        print(f"Exception: {e}", file=sys.stderr)
""";

    // ========== Main for testing ==========

    public static void main(String[] args) throws Exception {
        System.out.println("╔══════════════════════════════════════════════════════════════════╗");
        System.out.println("║           PythonMLBridge - Java/Python ML Integration            ║");
        System.out.println("╚══════════════════════════════════════════════════════════════════╝\n");

        System.out.println("This demo shows how Java can orchestrate ML training step-by-step,");
        System.out.println("while Python/PyTorch handles the GPU-accelerated computation.\n");

        PythonMLBridge bridge = new PythonMLBridge();

        try {
            // ============ Step 1: Initialize ============
            System.out.println("┌─────────────────────────────────────────────────────────────────┐");
            System.out.println("│ STEP 1: Initialize Python Bridge                                │");
            System.out.println("├─────────────────────────────────────────────────────────────────┤");
            System.out.println("│ Java starts a Python subprocess and establishes communication.  │");
            System.out.println("│ Python detects available GPU (MPS on Mac, CUDA on NVIDIA).      │");
            System.out.println("└─────────────────────────────────────────────────────────────────┘");
            System.out.println();
            System.out.println("  [Java]   → Starting Python process...");
            bridge.initialize(msg -> System.out.println("  [Python] ← " + msg));
            System.out.println("  [Python] ← Ready! Device: " + bridge.getDeviceType());
            String deviceDesc = switch (bridge.getDeviceType()) {
                case "mps" -> "Apple Silicon GPU (Metal Performance Shaders)";
                case "cuda" -> "NVIDIA GPU (CUDA)";
                default -> "CPU (no GPU acceleration)";
            };
            System.out.println("  [Info]   = " + deviceDesc);
            System.out.println();

            // ============ Step 2: Create Model ============
            System.out.println("┌─────────────────────────────────────────────────────────────────┐");
            System.out.println("│ STEP 2: Create CNN Model                                        │");
            System.out.println("├─────────────────────────────────────────────────────────────────┤");
            System.out.println("│ Java tells Python to create a Convolutional Neural Network.    │");
            System.out.println("│ The model is created on the GPU and stays in Python memory.    │");
            System.out.println("│                                                                 │");
            System.out.println("│ Architecture: Conv(1→8) → Pool → Conv(8→16) → Pool → FC → 10   │");
            System.out.println("└─────────────────────────────────────────────────────────────────┘");
            System.out.println();
            System.out.println("  [Java]   → createModel(\"cnn\", numClasses=10)");
            bridge.createModel("cnn", 10);
            System.out.println("  [Python] ← Model created on " + bridge.getDeviceType());
            System.out.println();

            // ============ Step 3: Load Dataset ============
            System.out.println("┌─────────────────────────────────────────────────────────────────┐");
            System.out.println("│ STEP 3: Load MNIST Dataset                                      │");
            System.out.println("├─────────────────────────────────────────────────────────────────┤");
            System.out.println("│ Python downloads/loads MNIST (handwritten digits 0-9).         │");
            System.out.println("│ Data stays in Python; Java receives only the counts.           │");
            System.out.println("└─────────────────────────────────────────────────────────────────┘");
            System.out.println();
            System.out.println("  [Java]   → loadDataset(\"mnist\", batchSize=128)");
            DatasetInfo info = bridge.loadDataset("mnist", 128);
            System.out.println("  [Python] ← Dataset loaded:");
            System.out.println("             Training samples: " + info.trainSize);
            System.out.println("             Test samples:     " + info.testSize);
            System.out.println();

            // ============ Step 4: Configure Training ============
            System.out.println("┌─────────────────────────────────────────────────────────────────┐");
            System.out.println("│ STEP 4: Configure Training                                      │");
            System.out.println("├─────────────────────────────────────────────────────────────────┤");
            System.out.println("│ Java sets hyperparameters. Python creates the optimizer.       │");
            System.out.println("└─────────────────────────────────────────────────────────────────┘");
            System.out.println();
            float learningRate = 0.001f;
            System.out.println("  [Java]   → configureTraining(learningRate=" + learningRate + ")");
            bridge.configureTraining(learningRate);
            System.out.println("  [Python] ← Adam optimizer configured");
            System.out.println();

            // ============ Step 5: Training Loop ============
            System.out.println("┌─────────────────────────────────────────────────────────────────┐");
            System.out.println("│ STEP 5: Training Loop (Java-controlled)                         │");
            System.out.println("├─────────────────────────────────────────────────────────────────┤");
            System.out.println("│ Java controls the epoch loop. Each trainEpoch() call:          │");
            System.out.println("│   1. Python runs forward pass (on GPU)                         │");
            System.out.println("│   2. Python computes loss                                       │");
            System.out.println("│   3. Python runs backward pass (gradients, on GPU)             │");
            System.out.println("│   4. Python updates weights                                     │");
            System.out.println("│   5. Returns loss/accuracy to Java                             │");
            System.out.println("│                                                                 │");
            System.out.println("│ Java can update UI, check for cancel, log progress, etc.       │");
            System.out.println("└─────────────────────────────────────────────────────────────────┘");
            System.out.println();

            int epochs = 3;
            for (int epoch = 1; epoch <= epochs; epoch++) {
                System.out.println("  [Java]   → trainEpoch() - Epoch " + epoch + "/" + epochs);
                System.out.println("             (Python is training on GPU...)");
                TrainResult result = bridge.trainEpoch();
                System.out.println("  [Python] ← " + result);
                System.out.println();

                // Simulate Java doing something between epochs
                if (epoch < epochs) {
                    System.out.println("  [Java]   = (Here Java could update progress bar, check cancel, etc.)");
                    System.out.println();
                }
            }

            // ============ Step 6: Evaluate ============
            System.out.println("┌─────────────────────────────────────────────────────────────────┐");
            System.out.println("│ STEP 6: Evaluate on Test Set                                    │");
            System.out.println("├─────────────────────────────────────────────────────────────────┤");
            System.out.println("│ Python runs inference on all test samples (no gradient calc).  │");
            System.out.println("│ Returns final accuracy to Java.                                │");
            System.out.println("└─────────────────────────────────────────────────────────────────┘");
            System.out.println();
            System.out.println("  [Java]   → evaluate()");
            EvalResult eval = bridge.evaluate();
            System.out.println("  [Python] ← " + eval);
            System.out.println();

            // ============ Summary ============
            System.out.println("┌─────────────────────────────────────────────────────────────────┐");
            System.out.println("│ SUMMARY                                                         │");
            System.out.println("├─────────────────────────────────────────────────────────────────┤");
            System.out.printf("│ Final Test Accuracy: %5.2f%%                                    │%n", eval.accuracy * 100);
            System.out.println("│                                                                 │");
            System.out.println("│ What ran in Java:                                               │");
            System.out.println("│   • Orchestration (start, stop, loop control)                  │");
            System.out.println("│   • Hyperparameter decisions                                   │");
            System.out.println("│   • Progress tracking and display                              │");
            System.out.println("│                                                                 │");
            System.out.println("│ What ran in Python (GPU-accelerated):                          │");
            System.out.println("│   • Model creation and storage                                 │");
            System.out.println("│   • Dataset loading and batching                               │");
            System.out.println("│   • Forward/backward passes                                    │");
            System.out.println("│   • Weight updates                                             │");
            System.out.println("│   • Inference                                                  │");
            System.out.println("└─────────────────────────────────────────────────────────────────┘");

        } finally {
            System.out.println();
            System.out.println("  [Java]   → shutdown()");
            bridge.shutdown();
            System.out.println("  [Python] ← Process terminated");
        }

        System.out.println("\nDone!");
    }
}
