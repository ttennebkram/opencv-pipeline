// ProcessMnistHybridExample.java
//
// JAVA + PYTHON HYBRID - Requires Python!
//
// Java spawns a Python subprocess to run PyTorch training.
// This gives us GPU acceleration (MPS on Mac, CUDA on NVIDIA) that
// pure Java solutions (DL4J, DJL) don't support.
//
// How it works:
//   1. Java embeds Python code as a string
//   2. Java writes it to a temp file
//   3. Java runs "python3 tempfile.py" via ProcessBuilder
//   4. Java streams the output in real-time
//
// Performance: ~3s/epoch (vs ~7s/epoch for pure Java)
//
// Prerequisites:
//   pip install torch torchvision
//
// Run with:
//   mvn dependency:build-classpath -Dmdep.outputFile=/tmp/cp.txt -q
//   java -cp "$(cat /tmp/cp.txt):target/classes:experiments" ProcessMnistHybridExample
//

import java.io.*;

public class ProcessMnistHybridExample {

    public static void main(String[] args) throws Exception {
        System.out.println("ProcessMnistHybridExample - Java orchestrates, Python computes");
        System.out.println("==============================================================\n");

        // ========================================================
        // Step 1: [JAVA] Prepare the Python script
        //
        // Java holds the Python code as a string. This keeps
        // everything in one file for easy distribution.
        // ========================================================
        System.out.println("Step 1: [JAVA] Preparing Python script...");

        String pythonScript = """
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import sys

# Unbuffered output
sys.stdout.reconfigure(line_buffering=True)

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
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Acc: {train_acc:.2f}%, Time: {epoch_time:.2f}s")

total_time = time.time() - total_start
print(f"  Total training time: {total_time:.2f}s")

# ========================================================
# Step 8: [PYTHON] Evaluate on test set
# ========================================================
print("\\nStep 8: [PYTHON] Evaluating on test set...")

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

test_acc = 100. * correct / total
print(f"  Test Accuracy: {test_acc:.2f}%")
print(f"  Test Loss: {test_loss/len(test_loader):.4f}")

# ========================================================
# Step 9: [PYTHON] Single sample prediction
# ========================================================
print("\\nStep 9: [PYTHON] Single sample prediction...")

test_iter = iter(test_loader)
data, target = next(test_iter)
single_image = data[0:1].to(device)
single_label = target[0].item()

with torch.no_grad():
    output = model(single_image)
    probs = torch.softmax(output, dim=1)
    predicted = output.argmax(dim=1).item()

print(f"  Actual: {single_label}")
print(f"  Predicted: {predicted}")
print(f"  Confidence: {probs[0][predicted].item()*100:.2f}%")
""";

        // Write script to temp file
        File tempScript = File.createTempFile("mnist_train", ".py");
        tempScript.deleteOnExit();
        try (PrintWriter writer = new PrintWriter(tempScript)) {
            writer.print(pythonScript);
        }
        System.out.println("  Script ready (" + pythonScript.length() + " chars)\n");

        // ========================================================
        // Step 1b: [JAVA] Launch Python subprocess
        //
        // Java spawns Python and streams its output in real-time.
        // From here, Steps 2-9 run in Python with GPU acceleration.
        // ========================================================
        System.out.println("Step 1b: [JAVA] Launching Python subprocess...\n");

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

        // ========================================================
        // Step 10: [JAVA] Done!
        // ========================================================
        System.out.println("\nStep 10: [JAVA] Complete!");
        System.out.println("  Python process exited with code: " + exitCode);
    }
}
