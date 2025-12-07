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
        System.out.println("Starting Python PyTorch MNIST via subprocess");
        System.out.println("=============================================\n");

        // Python script as heredoc
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

# Check for MPS (Metal) GPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using MPS (Metal) GPU acceleration!")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA GPU")
else:
    device = torch.device("cpu")
    print(f"Using CPU")

print(f"Device: {device}")

# Hyperparameters
batch_size = 128
epochs = 3
learning_rate = 0.001

# Data transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST
print("\\nLoading MNIST dataset...")
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

# Define CNN (same architecture as Java versions)
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

# Create model
model = CNN().to(device)
print(f"\\nModel architecture:")
print(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
print("\\n" + "="*50)
print("Starting training...")
print("="*50)

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
print(f"\\nTotal training time: {total_time:.2f}s")

# Evaluation
print("\\n" + "="*50)
print("Evaluating on test set...")
print("="*50)

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
print(f"Test Accuracy: {test_acc:.2f}%")
print(f"Test Loss: {test_loss/len(test_loader):.4f}")

# Single prediction
print("\\n" + "="*50)
print("Single sample prediction:")
print("="*50)

test_iter = iter(test_loader)
data, target = next(test_iter)
single_image = data[0:1].to(device)
single_label = target[0].item()

with torch.no_grad():
    output = model(single_image)
    probs = torch.softmax(output, dim=1)
    predicted = output.argmax(dim=1).item()

print(f"Actual: {single_label}")
print(f"Predicted: {predicted}")
print(f"Confidence: {probs[0][predicted].item()*100:.2f}%")
""";

        // Write script to temp file
        File tempScript = File.createTempFile("mnist_train", ".py");
        tempScript.deleteOnExit();
        try (PrintWriter writer = new PrintWriter(tempScript)) {
            writer.print(pythonScript);
        }

        // Run Python
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

        System.out.println("\n=============================================");
        System.out.println("Python process exited with code: " + exitCode);
    }
}
