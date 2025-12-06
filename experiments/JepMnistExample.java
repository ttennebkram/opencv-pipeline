// JepMnistExample.java
//
// A CNN on MNIST using Jep to call Python PyTorch with MPS (Metal) GPU acceleration.
// This gives us access to Apple Silicon GPU from Java!
//
// Prerequisites:
//   1. pip install jep
//   2. pip install torch torchvision
//
// Run with:
//   java -Djava.library.path=$(python -c "import jep; print(jep.__path__[0])") \
//        -cp "$(cat /tmp/cp.txt):experiments" JepMnistExample

import jep.Interpreter;
import jep.SharedInterpreter;

public class JepMnistExample {

    public static void main(String[] args) {
        System.out.println("Starting Jep + PyTorch MNIST Example");
        System.out.println("=====================================\n");

        try (Interpreter interp = new SharedInterpreter()) {

            // Python code as a string - the full MNIST CNN training
            String pythonCode = """
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

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
print(f"All probabilities: {[f'{p:.3f}' for p in probs[0].cpu().numpy()]}")
""";

            // Execute the Python code
            interp.exec(pythonCode);

            System.out.println("\n=====================================");
            System.out.println("Jep + PyTorch execution complete!");

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();

            System.err.println("\n--- Troubleshooting ---");
            System.err.println("1. Make sure you installed jep: pip install jep");
            System.err.println("2. Make sure you have PyTorch: pip install torch torchvision");
            System.err.println("3. Run with the java.library.path set:");
            System.err.println("   java -Djava.library.path=$(python -c \"import jep; print(jep.__path__[0])\") ...");
        }
    }
}
