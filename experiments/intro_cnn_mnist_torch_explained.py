#!/usr/bin/env python3
"""
intro_cnn_mnist_torch_explained.py
This is a very slow, heavily-explained, beginner-friendly CNN example using PyTorch.
Every step prints what is happening and WHY.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


print("\n==========================")
print(" STEP 1: Defining a CNN ")
print("==========================\n")

print("A CNN (Convolutional Neural Network) works by scanning small filter windows")
print("across the image and learning edge-like patterns first (like curves or lines)")
print("and more complex patterns later (like shapes or digits).")
print("\nWe define a VERY simple network with:")
print(" • One convolution layer (detects patterns)")
print(" • One max-pooling layer (shrinks the image but keeps strong signals)")
print(" • One fully-connected layer (makes a final prediction: digit 0-9)\n")


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        print("Creating the CNN model now...\n")

        # Convolution Layer
        print("Adding a convolution layer:")
        print("  Input to this layer: 1 grayscale image channel (28x28)")
        print("  Output: 8 feature maps using 3x3 filters\n")
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)

        # Pooling Layer
        print("Adding a Max Pooling layer (2x2):")
        print("  This reduces the size of each feature map by HALF")
        print("  That reduces computation and keeps strongest features\n")
        self.pool = nn.MaxPool2d(2, 2)

        # Fully Connected / Linear Layer
        print("Connecting features to a final prediction layer:")
        print("  After conv + pool, image shrinks to size: 13 x 13 with 8 channels")
        print("  That is 8 * 13 * 13 = 1352 values per image")
        print("  We feed that into a dense layer that outputs 10 classes (digits 0-9)\n")
        self.fc = nn.Linear(8 * 13 * 13, 10)

    def forward(self, x):
        print("\nForward pass started for one batch...")

        print("Shape BEFORE convolution:", tuple(x.shape))
        x = self.conv1(x)
        print("Shape AFTER convolution:", tuple(x.shape), "<-- features detected")

        x = F.relu(x)
        print("Applied ReLU activation (keeps positives, zeroes negatives).")

        x = self.pool(x)
        print("Shape AFTER max pooling:", tuple(x.shape), "<-- shrunk but informative")

        x = x.view(x.size(0), -1)
        print("Flatten into 1D vector per image:", tuple(x.shape))

        x = self.fc(x)
        print("Output logits:", tuple(x.shape), "<-- raw scores before softmax")

        print("Forward pass complete.\n")
        return x


print("\n===============================")
print(" STEP 2: Loading the MNIST DATA")
print("===============================\n")

print("MNIST is a set of 70,000 handwritten digit images.")
print("Each image is 28x28 pixels, grayscale, digit 0 through 9.")

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)


print("\nCreating DataLoader (batches the dataset so we can train faster)")
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False)


print("\n===================================")
print(" STEP 3: Creating Model + Optimizer")
print("===================================\n")

device = torch.device("cpu")
print("Using CPU — totally fine for this small model.\n")

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nOptimizer: Adam")
print("Learning Rate: 0.001")
print("Loss function: CrossEntropyLoss (good for multi-class classification)\n")


print("\n=======================")
print(" STEP 4: Training begin")
print("=======================\n")

print("We train for a few epochs (passes over the whole dataset)")
print("Each training step attempts to reduce loss — i.e.,")
print("the difference between prediction and correct label.\n")

for epoch in range(1, 4):
    running_loss = 0.0

    print(f"---- Epoch {epoch} ----")

    for batch_index, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_index % 50 == 0:
            print(f"Batch {batch_index}: current loss = {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    print(f"Average loss for epoch {epoch}: {avg_loss:.4f}\n")


print("\n========================")
print(" STEP 5: Testing accuracy")
print("========================\n")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print("This means: For every 100 test digits, the network got about")
print(f"{accuracy*100:.2f} correct.\n")


print("\n==============================")
print(" STEP 6: Predicting one sample")
print("==============================\n")

sample_img, sample_label = test_dataset[0]

print("Taking the FIRST test image and asking the model what digit it sees.")
model.eval()
with torch.no_grad():
    logits = model(sample_img.unsqueeze(0).to(device))
    probs = torch.softmax(logits, dim=1)[0]
    pred_label = int(torch.argmax(probs).item())

print("True label (correct digit):", sample_label)
print("Predicted digit by CNN:", pred_label)
print("Raw class probabilities (0-9):", probs.tolist())

print("\n")
print("===========================================================")
print(" FINISHED — You just trained and tested a CNN step-by-step!")
print("===========================================================\n")


