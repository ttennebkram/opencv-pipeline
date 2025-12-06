#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


print("\n==========================")
print(" STEP 1: Defining a CNN ")
print("==========================\n")

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        print("Creating the CNN model...\n")

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(8 * 13 * 13, 10)

    def forward(self, x):
        print("\nForward pass started...")
        print("Input batch shape BEFORE convolution:", tuple(x.shape))

        x = self.conv1(x)
        print("Shape AFTER convolution:", tuple(x.shape))

        x = F.relu(x)
        print("ReLU activation applied.")

        x = self.pool(x)
        print("Shape AFTER max pooling:", tuple(x.shape))

        x = x.view(x.size(0), -1)
        print("Flatten to vector:", tuple(x.shape))

        x = self.fc(x)
        print("Output logits:", tuple(x.shape))
        print("Forward pass complete.\n")
        return x


print("\n===============================")
print(" STEP 2: Loading MNIST DATA")
print("===============================\n")

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False)


print("\n===================================")
print(" STEP 3: Creating Model + Optimizer")
print("===================================\n")

device = torch.device("cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


print("\n=======================")
print(" STEP 4: Training begin")
print("=======================\n")

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

    # 🔵 ADDING YOUR REQUESTED SEARCHABLE MARKER 🔵
    print(f"=== END OF EPOCH ({epoch}) ===\n")


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
print(f"Test Accuracy: {accuracy * 100:.2f}%\n")


print("\n==============================")
print(" STEP 6: Predicting one sample")
print("==============================\n")

sample_img, sample_label = test_dataset[0]
with torch.no_grad():
    logits = model(sample_img.unsqueeze(0).to(device))
    probs = torch.softmax(logits, dim=1)[0]
    pred_label = int(torch.argmax(probs).item())

print("True digit:     ", sample_label)
print("Predicted digit:", pred_label)
print("Probabilities:  ", probs.tolist())
print("\n========== TRAINING COMPLETE ==========\n")


