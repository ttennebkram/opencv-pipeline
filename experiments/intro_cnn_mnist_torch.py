#!/usr/bin/env python3
"""
intro_cnn_mnist_torch.py
Minimal CNN for MNIST using PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 input channel (grayscale), 8 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        # After conv + pool: 28x28 -> 26x26 -> 13x13, with 8 channels
        self.fc = nn.Linear(8 * 13 * 13, 10)  # 10 classes (digits 0-9)

    def forward(self, x):
        x = self.conv1(x)      # (N, 1, 28, 28) -> (N, 8, 26, 26)
        x = F.relu(x)
        x = self.pool(x)       # -> (N, 8, 13, 13)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)         # -> (N, 10)
        return x


def main():
    # Use CPU; this is tiny and fast enough
    device = torch.device("cpu")

    # 1. Dataset and transforms
    transform = transforms.Compose(
        [
            transforms.ToTensor(),               # (H,W) -> (1,H,W), scaled to [0,1]
        ]
    )

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # 2. Model, loss, optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 3. Training loop (few epochs)
    model.train()
    for epoch in range(1, 4):  # 3 epochs
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch}: loss = {avg_loss:.4f}")

    # 4. Evaluation on test set
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # (N, 10)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"\nTest accuracy: {accuracy:.4f}")

    # 5. Single example prediction
    sample_img, sample_label = test_dataset[0]
    model.eval()
    with torch.no_grad():
        logits = model(sample_img.unsqueeze(0).to(device))  # add batch dimension
        probs = torch.softmax(logits, dim=1)[0]
        pred_label = int(torch.argmax(probs).item())

    print("\nExample prediction:")
    print("  True label:     ", sample_label)
    print("  Predicted label:", pred_label)
    print("  Probabilities:  ", probs.tolist())


if __name__ == "__main__":
    main()


