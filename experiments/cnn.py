#!/usr/bin/env python3
"""
cnn_dim_traced_with_analogy.py
Two-block CNN with dimensional tracing and analogy comments included.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ================================================================
# Step 1:
# "The architect writes the blueprint"
#
# In this step, we DEFINE the neural network architecture:
#   - We describe what layers exist (Conv, Pool, Fully Connected)
#   - We describe how the data flows through the layers
#
# IMPORTANT:
#   This step does NOT create the working trained model.
#   This step only describes the design (the PLAN) of the model.
#
# Analogy:
#   - An architect can draw:
#          2 bathrooms,
#          a garage,
#          a kitchen,
#     ...but no house has been built yet.
#
# The class below = the blueprint, NOT the house.
# ================================================================

print("\n==========================")
print(" STEP 1: Defining a CNN ")
print("==========================\n")

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        print("Building CNN with dimension tracing...\n")

        # ===== INPUT SHAPE =====
        self.input_channels = 1    # e.g., grayscale → 1
        self.input_height   = 28   # MNIST → 28x28 pixels
        self.input_width    = 28

        print(f"Input Shape:")
        print(f"  channels = {self.input_channels}  # e.g., 1")
        print(f"  height   = {self.input_height}   # e.g., 28")
        print(f"  width    = {self.input_width}    # e.g., 28\n")

        # ----------------------------------------------------------
        # Convolution Layer 1
        # ----------------------------------------------------------
        self.conv1_out_channels = 8   # e.g., 8 learned feature maps
        self.kernel_size = 3
        self.conv_stride = 1
        self.conv_padding = 0

        conv1_output_height = ((self.input_height - self.kernel_size + 2*self.conv_padding) //
                               self.conv_stride + 1)  # (28 - 3 + 0) +1 = 26
        conv1_output_width  = ((self.input_width - self.kernel_size + 2*self.conv_padding) //
                               self.conv_stride + 1)  # → 26

        print("After Convolution 1 (Conv2d: 1 → 8 filters):")
        print("  Formula: (old_size - kernel + 2*padding) // stride + 1")
        print(f"  Height: ({self.input_height} - {self.kernel_size} + 2*{self.conv_padding}) "
              f"// {self.conv_stride} + 1 = {conv1_output_height}  # e.g., 26")
        print(f"  Width:  ({self.input_width}  - {self.kernel_size} + 2*{self.conv_padding}) "
              f"// {self.conv_stride} + 1 = {conv1_output_width}   # e.g., 26")
        print(f"  Channels: {self.conv1_out_channels}  # e.g., 8\n")

        self.conv1 = nn.Conv2d(
            self.input_channels,
            self.conv1_out_channels,
            kernel_size=self.kernel_size,
            stride=self.conv_stride,
            padding=self.conv_padding
        )

        # ----------------------------------------------------------
        # Max Pool Layer 1
        # ----------------------------------------------------------
        self.pool1_kernel = 2
        self.pool1_stride = 2

        pool1_output_height = conv1_output_height // self.pool1_kernel  # 26 // 2 = 13
        pool1_output_width  = conv1_output_width  // self.pool1_kernel  # → 13

        print("After MaxPooling 1 (2x2):")
        print(f"  Height: {conv1_output_height} // {self.pool1_kernel} = {pool1_output_height}  # e.g., 13")
        print(f"  Width:  {conv1_output_width}  // {self.pool1_kernel} = {pool1_output_width}   # e.g., 13\n")


        # ----------------------------------------------------------
        # Convolution Layer 2
        # ----------------------------------------------------------
        self.conv2_out_channels = 16  # e.g., 16 learned feature maps
        self.kernel2_size = 3
        self.conv2_stride = 1
        self.conv2_padding = 0

        conv2_output_height = ((pool1_output_height - self.kernel2_size + 2*self.conv2_padding) //
                               self.conv2_stride + 1)  # (13 - 3 + 0) +1 = 11
        conv2_output_width  = ((pool1_output_width - self.kernel2_size + 2*self.conv2_padding) //
                               self.conv2_stride + 1)  # → 11

        print("After Convolution 2 (Conv2d: 8 → 16 filters):")
        print(f"  Height: ({pool1_output_height} - {self.kernel2_size}) + 1 = {conv2_output_height}  # e.g., 11")
        print(f"  Width:  ({pool1_output_width}  - {self.kernel2_size}) + 1 = {conv2_output_width}   # e.g., 11")
        print(f"  Channels: {self.conv2_out_channels}  # e.g., 16\n")

        self.conv2 = nn.Conv2d(
            self.conv1_out_channels,
            self.conv2_out_channels,
            kernel_size=self.kernel2_size,
            stride=self.conv2_stride,
            padding=self.conv2_padding
        )


        # ----------------------------------------------------------
        # Max Pool Layer 2
        # ----------------------------------------------------------
        self.pool2_kernel = 2
        self.pool2_stride = 2

        pool2_output_height = conv2_output_height // self.pool2_kernel  # 11 // 2 = 5
        pool2_output_width  = conv2_output_width  // self.pool2_kernel  # → 5

        print("After MaxPooling 2 (2x2):")
        print(f"  Height: {conv2_output_height} // {self.pool2_kernel} = {pool2_output_height}  # e.g., 5")
        print(f"  Width:  {conv2_output_output_width if False else conv2_output_width}  // {self.pool2_kernel} = {pool2_output_width}   # e.g., 5\n")


        # ----------------------------------------------------------
        # Final Shape Before Fully Connected Layer
        # ----------------------------------------------------------
        self.final_channels = self.conv2_out_channels      # e.g., 16
        self.final_height   = pool2_output_height          # e.g., 5
        self.final_width    = pool2_output_width           # e.g., 5

        flattened_size = self.final_channels * self.final_height * self.final_width  # 16 * 5 * 5 = 400

        print("Final feature map before Fully Connected Layer:")
        print(f"  Channels: {self.final_channels}  # e.g., 16")
        print(f"  Height:   {self.final_height}    # e.g., 5")
        print(f"  Width:    {self.final_width}     # e.g., 5")
        print(f"  Flattened size: {self.final_channels} * {self.final_height} * {self.final_width} "
              f"= {flattened_size}  # e.g., 16 * 5 * 5 = 400\n")

        # ----------------------------------------------------------
        # Fully Connected Output Layer (Classifier)
        # ----------------------------------------------------------
        self.fc = nn.Linear(flattened_size, 10)  # 10 digits (0–9)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=self.pool1_kernel, stride=self.pool1_stride)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=self.pool2_kernel, stride=self.pool2_stride)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Build the "house" from the blueprint so we can summarize parameters
temp_model = SimpleCNN()

total_params     = sum(p.numel() for p in temp_model.parameters())
trainable_params = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)

print("=== END OF STEP 1 ===")
print("Model Summary:")
print("  - Conv Blocks:        2")
print("  - Pool Blocks:        2")
print("  - Fully Connected:    1")
print(f"  - Total Parameters:   {total_params:,}")
print(f"  - Trainable Params:   {trainable_params:,}")
print("  - Expected Input:     (batch_size, 1, 28, 28)\n")


# ================================================================
# Step 3:
# "The construction crew builds the actual house and brings tools"
#
# In this step, we CREATE the working model from the blueprint:
#   - Instantiate the model object from the class (the house)
#   - LOSS FUNCTION = the "inspector" scoring mistakes
#   - OPTIMIZER = the "mechanic" adjusting the weights
#
# Step 1 only DESCRIBED what the house would look like.
# Step 3 actually BUILDS it and equips it to improve itself.
# ================================================================

print("\n===================================")
print(" STEP 3: Creating Model + Optimizer")
print("===================================\n")

device = torch.device("cpu")
model = temp_model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ================================================================
# Step 2: Load Dataset
# ================================================================

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False)


# ================================================================
# Step 4: Training Loop
# ================================================================

for epoch in range(1, 4):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()


# ================================================================
# Step 5: Evaluate
# ================================================================

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("\nTest Accuracy: {:.2f}%".format(100 * correct / total))


# ================================================================
# Step 6: Predict One Sample
# ================================================================

sample_img, sample_label = test_dataset[0]
with torch.no_grad():
    sample_batch = sample_img.unsqueeze(0).to(device)
    logits = model(sample_batch)
    pred = torch.argmax(logits)
print(f"\nSingle sample prediction → Predicted: {pred}, Actual: {sample_label}\n")


