import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import CNNClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root=str(Path(__file__).resolve().parent / "data"),
    train = True,
    download = True,
    transform = transform
)

train_loader = DataLoader(
    train_data,
    batch_size = 64,
    shuffle = True
)

test_data = datasets.MNIST(
    root=str(Path(__file__).resolve().parent / "data"),
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False
)

model = CNNClassifier().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()

for epoch in range(5):
    start_time = time.perf_counter()

    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        predictions = model(images)
        loss = loss_fn(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Wait for GPU calculations to finish before measuring
    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time
    average_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch + 1}")
    print(f"Average Loss: {average_loss:.4f}")
    print(f"Time: {elapsed:.2f} seconds")
    print("--------------------")

model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")

model_path = Path(__file__).resolve().parent / "mnist_cnn.pth"
torch.save(model.state_dict(), model_path)

print("Model saved")