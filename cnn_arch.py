import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os

class DigitDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for digit in range(1, 10):
            digit_folder = os.path.join(folder_path, str(digit))
            if os.path.isdir(digit_folder):
                for image_name in os.listdir(digit_folder):
                    image_path = os.path.join(digit_folder, image_name)
                    self.image_paths.append(image_path)
                    self.labels.append(digit - 1)  # 0-8

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')  # single-channel
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class SimpleDigitCNN(nn.Module):
    def __init__(self, dropout_p=0.5):
        super(SimpleDigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(dropout_p)

        # For an input size of 64x64 (after transforms):
        # After conv1 -> pool -> conv2 -> pool -> conv3 -> pool
        # Dimensions: 64 -> 32 -> 16 -> 8
        # Flatten: 64 * 8 * 8 = 4096
        self.fc1 = nn.Linear(64*8*8, 128)
        self.fc2 = nn.Linear(128, 9)

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))

        x = x.view(x.size(0), -1)  # flatten
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_model(
    train_folder,
    test_split=0.2,
    num_epochs=10,
    batch_size=16
):
    print("Starting training setup...")

    # Transform includes data augmentation and normalization
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = DigitDataset(train_folder, transform=transform)
    print(f"Found {len(dataset)} images in total.")

    # Split into training and test sets
    train_size = int((1 - test_split) * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # Choose device: GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model and move to the chosen device
    model = SimpleDigitCNN(dropout_p=0.5).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            # Move images & labels to the GPU if available
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 5 == 0:
                avg_loss = train_loss / (batch_idx + 1)
                acc = 100.0 * correct / total
                print(f" Batch {batch_idx+1}/{len(train_loader)} "
                      f"- Loss: {avg_loss:.4f}, "
                      f"Accuracy: {acc:.2f}%")

        scheduler.step()

        # Evaluation
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_acc = 100.0 * test_correct / test_total
        print(f" Test Accuracy: {test_acc:.2f}%")

    return model

def save_model(model, path='digit_model.pth'):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(path='digit_model.pth', device=None):
    # If device is not specified, use "cuda" if available
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleDigitCNN(dropout_p=0.5)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model