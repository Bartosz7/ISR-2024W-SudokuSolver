import torch
import torch.nn as nn
import torch.optim as optim
import io
import os
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import random
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class DigitDataset(Dataset):
    def __init__(self, folder_path=None, mnist_df=None, transform=None, mnist_ratio=0.2):
        self.transform = transform
        self.samples = []

        # Load custom dataset (1-9)
        if folder_path:
            if not os.path.exists(folder_path):
                raise ValueError(f"Custom dataset path {folder_path} does not exist")
                
            for digit in range(1, 10):
                digit_folder = os.path.join(folder_path, str(digit))
                if os.path.isdir(digit_folder):
                    images = os.listdir(digit_folder)
                    if not images:
                        raise ValueError(f"No images found in {digit_folder}")
                    self.samples.extend([
                        (os.path.join(digit_folder, img), digit)
                        for img in images
                        if img.lower().endswith(('.png', '.jpg', '.jpeg'))
                    ])

        # Load MNIST data (1-9)
        if mnist_df is not None:
            if len(mnist_df) == 0:
                raise ValueError("MNIST DataFrame is empty")
                
            # Filter out MNIST zeros
            mnist_df = mnist_df[mnist_df['label'] != 0]
            if len(mnist_df) == 0:
                raise ValueError("No non-zero MNIST samples available")
                
            # Calculate MNIST sample size
            total_samples = len(self.samples)
            mnist_samples = int((mnist_ratio * total_samples) / (1 - mnist_ratio))
            mnist_samples = min(mnist_samples, len(mnist_df))
            
            mnist_subset = mnist_df.sample(n=mnist_samples, random_state=42)
            self.samples.extend([
                (row['image']['bytes'], row['label'])
                for _, row in mnist_subset.iterrows()
            ])

        # Convert labels to 0-8
        self.samples = [(data, label-1) for data, label in self.samples]
        
        if len(self.samples) == 0:
            raise ValueError("No valid samples found after processing")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data, label = self.samples[idx]
        
        if isinstance(data, str):  # Custom dataset
            image = Image.open(data).convert('L')
        else:  # MNIST bytes
            image = Image.open(io.BytesIO(data)).convert('L')

        if self.transform:
            image = self.transform(image)

        return image, label

class EnhancedDigitCNN(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.4),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256*4*4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 9)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.1, num_classes=9):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, preds, targets):
        probs = torch.nn.functional.log_softmax(preds, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * probs, dim=-1))

def train_model(
    train_folder=None,
    mnist_df=None,
    num_epochs=20,
    batch_size=64,
    l2_lambda=0.01,
    smoothing=0.2,
    mnist_ratio=0.2
):
    # Data augmentations with safe rotation range
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomRotation(30),  # ±30° max rotation
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.GaussianBlur(3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    try:
        dataset = DigitDataset(
            folder_path=train_folder,
            mnist_df=mnist_df,
            transform=transform,
            mnist_ratio=mnist_ratio
        )
    except ValueError as e:
        print(f"Dataset creation failed: {str(e)}")
        return None

    # Stratified train-validation split
    labels = [label for (_, label) in dataset.samples]
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=42
    )

    # Class balancing
    class_counts = torch.zeros(9)
    for idx in train_indices:
        _, label = dataset[idx]
        class_counts[label] += 1
        
    weights = 1.0 / (class_counts + 1e-6)
    samples_weights = weights[[dataset[idx][1] for idx in train_indices]]

    sampler = WeightedRandomSampler(
        weights=samples_weights,
        num_samples=len(train_indices),
        replacement=True
    )

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=batch_size*2,
        num_workers=2,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedDigitCNN(dropout_p=0.6).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=l2_lambda)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = LabelSmoothingCE(smoothing=smoothing)

    best_acc = 0.0
    patience_counter = 0
    best_model = None
    
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        scheduler.step()

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
        
        val_acc = 100 * val_correct / len(val_indices)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model.state_dict()
            patience_counter = 0
            torch.save(best_model, 'best_model_0_3_with_augmented.pth')
            print(f"Epoch {epoch+1}: New best val accuracy {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}: Val accuracy {val_acc:.2f}%")

        if patience_counter >= 5:
            print("Early stopping triggered")
            break

    model.load_state_dict(best_model)
    return model

if __name__ == "__main__":
    splits = {'train': 'mnist/train-00000-of-00001.parquet', 'test': 'mnist/test-00000-of-00001.parquet'}
    mnsit = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["train"])
    mnist = mnsit[mnsit['label'] != 0]
    mnist = mnist.reset_index(drop=True)

    path_to_train_dataset = ""
    model = train_model(
        train_folder=path_to_train_dataset,
        mnist_df=mnist,
        num_epochs=30,
        batch_size=128,
        l2_lambda=0.01,
        smoothing=0.2,
        mnist_ratio=0.4
    )
    
    if model is not None:
        torch.save(model.state_dict(), 'final_model_0_3_with_augmented.pth')
        print("Training completed successfully!")