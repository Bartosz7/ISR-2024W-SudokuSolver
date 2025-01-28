import torch
import torch.nn as nn


class EnhancedDigitCNN(torch.nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(0.3),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(0.4),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(0.5),
            torch.nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(256 * 4 * 4, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.6),
            torch.nn.Linear(512, 9),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class CNNClassifier(nn.Module):
    """Blank cell recognizer"""

    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(3136, 128),  # Adjust this to match the trained model
            nn.ReLU(),
            nn.Linear(128, 2),  # Binary output: Blank or not
        )

    def forward(self, x):
        return self.model(x)
