import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from src.model import EnhancedDigitCNN
from src.model import CNNClassifier

# Allowlist the CNNClassifier class
torch.serialization.add_safe_globals([CNNClassifier])


def load_model(model_path, device="auto"):
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = EnhancedDigitCNN(dropout_p=0.6)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model, device


def load_blank_classifier(model_path, device="auto"):
    """
    Loads the CNNClassifier model for blank cell recognition.

    Args:
        model_path (str): Path to the saved model's state_dict file.
        device (str): Device to load the model onto ('auto', 'cpu', or 'cuda').

    Returns:
        tuple: The loaded CNNClassifier model and the device it's loaded onto.
    """
    # Determine the device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Load the model without weights_only=True
    model = torch.load(model_path, map_location=device)

    # Move the model to the specified device
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode

    return model, device


def predict_digit(model, device, image_path=None, image=None, topk=3):
    """Make prediction on an input image
    Args:
        model: Trained model
        device: Device where model is located
        image_path (str): Path to image file
        image: PIL Image object (alternative to image_path)
        topk (int): Number of top predictions to return
    Returns:
        tuple: (probabilities, digits) where digits are 1-9
    """

    # Load image if path is provided
    if image_path is not None:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
        image = Image.open(image_path).convert("L")

    # Define preprocessing pipeline
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    image_tensor = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

    probs = probabilities.cpu().numpy()[0]
    classes = np.arange(1, 10)  # Digits 1-9 (original labels)

    # Get the single best prediction
    max_index = probs.argmax()
    best_digit = classes[max_index]
    best_prob = probs[max_index]

    return best_prob, best_digit
