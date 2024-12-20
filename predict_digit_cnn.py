import torch
from torchvision import transforms
from PIL import Image

def predict_digit(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    device = next(model.parameters()).device
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item() + 1