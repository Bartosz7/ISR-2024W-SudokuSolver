import torch
from torchvision import transforms
from PIL import Image

def predict_digit(model, image_path, threshold=0.3):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),           
        transforms.Grayscale(num_output_channels=1),  
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])    
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  

    device = next(model.parameters()).device
    image = image.to(device)

    # Inference
    with torch.no_grad():
        outputs = model(image)            
        max_probs, predicted = torch.max(outputs, 1)
        max_prob = max_probs.item()

    if max_prob < threshold:
        return None
    else:
        return predicted.item() + 1 


# model = load_model('/content/digit1.pth')
# digit = predict_digit(model, '/content/9.jpg')
# print(f'Predicted digit: {digit}')
