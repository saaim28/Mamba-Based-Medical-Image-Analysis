import os
from gradcam import generate_gradcam
from mamba_model import VisionMambaClassifier
import torch
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VisionMambaClassifier().to(device)
model.load_state_dict(torch.load("best_mamba_model.pth", map_location=device))
model.eval()

# Target layer = last layer of the model before classification
target_layer = model.mamba

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Grayscale
])

example_paths = [
    "chest_xray/test/NORMAL/IM-0105-0001.jpeg",
    "chest_xray/test/PNEUMONIA/person86_bacteria_429.jpeg",
]

os.makedirs("outputs/gradcam", exist_ok=True)

for path in example_paths:
    image = Image.open(path).convert("L")  # Convert to 1 channel
    image_tensor = transform(image).to(device)
    print(f"Image tensor shape: {image_tensor.shape}")  # Should be [1, 64, 64]

    output = model(image_tensor.unsqueeze(0))  # -> [1, 1, 64, 64]
    pred = torch.argmax(output, dim=1).item()

    class_name = "Normal" if pred == 0 else "Pneumonia"
    filename = os.path.basename(path).replace(".jpeg", f"_gradcam_{class_name}.png")
    save_path = os.path.join("outputs/gradcam", filename)

    generate_gradcam(model, image_tensor.unsqueeze(0), pred, target_layer, save_path)
    print(f"Saved Grad-CAM to: {save_path}")
