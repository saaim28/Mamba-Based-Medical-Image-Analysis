from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

test_dir = "chest_xray/test"
val_dir = "chest_xray/val"

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# Datasets
test_dataset = datasets.ImageFolder(os.path.join(test_dir), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(val_dir), transform=transform)

# Loaders
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
