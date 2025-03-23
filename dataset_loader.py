import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import random

# Download latest version
DATASET_PATH = "chest_xray"

# Define transformations
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 for Vision Mamba compatibility
    transforms.RandomHorizontalFlip(p=0.5),  # Augmentation: Random flip
    transforms.RandomRotation(10),  # Small rotation
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Custom dataset class
class PneumoniaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label, class_name in enumerate(["NORMAL", "PNEUMONIA"]):
            class_path = os.path.join(root_dir, class_name)
            image_files = os.listdir(class_path)
            self.image_paths.extend([os.path.join(class_path, f) for f in image_files])
            self.labels.extend([label] * len(image_files))

        # Shuffle dataset
        combined = list(zip(self.image_paths, self.labels))
        random.shuffle(combined)
        self.image_paths, self.labels = zip(*combined)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert("L") 
        
        if self.transform:
            image = self.transform(image)

        return image, label

# Load datasets
train_dataset = PneumoniaDataset(os.path.join(DATASET_PATH, "train"), transform=train_transforms)
val_dataset = PneumoniaDataset(os.path.join(DATASET_PATH, "val"), transform=test_transforms)
test_dataset = PneumoniaDataset(os.path.join(DATASET_PATH, "test"), transform=test_transforms)

# Create dataloaders
BATCH_SIZE = 32

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Check dataset size
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
