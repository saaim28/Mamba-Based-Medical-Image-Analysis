
from dataset_loader import train_loader

if __name__ == '__main__':
    for images, labels in train_loader:
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")
        break
