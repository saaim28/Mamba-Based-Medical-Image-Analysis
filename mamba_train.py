import torch
import torch.nn as nn
import torch.optim as optim
from dataset_loader import train_loader, val_loader
from test_loader import test_loader
from mamba_model import VisionMambaClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.makedirs("outputs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionMambaClassifier().to(device)

# Weighted Loss (Based on True Class Distribution)
label_list = [label for _, label in train_loader.dataset]
class_counts = torch.bincount(torch.tensor(label_list))
total_samples = class_counts.sum().item()
class_weights = [total_samples / c.item() for c in class_counts]  
weights_tensor = torch.FloatTensor(class_weights).to(device)
loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)

# Optimizer & Epochs
optimizer = optim.Adam(model.parameters(), lr=0.001)
EPOCHS = 15
best_val_acc = 0.0

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if i % 10 == 0:
            print(f"[Epoch {epoch+1} | Batch {i+1}/{len(train_loader)}] Loss: {loss.item():.4f}", flush=True)

    train_acc = 100 * correct / total
    print(f"[Epoch {epoch+1}] Avg Loss: {running_loss:.4f} | Train Accuracy: {train_acc:.2f}%")

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total
    print(f"[Epoch {epoch+1}] Validation Accuracy: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_mamba_model.pth")
        print(f"[Epoch {epoch+1}] 🔥 New best model saved (Val Acc: {val_acc:.2f}%)")

print("\n✅ Training complete.")
print(f"🥇 Best Validation Accuracy: {best_val_acc:.2f}%")

# Test Evaluation
print("\n🧪 Evaluating on Test Set...")
model.load_state_dict(torch.load("best_mamba_model.pth"))
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Results
class_names = ["Normal", "Pneumonia"]
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
