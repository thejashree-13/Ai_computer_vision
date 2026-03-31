import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from PIL import Image

# =========================
# STEP 1: DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# STEP 2: DATA PATH
# =========================
data_dir = r"C:\Users\Swetha\Desktop\archive\Dataset"   # 🔥 change this

# =========================
# STEP 3: TRANSFORMS
# =========================
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # smaller = faster
    transforms.ToTensor(),
])

# =========================
# STEP 4: LOAD DATA
# =========================
dataset = datasets.ImageFolder(data_dir, transform=transform)

# Classes
class_names = dataset.classes
print("Classes:", class_names)

# Split into train & test (80% / 20%)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# =========================
# STEP 5: LOAD RESNET
# =========================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Modify final layer
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# =========================
# STEP 6: LOSS + OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========================
# STEP 7: TRAINING
# =========================
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load("waste_classifier.pth"))
model = model.to(device)

model.eval()
print("Model loaded successfully!")
# =========================
# STEP 8: TEST
# =========================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")

# =========================
# STEP 9: SAVE MODEL
# =========================
torch.save(model.state_dict(), "waste_classifier.pth")
print("Model saved!")

# =========================
# STEP 10: PREDICT FUNCTION
# =========================
def predict_image(image_path):
    model.eval()

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    return class_names[predicted.item()]

# Example
print("Prediction:", predict_image(r"C:\Users\Swetha\Desktop\ML\test1.webp"))