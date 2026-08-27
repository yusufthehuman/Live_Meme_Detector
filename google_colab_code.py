"""
the following code should be inserted into google colab and NOT in vscode or pycharm so that u can get your model trained properly.
Run each number as a separate cell and not all in one go.
"""

#1st cell

from google.colab import files

print("Click 'Choose Files' below and select your dataset.zip file from your PC:")
uploaded = files.upload()

if "dataset.zip" in uploaded:
    print("\nSUCCESS: dataset.zip uploaded successfully!")
else:
    print("\nWarning: The uploaded file was not named 'dataset.zip'. Please rename it.")
#select the dataset.zip file that u generated from your data and wait for the download to finish

#2nd cell

import os

if os.path.exists("dataset.zip"):
    print("SUCCESS: dataset.zip found! You can now run the extraction cell.")
else:
    print("STILL MISSING: Current files in Colab directory:", os.listdir())
#this is to check if you have file downloaded and ready to use and move on to the next cell

#3rd cell

import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# -------------------------------------------------------------
# Extract Dataset
# -------------------------------------------------------------
zip_path = "dataset.zip"
extract_path = "dataset"

if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Dataset extracted successfully!")
else:
    raise FileNotFoundError("Please upload 'dataset.zip' to the Colab files sidebar first.")

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {device}")

#4th cell

# PyTorch MobileNetV2 requires 224x224 input and ImageNet normalization
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#5th cell

# Load complete dataset from folder
full_dataset = datasets.ImageFolder(root=extract_path, transform=data_transforms)
class_names = full_dataset.classes
num_classes = len(class_names)
print(f"Classes found ({num_classes}): {class_names}")

#6th cell

# Split dataset: 80% Training, 20% Validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

#7th cell

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

#8th cell

for param in model.parameters():
    param.requires_grad = False

#9th cell

# Extract the number of classes automatically from your dataset folders
num_classes = len(full_dataset.classes)  # Evaluates to 5 if you have 5 folders

# Replace the 2nd layer (index 1) of the classifier block
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

#10th cell

# -------------------------------------------------------------
# Training & Validation Loop
# -------------------------------------------------------------
num_epochs = 10
print("Starting Training...\n")

for epoch in range(num_epochs):
    # --- TRAINING PHASE ---
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += torch.sum(preds == labels.data)

    epoch_train_loss = running_loss / train_size
    epoch_train_acc = (correct_train.double() / train_size) * 100

    # --- VALIDATION PHASE ---
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += torch.sum(preds == labels.data)

    epoch_val_loss = val_loss / val_size
    epoch_val_acc = (correct_val.double() / val_size) * 100

    print(f"Epoch [{epoch+1:02d}/{num_epochs:02d}] | "
          f"Train Acc: {epoch_train_acc:.2f}% (Loss: {epoch_train_loss:.4f}) | "
          f"Val Acc: {epoch_val_acc:.2f}% (Loss: {epoch_val_loss:.4f})")

#make sure the loss is going down and u have an accuracy above 85%

#11th cell

#saving the model
save_dict = {
    'model_state_dict': model.state_dict(),
    'class_names': class_names
}

model_save_path = "emotion_model_pytorch.pth"
torch.save(save_dict, model_save_path)
print(f"\nModel successfully trained and saved to '{model_save_path}'!")

#12th cell

#make sure to save the model in the same directory as the rest of the code
from google.colab import files
import torch

# 1. Re-save the model file safely
save_dict = {
    'model_state_dict': model.state_dict(),
    'class_names': class_names
}
model_save_path = "emotion_model_pytorch.pth"
torch.save(save_dict, model_save_path)

print(f"Model re-saved successfully to '{model_save_path}'. Starting download...")

# 2. Trigger automatic browser download to your PC
files.download(model_save_path)