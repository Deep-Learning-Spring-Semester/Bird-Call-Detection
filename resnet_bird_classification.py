import os
import zipfile
import re
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GroupShuffleSplit

# --- CONFIGURATION ---
base_dir = "." 
zip_path = os.path.join(base_dir, 'data.zip')
# Detect if running on Windows or Linux/Docker to choose the best temp path
if os.name == 'nt': 
    # Windows (Standard folder in current directory)
    extract_path = os.path.join(base_dir, 'temp_bird_data')
else:
    # Linux/Mac/Docker (Fast RAM/SSD storage)
    extract_path = '/tmp/bird_data'
target_subfolder = 'data/processed/spectrograms/augmented'
save_model_path = os.path.join(base_dir, 'resnet_bird_baseline.pth')

# --- 1. DATA EXTRACTION ---
if not os.path.exists(zip_path):
    raise FileNotFoundError(f"Could not find data.zip at {zip_path}")

full_extract_path = os.path.join(extract_path, target_subfolder)

if not os.path.exists(full_extract_path):
    print(f"Extracting data from {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as archive:
        all_files = archive.namelist()
        images_to_extract = [f for f in all_files if f.startswith(target_subfolder) and (f.endswith('.png') or f.endswith('.jpg'))]
        archive.extractall(path=extract_path, members=images_to_extract)
    print("Extraction complete.")
else:
    print("Data directory already exists. Skipping extraction.")

# --- 2. DATA LOADERS (REGEX GROUPING) ---
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print(f"Loading data from: {full_extract_path}")
full_dataset = datasets.ImageFolder(root=full_extract_path, transform=data_transforms)

print("Grouping by XC Number to prevent leakage...")

# --- FIND 'XC' NUMBER ---
samples = full_dataset.samples
group_ids = []
xc_pattern = re.compile(r'(XC\d+)') # Looks for "XC" followed by digits

debug_count = 0
for path, _ in samples:
    filename = os.path.basename(path)
    match = xc_pattern.search(filename)
    
    if match:
        group_id = match.group(1) # e.g., "XC12345"
    else:
        # Fallback: If no XC number, use the whole filename (shouldn't happen often)
        group_id = filename
    
    group_ids.append(group_id)
    
    # Debug: Show the first 5 mappings to verify
    if debug_count < 5:
        print(f"   Debug: File '{filename}' -> Group '{group_id}'")
        debug_count += 1

group_ids = np.array(group_ids)
indices = np.arange(len(full_dataset))
labels = np.array([s[1] for s in samples])

print(f"Identified {len(np.unique(group_ids))} unique recordings (XC Numbers).")

# Split 1: Train vs Temp
gss = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
train_idx, temp_idx = next(gss.split(indices, labels, group_ids))

# Split 2: Val vs Test
temp_groups = group_ids[temp_idx]
temp_labels = labels[temp_idx]
temp_indices_global = indices[temp_idx] 

gss_val = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_relative_idx, test_relative_idx = next(gss_val.split(temp_indices_global, temp_labels, temp_groups))

val_idx = temp_indices_global[val_relative_idx]
test_idx = temp_indices_global[test_relative_idx]

train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)
test_dataset = Subset(full_dataset, test_idx)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

# Dataloaders
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# --- 3. MODEL SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_resnet_baseline(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

model = get_resnet_baseline(len(full_dataset.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 4. TRAINING LOOP ---
num_epochs = 5
best_val_acc = 0.0

print("Starting training...")

for epoch in range(num_epochs):
    start_time = time.time()
    
    # Training
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
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_acc = 100 * correct_train / total_train

    # Validation
    model.eval()
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            
    val_acc = 100 * correct_val / total_val
    epoch_time = time.time() - start_time
    
    # Save Best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_model_path)
        saved_msg = f"-> Saved! ({best_val_acc:.2f}%)"
    else:
        saved_msg = ""

    print(f"Epoch {epoch+1}/{num_epochs} | Time: {epoch_time:.0f}s | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}% {saved_msg}")

print(f"Training Complete. Best model saved to: {save_model_path}")

# --- 5. EVALUATION ---
print("Evaluating on Test Set...")
model.load_state_dict(torch.load(save_model_path))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Classification Report
print("\n--- Classification Report ---")
print(classification_report(all_labels, all_preds, target_names=full_dataset.classes))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=full_dataset.classes, yticklabels=full_dataset.classes)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png') 
print("Confusion Matrix saved as 'confusion_matrix.png'")