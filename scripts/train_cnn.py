import numpy as np
import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import time
import copy  # Needed to save the best model state

# --- Configuration ---
FEATURES_FOLDER = 'C:\\Users\\pinti\\Desktop\\proiect_TIA\\data\\features'
MODEL_SAVE_PATH = 'C:\\Users\\pinti\\Desktop\\proiect_TIA\\models\\audio_cnn_vehicle_classifier_early_stop.pth'
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
NUM_CLASSES = 8
EPOCHS = 50       # Increased max epochs; Early Stopping will halt it sooner
PATIENCE = 8      # How many epochs to wait for improvement before stopping
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 1. Custom Dataset Class ---
class SpectrogramDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.features = self.features.permute(0, 3, 1, 2)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# --- 2. Data Loading and Splitting ---
def load_data(folder_path):
    all_features, all_labels = [], []
    filename_pattern = re.compile(r'^(\d+)\s')
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            match = filename_pattern.match(filename)
            if match:
                class_label = int(match.group(1)) - 1 
                if 0 <= class_label < NUM_CLASSES:
                    feature = np.load(os.path.join(folder_path, filename))
                    all_features.append(feature)
                    all_labels.append(class_label)
    features = np.array(all_features)
    if features.ndim == 3: features = np.expand_dims(features, axis=-1)
    return (features - features.min()) / (features.max() - features.min()), np.array(all_labels)

X_data, y_data = load_data(FEATURES_FOLDER)
X_train_val, X_test, y_train_val, y_test = train_test_split(X_data, y_data, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_data)
val_rel = VALIDATION_SIZE / (1.0 - TEST_SIZE)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_rel, random_state=RANDOM_STATE, stratify=y_train_val)

train_loader = DataLoader(SpectrogramDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(SpectrogramDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(SpectrogramDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

# --- 3. Model Architecture ---
class AudioCNN(nn.Module):
    def __init__(self, input_channels, num_classes, input_h, input_w):
        super(AudioCNN, self).__init__()
        self.cnn_stack = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2), nn.Dropout(0.35),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2), nn.Dropout(0.35),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2), nn.Dropout(0.35)
        )
        with torch.no_grad():
            feature_size = self.cnn_stack(torch.zeros(1, input_channels, input_h, input_w)).flatten().shape[0]
        self.fc_stack = nn.Sequential(nn.Linear(feature_size, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.6), nn.Linear(256, num_classes))
    def forward(self, x):
        return self.fc_stack(torch.flatten(self.cnn_stack(x), 1))

# --- 4. Early Stopping Implementation ---
class EarlyStopping:
    def __init__(self, patience=5, path=MODEL_SAVE_PATH):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.path = path
        self.best_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_state = copy.deepcopy(model.state_dict()) # Save best weights
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# --- 5. Training Loop (REVISED with Accuracy Printing) ---
model = AudioCNN(1, NUM_CLASSES, X_data.shape[1], X_data.shape[2]).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
early_stopping = EarlyStopping(patience=PATIENCE)

print("\nStarting Training with Early Stopping...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    avg_train_loss = train_loss / train_total
    avg_train_acc = train_correct / train_total

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    avg_val_loss = val_loss / val_total
    avg_val_acc = val_correct / val_total
    
    # Updated Print Statement to give you all the info for Excel
    print(f"Epoch {epoch}: "
          f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc*100:.2f}% | "
          f"Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc*100:.2f}%")

    early_stopping(avg_val_loss, model) # Check for improvement
    if early_stopping.early_stop:
        print(f"Early stopping triggered. Restoring best weights from epoch {epoch - early_stopping.counter}.")
        model.load_state_dict(early_stopping.best_state)
        break   

# Save and Evaluate
torch.save(early_stopping.best_state, MODEL_SAVE_PATH)
model.eval()
all_preds, all_labs = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        _, predicted = torch.max(model(inputs.to(DEVICE)), 1)
        all_preds.extend(predicted.cpu().numpy()); all_labs.extend(labels.numpy())

print("\n### Classification Report ###")
print(classification_report(all_labs, all_preds, target_names=['Bus', 'Minibus', 'Pickup', 'Sports', 'Jeep', 'Truck', 'Crossover', 'Car']))