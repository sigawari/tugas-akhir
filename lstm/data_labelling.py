import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

actions = ['halo', 'terima_kasih']
DATA_PATH = 'MP_Data'

label_map = {label: idx for idx, label in enumerate(actions)}
print("Label map:", label_map)

all_data = []
all_labels = []

for action in actions:
    npy_path = os.path.join(DATA_PATH, f"{action}_combined.npy")
    data = np.load(npy_path)
    print(f"Loaded data shape for {action}: {data.shape}")  # ex: (30, 30, fitur)
    labels = np.full(data.shape[0], label_map[action], dtype=int)
    
    all_data.append(data)
    all_labels.append(labels)

X = np.concatenate(all_data, axis=0)
Y = np.concatenate(all_labels, axis=0)

print("Total data shape:", X.shape)
print("Total label shape:", Y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.05, random_state=42, stratify=Y)

print("Train data shape:", x_train.shape)
print("Train label shape:", y_train.shape)
print("Test data shape:", x_test.shape)
print("Test label shape:", y_test.shape)

class GestureDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

train_dataset = GestureDataset(x_train, y_train)
test_dataset = GestureDataset(x_test, y_test)
