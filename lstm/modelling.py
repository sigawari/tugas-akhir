# file: modelling.py
import os
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Data loading & labeling (tetap seperti sebelumnya)
actions = ['halo', 'terima_kasih']
DATA_PATH = 'MP_Data'
label_map = {label: idx for idx, label in enumerate(actions)}
print("Label map:", label_map)

all_data, all_labels = [], []
for action in actions:
    npy_path = os.path.join(DATA_PATH, f"{action}_combined.npy")
    data = np.load(npy_path)  # shape: (N_seq, 30, 1662)
    print(f"Loaded data shape for {action}: {data.shape}")
    labels = np.full(data.shape, label_map[action], dtype=int)
    all_data.append(data)
    all_labels.append(labels)

X = np.concatenate(all_data, axis=0)  # (N, 30, 1662)
Y = np.concatenate(all_labels, axis=0)  # (N,)
print("Total data shape:", X.shape)
print("Total label shape:", Y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.05, random_state=42, stratify=Y
)
print("Train data shape:", x_train.shape)
print("Train label shape:", y_train.shape)
print("Test data shape:", x_test.shape)
print("Test label shape:", y_test.shape)

class GestureDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)  # (N, 30, 1662)
        self.labels = torch.tensor(labels, dtype=torch.long)           # (N,)
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

train_dataset = GestureDataset(x_train, y_train)
test_dataset  = GestureDataset(x_test, y_test)

# Model LSTM (PyTorch)
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=1662, hidden1=64, hidden2=128, hidden3=64, num_classes=2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=hidden1, batch_first=True)
        self.act1  = nn.ReLU()
        self.lstm2 = nn.LSTM(input_size=hidden1, hidden_size=hidden2, batch_first=True)
        self.act2  = nn.ReLU()
        self.lstm3 = nn.LSTM(input_size=hidden2, hidden_size=hidden3, batch_first=True)
        self.act3  = nn.ReLU()
        self.fc1   = nn.Linear(hidden3, 64)
        self.fc2   = nn.Linear(64, 32)
        self.out   = nn.Linear(32, num_classes)
        # Catatan: tidak pakai softmax di sini, karena CrossEntropyLoss akan mengaplikasikan log-softmax internalnya.

    def forward(self, x):
        # x: (batch, 30, 1662)
        x, _ = self.lstm1(x)  # (batch, seq, hidden1)
        x = self.act1(x)
        x, _ = self.lstm2(x)  # (batch, seq, hidden2)
        x = self.act2(x)
        x, _ = self.lstm3(x)  # (batch, seq, hidden3)
        x = self.act3(x)
        # Ambil last timestep (many-to-one)
        x = x[:, -1, :]       # (batch, hidden3)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.out(x)  # (batch, num_classes)
        return logits

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMClassifier(input_dim=1662, num_classes=len(actions)).to(device)

criterion = nn.CrossEntropyLoss()  # target y: int64 class indices
optimizer = optim.Adam(model.parameters(), lr=1e-3)

batch_size = 8
num_epochs = 50  # mulai realistis; 2000 epochs terlalu besar untuk dataset kecil

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, drop_last=False)

# Train loop
def evaluate(loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss_sum += loss.item() * yb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return loss_sum / total, correct / total

best_acc = 0.0
for epoch in range(1, num_epochs + 1):
    model.train()
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

    train_loss, train_acc = evaluate(train_loader)
    test_loss, test_acc = evaluate(test_loader)
    print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} acc={train_acc:.3f} | "
          f"test_loss={test_loss:.4f} acc={test_acc:.3f}")

# Opsional: simpan model
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/lstm_classifier.pt")
print("Saved to checkpoints/lstm_classifier.pt")
