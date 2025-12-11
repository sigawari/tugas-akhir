# eval.py
# Evaluasi model di test set + confusion matrix.

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from utils import (
    DATA_DIR,
    PROCESSED_DIR,
    DEFAULT_SEED,
    seed,
    get_device,
    read_json,
)
from models import ResNet2DSign
from metrics import accuracy_from_logits, confusion_matrix_from_preds


# --- Dataset yang sama seperti di train.py (boleh disamain persis) ---

class SignDataset(Dataset):
    def __init__(self, items: List[Dict[str, Any]], variant: str) -> None:
        super().__init__()
        self.items = items
        self.variant = variant
        self._cache_X: Dict[str, np.ndarray] = {}

    def _load_X_for_word(self, word: str) -> np.ndarray:
        if word in self._cache_X:
            return self._cache_X[word]
        x_path = PROCESSED_DIR / word / self.variant / "X.npy"
        X = np.load(x_path)  # (N, T, D)
        self._cache_X[word] = X
        return X

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        word = item["word"]
        seq_idx = item["index"]
        label = item["label"]

        X_word = self._load_X_for_word(word)
        seq = X_word[seq_idx]   # (T, D)

        T, D = seq.shape
        assert D % 4 == 0
        L = D // 4

        seq = seq.reshape(T, L, 4)        # (T, L, 4)
        seq = np.transpose(seq, (2, 0, 1))  # (4, T, L)

        x = torch.from_numpy(seq).float()
        y = torch.tensor(label, dtype=torch.long)
        return {"x": x, "y": y, "word": word, "index": seq_idx}


@torch.no_grad()
def evaluate_test(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_count = 0

    criterion = torch.nn.CrossEntropyLoss()
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    for batch in loader:
        x = batch["x"].to(device)    # (B, 4, T, L)
        y = batch["y"].to(device)    # (B,)

        logits = model(x)
        loss = criterion(logits, y)

        batch_size = y.size(0)
        acc = accuracy_from_logits(logits, y)

        total_loss += loss.item() * batch_size
        total_acc += acc * batch_size
        total_count += batch_size

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())

    avg_loss = total_loss / total_count
    avg_acc = total_acc / total_count

    all_preds_t = torch.cat(all_preds, dim=0)
    all_targets_t = torch.cat(all_targets, dim=0)
    cm = confusion_matrix_from_preds(all_preds_t, all_targets_t, num_classes)

    return avg_loss, avg_acc, cm, all_targets_t

def plot_confusion_matrix(cm: np.ndarray, idx2label: dict, save_path: str = None):
    """
    Plot confusion matrix sederhana.
    cm: numpy array shape (num_classes, num_classes)
    idx2label: dict {label_index: label_name}
    """

    num_classes = cm.shape[0]
    labels = [idx2label[i] for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix (Test Set)"
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Tulis angka dalam kotak
    thresh = cm.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Confusion matrix saved to {save_path}")

    plt.show()

def main():
    seed(DEFAULT_SEED)
    device = get_device(prefer_gpu=True)
    print(f"Device: {device}")

    # --- Load split & label mapping ---
    split_path = DATA_DIR / "splits" / "split.json"
    split_data = read_json(split_path)

    label2idx = split_data["label2idx"]
    idx2label = {int(k): v for k, v in split_data["idx2label"].items()}
    num_classes = len(label2idx)

    test_items = split_data["splits"]["test"]
    print(f"Total test samples: {len(test_items)}")

    # --- Dataset & loader untuk TEST ---
    variant = "pose"  # lagi eval model pose
    test_ds = SignDataset(test_items, variant=variant)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    # --- Load checkpoint terbaik ---
    ckpt_path = Path("checkpoints") / "resnet18_pose_best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model = ResNet2DSign(num_classes=num_classes, in_channels=4)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    # --- Evaluasi ---
    test_loss, test_acc, cm, y_true = evaluate_test(
        model, test_loader, device, num_classes
    )

    print(f"\n=== TEST RESULT (variant={variant}) ===")
    print(f"Test loss : {test_loss:.4f}")
    print(f"Test acc  : {test_acc:.4f}")

    cm_np = cm.numpy()

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm.numpy())

    save_path = f"confusion_matrix_{variant}.png"
    plot_confusion_matrix(cm_np, idx2label, save_path=save_path)

    print("\nLabel index mapping:")
    for idx, name in idx2label.items():
        print(f"  {idx}: {name}")


if __name__ == "__main__":
    main()
