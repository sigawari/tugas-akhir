# metrics.py
# Compute evaluation metrics:
# accuracy, f1-score, confusion matrix, classification report.

from typing import Tuple
import torch


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Hitung akurasi dari logits model dan label integer."""
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0


def topk_accuracy_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 3
) -> float:
    """Optional: top-k accuracy (misal top-3)."""
    _, topk = torch.topk(logits, k, dim=1)
    # topk: (B, k)
    correct = (topk == targets.view(-1, 1)).any(dim=1).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0

def confusion_matrix_from_preds(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Bikin confusion matrix sederhana (num_classes x num_classes).

    Baris = label asli (true)
    Kolom = prediksi model (pred)
    """
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(targets.view(-1), preds.view(-1)):
        t = int(t)
        p = int(p)
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm
