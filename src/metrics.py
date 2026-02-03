# metrics.py
# Compute evaluation metrics:
# accuracy, f1-score, confusion matrix, classification report.

from typing import Tuple, Optional
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
    """Bikin confusion matrix (num_classes x num_classes).

    Baris = label asli (true)
    Kolom = prediksi model (pred)
    """
    preds = preds.view(-1).to(torch.int64)
    targets = targets.view(-1).to(torch.int64)

    # filter out-of-range
    mask = (targets >= 0) & (targets < num_classes) & (preds >= 0) & (preds < num_classes)
    preds = preds[mask]
    targets = targets[mask]

    # bincount on flattened indices
    flat = targets * num_classes + preds
    cm = torch.bincount(flat, minlength=num_classes * num_classes).view(num_classes, num_classes)
    return cm.to(torch.int64)


def f1_from_confusion(cm: torch.Tensor) -> Tuple[float, float]:
    """Return (f1_macro, f1_weighted) dari confusion matrix."""
    cm = cm.to(torch.float32)
    tp = torch.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    support = cm.sum(1)

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    f1_macro = f1.mean().item()
    f1_weighted = (f1 * support / (support.sum() + 1e-12)).sum().item()
    return f1_macro, f1_weighted


def balanced_accuracy_from_confusion(cm: torch.Tensor) -> float:
    """Balanced accuracy = rata-rata recall per kelas."""
    cm = cm.to(torch.float32)
    tp = torch.diag(cm)
    fn = cm.sum(1) - tp
    recall = tp / (tp + fn + 1e-12)
    return recall.mean().item()
