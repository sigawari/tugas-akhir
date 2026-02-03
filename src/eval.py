# eval.py
# Evaluasi model di test set + confusion matrix + metrik lengkap (F1, precision/recall, per-class).
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    top_k_accuracy_score,
)

from utils import (
    DATA_DIR,
    PROCESSED_DIR,
    DEFAULT_SEED,
    seed,
    get_device,
    read_json,
)

from models import Basic2DCNN, ResNet18, ResNet34, ResNet50


# -------------------------
# Dataset (samain dengan train.py)
# -------------------------
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
        if not x_path.is_file():
            raise FileNotFoundError(f"X.npy tidak ditemukan: {x_path}")
        X = np.load(x_path, mmap_mode="r")  # (N, T, D)
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
        if D % 4 != 0:
            raise ValueError(f"D={D} tidak habis dibagi 4. Pastikan fitur per landmark = 4.")
        L = D // 4

        seq = seq.reshape(T, L, 4)          # (T, L, 4)
        seq = np.transpose(seq, (2, 0, 1))  # (4, T, L)

        x = torch.from_numpy(seq).float()
        y = torch.tensor(label, dtype=torch.long)
        return {"x": x, "y": y, "word": word, "index": seq_idx}


# -------------------------
# Helper: plot confusion matrix
# -------------------------
def plot_confusion_matrix(
    cm: np.ndarray,
    idx2label: Dict[int, str],
    save_path: Optional[str] = None,
    normalize: Optional[str] = None,  # None | "true" | "pred" | "all"
) -> None:
    """
    normalize:
      - None: raw counts
      - "true": normalize per baris (true label) -> jadi recall per class
      - "pred": normalize per kolom (pred label)
      - "all": normalize total
    """
    labels = [idx2label[i] for i in range(len(idx2label))]

    cm_to_plot = cm.astype(np.float32)
    title = "Confusion Matrix (Test)"

    if normalize is not None:
        eps = 1e-12
        if normalize == "true":
            cm_to_plot = cm_to_plot / (cm_to_plot.sum(axis=1, keepdims=True) + eps)
            title = "Confusion Matrix (Normalized by True)"
        elif normalize == "pred":
            cm_to_plot = cm_to_plot / (cm_to_plot.sum(axis=0, keepdims=True) + eps)
            title = "Confusion Matrix (Normalized by Pred)"
        elif normalize == "all":
            cm_to_plot = cm_to_plot / (cm_to_plot.sum() + eps)
            title = "Confusion Matrix (Normalized)"
        else:
            raise ValueError("normalize harus None/'true'/'pred'/'all'")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_to_plot, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True Label",
        xlabel="Predicted Label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # annotate numbers
    thresh = cm_to_plot.max() / 2.0 if cm_to_plot.size else 0.5
    for i in range(cm_to_plot.shape[0]):
        for j in range(cm_to_plot.shape[1]):
            if normalize is None:
                text = f"{int(cm_to_plot[i, j])}"
            else:
                text = f"{cm_to_plot[i, j]:.2f}"
            ax.text(
                j, i, text,
                ha="center",
                va="center",
                color="white" if cm_to_plot[i, j] > thresh else "black",
                fontsize=10,
            )

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"✅ Saved: {save_path}")
    plt.show()


# -------------------------
# Evaluasi test: collect logits + preds
# -------------------------
@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
      avg_loss,
      y_true (N,),
      y_pred (N,),
      y_prob (N, C) softmax probabilities
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    total_count = 0

    ys: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    probs: List[np.ndarray] = []

    for batch in loader:
        x = batch["x"].to(device)  # (B, 4, T, L)
        y = batch["y"].to(device)  # (B,)

        logits = model(x)
        loss = criterion(logits, y)

        bsz = y.size(0)
        total_loss += loss.item() * bsz
        total_count += bsz

        p = torch.softmax(logits, dim=1)

        y_np = y.detach().cpu().numpy()
        pred_np = torch.argmax(logits, dim=1).detach().cpu().numpy()
        prob_np = p.detach().cpu().numpy()

        ys.append(y_np)
        preds.append(pred_np)
        probs.append(prob_np)

    avg_loss = total_loss / max(total_count, 1)
    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(preds, axis=0)
    y_prob = np.concatenate(probs, axis=0)

    return avg_loss, y_true, y_pred, y_prob


def build_model(model_name: str, num_classes: int, in_channels: int = 4) -> torch.nn.Module:
    model_name = model_name.lower()
    if model_name in ("cnn2d", "basic2dcnn", "cnn"):
        return Basic2DCNN(num_classes=num_classes, in_channels=in_channels)
    if model_name in ("resnet18", "r18"):
        return ResNet18(num_classes=num_classes, in_channels=in_channels)
    if model_name in ("resnet34", "r34"):
        return ResNet34(num_classes=num_classes, in_channels=in_channels)
    if model_name in ("resnet50", "r50"):
        return ResNet50(num_classes=num_classes, in_channels=in_channels)
    raise ValueError(f"Unknown --model: {model_name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", type=str, default="pose", choices=["pose", "full", "noface", "hands"])
    p.add_argument("--model", type=str, default="resnet18", choices=["cnn2d", "resnet18", "resnet34", "resnet50"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--split_path", type=str, default=str(DATA_DIR / "splits" / "split.json"))
    p.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Optional. Kalau tidak diisi, otomatis ambil checkpoint *__best.pt terbaru di checkpoints/<model>/<variant>/",
    )
    p.add_argument("--cm_normalize", type=str, default="true", choices=["none", "true", "pred", "all"])
    return p.parse_args()


def main():
    args = parse_args()

    seed(DEFAULT_SEED)
    device = get_device(prefer_gpu=True)
    print(f"Device: {device}")

    # --- Load split & labels ---
    split_path = Path(args.split_path)
    split_data = read_json(split_path)

    label2idx = split_data["label2idx"]
    idx2label = {int(k): v for k, v in split_data["idx2label"].items()}
    num_classes = len(label2idx)

    test_items = split_data["splits"]["test"]
    print(f"Total test samples: {len(test_items)}")

    # --- Dataset/Loader ---
    test_ds = SignDataset(test_items, variant=args.variant)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # --- Load checkpoint (best) ---
    if args.ckpt_path is not None:
        ckpt_path = Path(args.ckpt_path)
    else:
        ckpt_root = Path("checkpoints") / args.model / args.variant
        ckpt_path = max(ckpt_root.glob("*__best.pt"), key=lambda p: p.stat().st_mtime)

    print("Using checkpoint:", ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(args.model, num_classes=num_classes, in_channels=4).to(device)
    model.load_state_dict(ckpt["model_state"])

    # --- Inference ---
    test_loss, y_true, y_pred, y_prob = run_inference(model, test_loader, device)

    # --- Metrics ---
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)  # bagus kalau ada imbalance
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    # top-2 accuracy (opsional, tapi sering menarik)
    top2 = None
    if num_classes >= 2:
        top2 = top_k_accuracy_score(y_true, y_prob, k=2, labels=np.arange(num_classes))

    # per-class metrics
    prec_c, rec_c, f1_c, sup_c = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(num_classes), zero_division=0
    )

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

    print(f"\n=== TEST RESULT (variant={args.variant}, model={args.model}) ===")
    print(f"Test loss         : {test_loss:.4f}")
    print(f"Accuracy          : {acc:.4f}")
    print(f"Balanced Accuracy : {bal_acc:.4f}")
    print(f"F1 macro          : {f1_macro:.4f}")
    print(f"F1 weighted       : {f1_weighted:.4f}")
    if top2 is not None:
        print(f"Top-2 Accuracy     : {top2:.4f}")

    print("\n--- Per-class metrics ---")
    for i in range(num_classes):
        print(f"{i:>2} {idx2label[i]:<15} | P={prec_c[i]:.3f} R={rec_c[i]:.3f} F1={f1_c[i]:.3f} | support={sup_c[i]}")

    print("\n--- Classification report ---")
    # target_names harus urut label 0..C-1
    target_names = [idx2label[i] for i in range(num_classes)]
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0))

    print("\nConfusion matrix (counts):")
    print(cm)

    # --- Save & plot confusion matrices ---
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    cm_path = out_dir / f"cm_{args.model}_{args.variant}.png"
    cmn_path = out_dir / f"cm_{args.model}_{args.variant}_norm_true.png"

    plot_confusion_matrix(cm, idx2label, save_path=str(cm_path), normalize=None)

    normalize = None if args.cm_normalize == "none" else args.cm_normalize
    if normalize is not None:
        plot_confusion_matrix(cm, idx2label, save_path=str(cmn_path), normalize=normalize)


if __name__ == "__main__":
    main()
