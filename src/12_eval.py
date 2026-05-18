# 12_eval.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    top_k_accuracy_score,
)

from models import Basic2DCNN, ResNet18, ResNet34, ResNet50, CNN2DResidual
from utils import (
    DEFAULT_SEED,
    seed,
    get_device,
    ensure_dir,
)
from dataset import create_dataloaders


# =========================
# MODEL FACTORY
# =========================

def build_model(model_name: str, num_classes: int, in_channels: int = 4) -> torch.nn.Module:
    model_name = model_name.lower()

    if model_name in ("cnn2d", "basic2dcnn", "cnn"):
        return Basic2DCNN(num_classes=num_classes, in_channels=in_channels)

    if model_name in ("cnn2d_residual", "residualcnn", "custom_resnet"):
        return CNN2DResidual(num_classes=num_classes, in_channels=in_channels)

    if model_name in ("resnet18", "r18"):
        return ResNet18(num_classes=num_classes, in_channels=in_channels)

    if model_name in ("resnet34", "r34"):
        return ResNet34(num_classes=num_classes, in_channels=in_channels)

    if model_name in ("resnet50", "r50"):
        return ResNet50(num_classes=num_classes, in_channels=in_channels)

    raise ValueError(f"Unknown model: {model_name}")


# =========================
# INFERENCE
# =========================

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device | str,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
      avg_loss,
      y_true (N,),
      y_pred (N,),
      y_prob (N, C)
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

        prob = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1)

        ys.append(y.detach().cpu().numpy())
        preds.append(pred.detach().cpu().numpy())
        probs.append(prob.detach().cpu().numpy())

    avg_loss = total_loss / max(total_count, 1)
    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(preds, axis=0)
    y_prob = np.concatenate(probs, axis=0)

    return avg_loss, y_true, y_pred, y_prob


# =========================
# CONFUSION MATRIX PLOT
# =========================

def plot_confusion_matrix(
    cm: np.ndarray,
    idx2label: Dict[int, str],
    save_path: Optional[str] = None,
    normalize: Optional[str] = None,  # None | "true" | "pred" | "all"
) -> None:
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
            title = "Confusion Matrix (Normalized by All)"
        else:
            raise ValueError("normalize harus None/'true'/'pred'/'all'")

    fig, ax = plt.subplots(figsize=(8, 7))
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

    thresh = cm_to_plot.max() / 2.0 if cm_to_plot.size else 0.5
    for i in range(cm_to_plot.shape[0]):
        for j in range(cm_to_plot.shape[1]):
            text = f"{int(cm_to_plot[i, j])}" if normalize is None else f"{cm_to_plot[i, j]:.2f}"
            ax.text(
                j, i, text,
                ha="center",
                va="center",
                color="white" if cm_to_plot[i, j] > thresh else "black",
                fontsize=10,
            )

    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved confusion matrix to: {save_path}")

    plt.show()


# =========================
# ARGPARSE
# =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--model", type=str, default="resnet18",
                   choices=["cnn2d", "cnn2d_residual", "resnet18", "resnet34", "resnet50"])

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)

    p.add_argument("--split_path", type=str, default=None)

    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.2)

    p.add_argument("--use_delta", type=int, default=1)

    p.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Kalau kosong, otomatis cari checkpoint *__best.pt terbaru di checkpoints/<model>/"
    )

    p.add_argument("--report_dir", type=str, default="reports")
    p.add_argument("--cm_normalize", type=str, default="true",
                   choices=["none", "true", "pred", "all"])

    return p.parse_args()


# =========================
# CKPT HELPER
# =========================

def find_latest_best_checkpoint(model_name: str, ckpt_root: str = "checkpoints") -> Path:
    ckpt_dir = Path(ckpt_root) / model_name
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory tidak ditemukan: {ckpt_dir}")

    candidates = list(ckpt_dir.glob("*__best.pt"))
    if not candidates:
        raise FileNotFoundError(f"Tidak ada checkpoint *__best.pt di {ckpt_dir}")

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest


# =========================
# MAIN
# =========================

def main() -> None:
    args = parse_args()
    seed(args.seed)

    device = get_device(prefer_gpu=True)
    print(f"Using device: {device}")

    split_path = args.split_path
    if split_path is None:
        split_path = Path(__file__).resolve().parent.parent / "dataset" / "splits" / "split_70_20_10.json"

    ckpt_config = ckpt.get("config", {})
    if "use_delta" in ckpt_config:
        eval_use_delta = bool(ckpt_config["use_delta"])
    elif "in_channels" in ckpt_config:
        eval_use_delta = ckpt_config["in_channels"] == 4
    else:
        # Fallback: gunakan args.use_delta jika config tidak ada
        eval_use_delta = bool(args.use_delta)
    
    print(f"📥 Dataloader akan menggunakan use_delta={eval_use_delta} (dari checkpoint config)")

    # augment=False untuk semua split saat eval
    train_loader, val_loader, split_data = create_dataloaders(
        split_path=split_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        train_augment=False,
        use_delta=eval_use_delta, 
    )

    num_classes = split_data["meta"]["num_classes"]
    idx2label = {int(k): v for k, v in split_data["idx2label"].items()}

    # checkpoint
    if args.ckpt_path is not None:
        ckpt_path = Path(args.ckpt_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint tidak ditemukan: {ckpt_path}")
    else:
        ckpt_path = find_latest_best_checkpoint(args.model)

    print(f"Using checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    ckpt_config = ckpt.get("config", {})

    if "use_delta" in ckpt_config:
        in_channels = 4 if ckpt_config["use_delta"] else 2
    elif "in_channels" in ckpt_config:
        in_channels = ckpt_config["in_channels"]
    else:
        # Fallback: ambil langsung dari shape weight layer pertama di checkpoint
        first_conv = ckpt["model_state"].get("features.0.weight")
        if first_conv is not None:
            in_channels = first_conv.shape[1]
        else:
            in_channels = 4 if bool(args.use_delta) else 2

    print(f"📥 Model di-load dengan in_channels={in_channels} (auto-detected dari checkpoint)")

    model = build_model(
        model_name=args.model,
        num_classes=num_classes,
        in_channels=in_channels,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])

    # inference
    val_loss, y_true, y_pred, y_prob = run_inference(
        model=model,
        loader=val_loader,
        device=device,
    )

    # metrics
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")

    top2 = None
    if num_classes >= 2:
        top2 = top_k_accuracy_score(
            y_true,
            y_prob,
            k=2,
            labels=np.arange(num_classes),
        )

    prec_c, rec_c, f1_c, sup_c = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

        # === HITUNG TP, TN, FP, FN PER KELAS ===
    tp = np.diag(cm)                  # True Positives
    fp = cm.sum(axis=0) - tp          # False Positives
    fn = cm.sum(axis=1) - tp          # False Negatives
    tn = cm.sum() - (tp + fp + fn)    # True Negatives

    print("\n--- Per-class TP, TN, FP, FN ---")
    for i in range(num_classes):
        print(
            f"{i:>2} {idx2label[i]:<15} | "
            f"TP={tp[i]:<4} FP={fp[i]:<4} FN={fn[i]:<4} TN={tn[i]:<4}"
        )

    print(f"\n=== TEST RESULT (model={args.model}) ===")
    print(f"Checkpoint         : {ckpt_path}")
    print(f"Best val loss      : {ckpt.get('best_val_loss', 'N/A')}")
    print(f"Saved epoch        : {ckpt.get('epoch', 'N/A')}")
    print(f"Val loss          : {val_loss:.4f}")
    print(f"Accuracy           : {acc:.4f}")
    print(f"Balanced Accuracy  : {bal_acc:.4f}")
    print(f"F1              : {f1_macro:.4f}")
    if top2 is not None:
        print(f"Top-2 Accuracy     : {top2:.4f}")

    print("\n--- Per-class metrics ---")
    for i in range(num_classes):
        print(
            f"{i:>2} {idx2label[i]:<15} | "
            f"P={prec_c[i]:.3f} R={rec_c[i]:.3f} F1={f1_c[i]:.3f} | support={sup_c[i]}"
        )

    print("\n--- Classification report ---")
    target_names = [idx2label[i] for i in range(num_classes)]
    print(classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4,
        zero_division=0,
    ))

    print("\nConfusion matrix (counts):")
    print(cm)

    # save reports
    report_dir = Path(args.report_dir)
    ensure_dir(report_dir)

    # 🔑 Ambil LR otomatis dari config yang tersimpan saat training
    lr_val = ckpt.get("config", {}).get("lr", "unknown")
    
    stem = ckpt_path.stem.replace("__best", "")
    cm_raw_path = report_dir / f"cm_{stem}_lr{lr_val}_raw.png"
    cm_norm_path = report_dir / f"cm_{stem}_lr{lr_val}_norm_{args.cm_normalize}.png"
    metrics_path = report_dir / f"metrics_{stem}_lr{lr_val}.json"

    plot_confusion_matrix(
        cm,
        idx2label,
        save_path=str(cm_raw_path),
        normalize=None,
    )

    normalize = None if args.cm_normalize == "none" else args.cm_normalize
    if normalize is not None:
        plot_confusion_matrix(
            cm,
            idx2label,
            save_path=str(cm_norm_path),
            normalize=normalize,
        )

    metrics_payload = {
        "checkpoint": str(ckpt_path),
        "saved_epoch": ckpt.get("epoch"),
        "best_val_loss": ckpt.get("best_val_loss"),
        "val_loss": float(val_loss),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "f1_macro": float(f1_macro),
        "top2_accuracy": None if top2 is None else float(top2),
        "per_class": {
            idx2label[i]: {
                "precision": float(prec_c[i]),
                "recall": float(rec_c[i]),
                "f1": float(f1_c[i]),
                "support": int(sup_c[i]),
                "TP": int(tp[i]),
                "FP": int(fp[i]),
                "FN": int(fn[i]),
                "TN": int(tn[i]),
            }
            for i in range(num_classes)
        },
        "confusion_matrix": cm.tolist(),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        import json
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    print(f"\nSaved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()