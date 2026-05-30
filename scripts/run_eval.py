"""
run_eval.py
-----------
Evaluasi checkpoint terhadap test set eksternal.

Cara pakai:
    python scripts/run_eval.py --checkpoint outputs/checkpoints/resnet18_delta_best.pt
    python scripts/run_eval.py --checkpoint outputs/checkpoints/resnet18_delta_best.pt --npy-dir data/processed/testnpy
"""

import sys
import json
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from src.dataset import BISINDODataset
from src.models  import build_model
from src.utils   import get_logger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--npy-dir",    type=str, default="data/processed/testnpy")
    p.add_argument("--batch",      type=int, default=8)
    p.add_argument("--use-delta",  type=lambda x: x.lower() == "true", default=True)
    return p.parse_args()


def main():
    logger = get_logger("Eval")
    args   = parse_args()

    ckpt_path = ROOT_DIR / args.checkpoint
    npy_dir   = ROOT_DIR / args.npy_dir

    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)

    model_name = ckpt["model_name"]
    labels     = ckpt["labels"]
    n_classes  = len(labels)

    logger.info(f"Checkpoint : {ckpt_path.name}")
    logger.info(f"Model      : {model_name}")
    logger.info(f"Best epoch : {ckpt['epoch']}")
    logger.info(f"Val acc    : {ckpt['val_acc']:.4f}")
    logger.info(f"Kelas      : {labels}")

    # Load test data
    X      = np.load(npy_dir / "X.npy")
    y      = np.load(npy_dir / "y.npy")
    logger.info(f"Test X shape: {X.shape} | N={len(X)}")

    test_ds     = BISINDODataset(X, y, augment=False,
                                 cfg_aug=None, use_delta=args.use_delta)
    test_loader = DataLoader(test_ds, batch_size=args.batch,
                             shuffle=False, num_workers=0)

    # Build & load model
    model = build_model(model_name, n_classes=n_classes, in_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b  = X_b.to(device), y_b.to(device)
            logits     = model(X_b)
            loss       = criterion(logits, y_b)
            total_loss += loss.item() * len(y_b)
            preds       = logits.argmax(dim=1)
            correct    += (preds == y_b).sum().item()
            total      += len(y_b)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_b.cpu().numpy())

    acc = correct / total
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm  = confusion_matrix(all_labels, all_preds)

    logger.info("=" * 55)
    logger.info(f"  TEST RESULTS ({args.npy_dir})")
    logger.info(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    logger.info(f"  F1-Macro : {f1:.4f}")
    logger.info(f"  Loss     : {total_loss/total:.4f}")
    logger.info("=" * 55)
    logger.info(f"\n{classification_report(all_labels, all_preds, target_names=labels)}")
    logger.info(f"Confusion Matrix:\n{cm}")

    # Simpan hasil
    out_path = ROOT_DIR / "outputs" / "logs" / f"eval_{ckpt_path.stem}.json"
    with open(out_path, "w") as f:
        json.dump({
            "checkpoint": ckpt_path.name,
            "model": model_name,
            "npy_dir": str(args.npy_dir),
            "use_delta": args.use_delta,
            "test_size": total,
            "accuracy": float(acc),
            "f1_macro": float(f1),
            "confusion_matrix": cm.tolist(),
            "labels": list(labels),
            "per_sample": {
                "preds":  [int(p) for p in all_preds],
                "labels": [int(l) for l in all_labels],
            }
        }, f, indent=2)
    logger.info(f"Hasil → {out_path}")


if __name__ == "__main__":
    main()