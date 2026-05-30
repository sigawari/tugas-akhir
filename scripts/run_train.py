"""
run_train.py
------------
Training pipeline BISINDO SLR dengan Stratified K-Fold CV.

Cara pakai:
    python scripts/run_train.py
    python scripts/run_train.py --model resnet34
    python scripts/run_train.py --model resnet18 --use-delta False

Model tersedia: cnn2d | resnet18 | resnet34 | resnet50
Config dari   : configs/train.yaml

Output:
    outputs/checkpoints/<label>_fold<k>_best.pt
    outputs/logs/train_<label>.json        ← hasil semua fold + agregat
    outputs/plots/loss_<label>_fold<k>.png
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
from sklearn.metrics import f1_score, confusion_matrix

from src.dataset       import BISINDODataset
from src.models        import build_model
from src.data_splitter import kfold_split
from src.utils         import load_config, get_logger


# ===========================================================================
# Args
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Training BISINDO SLR — K-Fold CV")
    p.add_argument("--model",     type=str,   default=None)
    p.add_argument("--epochs",    type=int,   default=None)
    p.add_argument("--batch",     type=int,   default=None)
    p.add_argument("--lr",        type=float, default=None)
    p.add_argument("--config",    type=str,   default="train.yaml")
    p.add_argument("--use-delta", type=lambda x: x.lower() == "true", default=None)
    p.add_argument("--label",     type=str,   default=None)
    return p.parse_args()


# ===========================================================================
# Evaluasi
# ===========================================================================

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits = model(X_b)
            loss   = criterion(logits, y_b)
            total_loss += loss.item() * len(y_b)
            preds       = logits.argmax(dim=1)
            correct    += (preds == y_b).sum().item()
            total      += len(y_b)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_b.cpu().numpy())

    return (
        total_loss / total,
        correct / total,
        f1_score(all_labels, all_preds, average="macro", zero_division=0),
        all_preds,
        all_labels,
    )


# ===========================================================================
# Train satu fold
# ===========================================================================

def train_one_fold(
    fold_idx, X_train, X_val, y_train, y_val,
    model_name, output_label, n_classes, in_channels, dropout,
    epochs, batch_size, lr, weight_decay, patience,
    use_delta, cfg_aug, device, ckpt_dir, plot_dir, logger,
):
    logger.info(f"\n{'─'*55}")
    logger.info(f"  FOLD {fold_idx + 1}  |  train={len(X_train)}  val={len(X_val)}")
    logger.info(f"{'─'*55}")

    train_ds = BISINDODataset(X_train, y_train, augment=True,
                              cfg_aug=cfg_aug, use_delta=use_delta)
    val_ds  = BISINDODataset(X_val,  y_val,  augment=False,
                              cfg_aug=None,      use_delta=use_delta)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0)
    val_loader  = DataLoader(val_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=0)

    model = build_model(model_name, n_classes=n_classes,
                        in_channels=in_channels, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    history = {"train_loss": [], "train_acc": [],
               "val_loss":   [], "val_acc":   [], "val_f1": []}
    best_val_loss  = float("inf")
    patience_count = 0
    ckpt_path = ckpt_dir / f"{output_label}_fold{fold_idx+1}_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out  = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss    += loss.item() * len(y_b)
            t_correct += (out.argmax(1) == y_b).sum().item()
            t_total   += len(y_b)
        scheduler.step()

        tr_loss = t_loss / t_total
        tr_acc  = t_correct / t_total
        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        logger.info(
            f"  [{epoch:03d}/{epochs}] "
            f"Train Loss={tr_loss:.4f} Acc={tr_acc:.3f} | "
            f"val Loss={val_loss:.4f} Acc={val_acc:.3f} F1={val_f1:.3f} "
            f"Pat={patience_count}/{patience}"
        )

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_loss": val_loss, "val_acc": val_acc,
                        "val_f1": val_f1}, ckpt_path)
            logger.info(f"    ✓ Checkpoint disimpan (val_loss={val_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= patience:
                logger.info(f"    Early stopping epoch {epoch}.")
                break

    # Evaluasi final dengan best checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    _, final_acc, final_f1, final_preds, final_gt = evaluate(
        model, val_loader, criterion, device)
    cm = confusion_matrix(final_gt, final_preds)

    logger.info(f"  → Fold {fold_idx+1} final: Acc={final_acc:.4f}  F1={final_f1:.4f}"
                f"  BestEpoch={ckpt['epoch']}")

    # Plot
    _save_plot(history, plot_dir / f"loss_{output_label}_fold{fold_idx+1}.png", logger)

    return {
        "fold":        fold_idx + 1,
        "accuracy":    final_acc,
        "f1_macro":    final_f1,
        "best_epoch":  ckpt["epoch"],
        "confusion_matrix": cm.tolist(),
        "history":     history,
    }


# ===========================================================================
# Plot
# ===========================================================================

def _save_plot(history, path, logger):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ep = range(1, len(history["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(ep, history["train_loss"], label="Train")
        axes[0].plot(ep, history["val_loss"],   label="val")
        axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)
        axes[1].plot(ep, history["train_acc"],  label="Train")
        axes[1].plot(ep, history["val_acc"],    label="val")
        axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(True)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"  Plot → {path}")
    except ImportError:
        pass


# ===========================================================================
# Main
# ===========================================================================

def main():
    logger = get_logger("Train")
    args   = parse_args()
    cfg    = load_config(args.config)

    model_name   = args.model or cfg["model"]["resnet_version"]
    output_label = args.label or model_name
    n_classes    = cfg["model"]["num_classes"]
    in_channels  = cfg["model"]["input_channels"]
    dropout      = cfg["model"].get("dropout", 0.5)
    use_delta    = (args.use_delta if args.use_delta is not None
                    else cfg["model"].get("use_delta", True))

    epochs       = args.epochs or cfg["training"]["epochs"]
    batch_size   = args.batch  or cfg["training"]["batch_size"]
    lr           = args.lr     or cfg["training"]["learning_rate"]
    weight_decay = cfg["training"]["weight_decay"]
    patience     = cfg["training"]["early_stopping_patience"]
    seed         = cfg["training"]["seed"]
    n_splits     = cfg["split"].get("n_splits", 5)
    cfg_aug      = cfg.get("augmentation", None)

    npy_dir  = ROOT_DIR / cfg["paths"]["npy_dir"]
    ckpt_dir = ROOT_DIR / cfg["paths"]["checkpoint_dir"]
    log_dir  = ROOT_DIR / cfg["paths"]["log_dir"]
    plot_dir = ROOT_DIR / cfg["paths"]["plot_dir"]
    for d in [ckpt_dir, log_dir, plot_dir]:
        d.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Model: {model_name} | Label: {output_label} | "
                f"use_delta: {use_delta} | Device: {device}")

    X      = np.load(npy_dir / "X.npy")
    y      = np.load(npy_dir / "y.npy")
    labels = np.load(npy_dir / "labels.npy", allow_pickle=True)
    logger.info(f"X: {X.shape} | Kelas: {list(labels)}")

    # K-Fold split
    folds = kfold_split(X, y, n_splits=n_splits, seed=seed)

    fold_results = []
    for fold_idx, (X_train, X_val, y_train, y_val) in enumerate(folds):
        result = train_one_fold(
            fold_idx, X_train, X_val, y_train, y_val,
            model_name, output_label, n_classes, in_channels, dropout,
            epochs, batch_size, lr, weight_decay, patience,
            use_delta, cfg_aug, device, ckpt_dir, plot_dir, logger,
        )
        fold_results.append(result)

    # Agregat semua fold
    accs = [r["accuracy"]  for r in fold_results]
    f1s  = [r["f1_macro"]  for r in fold_results]
    eps  = [r["best_epoch"] for r in fold_results]

    logger.info("\n" + "=" * 55)
    logger.info(f"  MODEL    : {model_name} ({output_label})")
    logger.info(f"  Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    logger.info(f"  F1-Macro : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    logger.info(f"  Epochs   : {eps}  (mean={np.mean(eps):.1f})")
    logger.info("=" * 55)

    # Simpan log
    log_path = log_dir / f"train_{output_label}.json"
    with open(log_path, "w") as f:
        json.dump({
            "model":        model_name,
            "label":        output_label,
            "use_delta":    use_delta,
            "n_splits":     n_splits,
            "folds":        fold_results,
            "aggregate": {
                "accuracy_mean": float(np.mean(accs)),
                "accuracy_std":  float(np.std(accs)),
                "f1_mean":       float(np.mean(f1s)),
                "f1_std":        float(np.std(f1s)),
                "best_epoch_mean": float(np.mean(eps)),
                "labels":        list(labels),
            },
        }, f, indent=2)
    logger.info(f"Log → {log_path}")


if __name__ == "__main__":
    main()