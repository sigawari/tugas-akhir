"""
run_train.py
------------
Training pipeline BISINDO SLR.

Cara pakai:
    python scripts/run_train.py
    python scripts/run_train.py --model resnet34
    python scripts/run_train.py --model cnn2d --epochs 100

Model tersedia: cnn2d | resnet18 | resnet34 | resnet50
Config dari   : configs/train.yaml (override via argumen)

Output:
    outputs/checkpoints/<model>_best.pt
    outputs/logs/train_<model>.json
    outputs/plots/loss_<model>.png
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

from src.dataset        import BISINDODataset
from src.models         import build_model, count_parameters
from src.data_splitter  import chronological_split
from src.utils          import load_config, get_logger


# ===========================================================================
# Args
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Training BISINDO SLR")
    p.add_argument("--model",    type=str,   default=None,
                   help="Override train.yaml model.resnet_version (cnn2d|resnet18|resnet34|resnet50)")
    p.add_argument("--epochs",   type=int,   default=None, help="Override epochs")
    p.add_argument("--batch",    type=int,   default=None, help="Override batch_size")
    p.add_argument("--lr",       type=float, default=None, help="Override learning_rate")
    p.add_argument("--config",   type=str,   default="train.yaml")
    p.add_argument("--use-delta", type=lambda x: x.lower() == "true", default=None)
    p.add_argument("--label",     type=str, default=None)
    return p.parse_args()


# ===========================================================================
# Evaluasi
# ===========================================================================

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)

            total_loss += loss.item() * len(y_batch)
            preds       = logits.argmax(dim=1)
            correct    += (preds == y_batch).sum().item()
            total      += len(y_batch)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, accuracy, f1, all_preds, all_labels


# ===========================================================================
# Plot
# ===========================================================================

def save_plots(history: dict, path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = range(1, len(history["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(epochs, history["train_loss"], label="Train")
        axes[0].plot(epochs, history["val_loss"],   label="Test")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend(); axes[0].grid(True)

        axes[1].plot(epochs, history["train_acc"], label="Train")
        axes[1].plot(epochs, history["val_acc"],   label="Test")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].legend(); axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Plot → {path}")
    except ImportError:
        logger.warning("matplotlib tidak tersedia, skip plot.")


# ===========================================================================
# Main
# ===========================================================================

def main():
    logger = get_logger("Train")
    args   = parse_args()
    cfg    = load_config(args.config)

    # Ambil config, bisa di-override via argumen
    model_name   = args.model or cfg["model"]["resnet_version"]
    output_label = args.label or model_name
    n_classes   = cfg["model"]["num_classes"]
    in_channels = cfg["model"]["input_channels"]   # harus 1
    dropout     = cfg["model"].get("dropout", 0.3)

    epochs      = args.epochs or cfg["training"]["epochs"]
    batch_size  = args.batch  or cfg["training"]["batch_size"]
    lr          = args.lr     or cfg["training"]["learning_rate"]

    use_delta  = args.use_delta if args.use_delta is not None else cfg["model"].get("use_delta", True)

    weight_decay = cfg["training"]["weight_decay"]
    patience    = cfg["training"]["early_stopping_patience"]
    seed        = cfg["training"]["seed"]

    n_total     = cfg["split"]["n_total"]
    n_test      = cfg["split"]["n_test"]
    cfg_aug     = cfg.get("augmentation", None)

    npy_dir      = ROOT_DIR / cfg["paths"]["npy_dir"]
    ckpt_dir     = ROOT_DIR / cfg["paths"]["checkpoint_dir"]
    log_dir      = ROOT_DIR / cfg["paths"]["log_dir"]
    plot_dir     = ROOT_DIR / cfg["paths"]["plot_dir"]

    for d in [ckpt_dir, log_dir, plot_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  MODEL    : {model_name} ({output_label})")

    # Load data
    X      = np.load(npy_dir / "X.npy")
    y      = np.load(npy_dir / "y.npy")
    labels = np.load(npy_dir / "labels.npy", allow_pickle=True)
    logger.info(f"X: {X.shape} | Kelas: {list(labels)}")

    # Split kronologis
    X_train, X_test, y_train, y_test = chronological_split(
        X, y, n_total=n_total, n_test=n_test
    )

    # Dataset & DataLoader

    train_ds = BISINDODataset(X_train, y_train, augment=True,
                            cfg_aug=cfg_aug, use_delta=use_delta)
    test_ds  = BISINDODataset(X_test,  y_test,  augment=False,
                            cfg_aug=None,    use_delta=use_delta)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=torch.cuda.is_available())

    logger.info(f"Train: {len(train_ds)} | Test: {len(test_ds)}")

    # Model
    model = build_model(model_name, n_classes=n_classes,
                        in_channels=in_channels, dropout=dropout).to(device)
    # for name, param in model.named_parameters():
    #     if not any(k in name for k in ['layer4', 'fc']):
    #         param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameter trainable setelah freeze: {trainable:,}")

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # Training loop
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_val_loss  = float("inf")
    patience_count = 0
    ckpt_path = ckpt_dir / f"{output_label}_best.pt"
    

    # Log awal — tampilkan keduanya
    logger.info(f"Model: {model_name} | Label: {output_label} | Device: {device}")

    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out  = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_loss    += loss.item() * len(y_b)
            t_correct += (out.argmax(1) == y_b).sum().item()
            t_total   += len(y_b)
        scheduler.step()

        tr_loss = t_loss / t_total
        tr_acc  = t_correct / t_total

        # Eval
        val_loss, val_acc, val_f1, _, _ = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        logger.info(
            f"[{epoch:03d}/{epochs}] "
            f"Train Loss={tr_loss:.4f} Acc={tr_acc:.3f} | "
            f"Test  Loss={val_loss:.4f} Acc={val_acc:.3f} F1={val_f1:.3f}"
        )

        # Checkpoint & early stopping
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save({
                "epoch": epoch, "model_name": model_name,
                "model_state": model.state_dict(),
                "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
                "labels": list(labels),
            }, ckpt_path)
            logger.info(f"  ✓ Checkpoint disimpan (val_loss={val_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= patience:
                logger.info(f"Early stopping di epoch {epoch}.")
                break

    # Evaluasi final dengan best checkpoint
    logger.info("Memuat best checkpoint untuk evaluasi final...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    _, final_acc, final_f1, final_preds, final_gt = evaluate(
        model, test_loader, criterion, device
    )
    cm = confusion_matrix(final_gt, final_preds)

    logger.info("=" * 55)
    logger.info(f"  MODEL    : {model_name}")
    logger.info(f"  Accuracy : {final_acc:.4f} ({final_acc*100:.2f}%)")
    logger.info(f"  F1-Macro : {final_f1:.4f}")
    logger.info(f"  Best epoch: {ckpt['epoch']}")
    logger.info(f"  Confusion Matrix:\n{cm}")
    logger.info("=" * 55)

    # Simpan log & plot
    log_path = log_dir / f"train_{output_label}.json"
    with open(log_path, "w") as f:
        json.dump({
            "model": model_name,
            "label":  output_label,
            "history": history,
            "final": {
                "accuracy": final_acc, "f1_macro": final_f1,
                "best_epoch": ckpt["epoch"],
                "confusion_matrix": cm.tolist(),
                "labels": list(labels),
            }
        }, f, indent=2)
    logger.info(f"Log → {log_path}")

    save_plots(history, plot_dir / f"loss_{output_label}.png")


if __name__ == "__main__":
    main()