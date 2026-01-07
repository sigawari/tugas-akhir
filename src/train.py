# train.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
)

import matplotlib.pyplot as plt

from utils import (
    DATA_DIR,
    PROCESSED_DIR,
    DEFAULT_SEED,
    seed,
    get_device,
    read_json,
    ensure_dir,
    wandb_init,
    wandb_log,
    wandb_finish,
    wandb_log_image,
    wandb_log_artifact_file,
    wandb_set_summary,
)
from models import ResNet2DSign
from metrics import accuracy_from_logits


# -------------------------
# Dataset
# -------------------------
class SignDataset(Dataset):
    """Dataset sign language berbasis NPY + split index.
    Output x: (C=4, H=T, W=L) untuk ResNet2D.
    """

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
        seq = X_word[seq_idx]  # (T, D)

        T, D = seq.shape
        if D % 4 != 0:
            raise ValueError(f"D={D} tidak habis dibagi 4, cek build_dataset.py (fitur per landmark harus 4).")
        L = D // 4

        seq = seq.reshape(T, L, 4)          # (T, L, 4)
        seq = np.transpose(seq, (2, 0, 1))  # (4, T, L)

        x_tensor = torch.from_numpy(seq).float()
        y_tensor = torch.tensor(label, dtype=torch.long)
        return {"x": x_tensor, "y": y_tensor, "word": word, "index": seq_idx}


# -------------------------
# Helpers
# -------------------------
def fmt_lr(lr: float) -> str:
    # 0.0001 -> 1e-04 (stabil untuk nama)
    return f"{lr:.0e}" if lr < 1e-3 else f"{lr}".replace(".", "p")


def infer_T_L_from_variant(variant: str) -> tuple[int, int]:
    """Ambil (T, L) dari satu file sample (halo)."""
    # ambil kata apa saja yang ada
    words = [d.name for d in PROCESSED_DIR.iterdir() if d.is_dir()]
    if not words:
        raise RuntimeError(f"Tidak ada folder kata di {PROCESSED_DIR}")
    word = sorted(words)[0]
    X = np.load(PROCESSED_DIR / word / variant / "X.npy")
    # X: (N, T, D)
    _, T, D = X.shape
    if D % 4 != 0:
        raise ValueError(f"D={D} tidak habis dibagi 4.")
    L = D // 4
    return T, L


def save_confusion_matrix_png(cm: np.ndarray, idx2label: Dict[int, str], save_path: Path, normalize_true: bool = False) -> None:
    labels = [idx2label[i] for i in range(len(idx2label))]

    cm_plot = cm.astype(np.float32)
    title = "Confusion Matrix (Test)"
    if normalize_true:
        eps = 1e-12
        cm_plot = cm_plot / (cm_plot.sum(axis=1, keepdims=True) + eps)
        title = "Confusion Matrix (Test) - Normalized by True"

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap=plt.cm.Blues)
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

    thresh = cm_plot.max() / 2.0 if cm_plot.size else 0.5
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            text = f"{cm_plot[i, j]:.2f}" if normalize_true else str(int(cm_plot[i, j]))
            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if cm_plot[i, j] > thresh else "black",
                fontsize=10
            )

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


# -------------------------
# Train/Eval
# -------------------------
def train_one_epoch(model, loader, optimizer, device) -> Dict[str, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_acc = 0.0
    total_count = 0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        bsz = y.size(0)
        acc = accuracy_from_logits(logits.detach(), y.detach())

        total_loss += loss.item() * bsz
        total_acc += acc * bsz
        total_count += bsz

    return {"loss": total_loss / total_count, "acc": total_acc / total_count}


@torch.no_grad()
def evaluate_basic(model, loader, device) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_acc = 0.0
    total_count = 0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        logits = model(x)
        loss = criterion(logits, y)

        bsz = y.size(0)
        acc = accuracy_from_logits(logits, y)

        total_loss += loss.item() * bsz
        total_acc += acc * bsz
        total_count += bsz

    return {"loss": total_loss / total_count, "acc": total_acc / total_count}


@torch.no_grad()
def evaluate_test_full(model, loader, device, num_classes: int) -> Dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_count = 0

    y_true = []
    y_pred = []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        logits = model(x)
        loss = criterion(logits, y)

        bsz = y.size(0)
        total_loss += loss.item() * bsz
        total_count += bsz

        preds = torch.argmax(logits, dim=1)
        y_true.append(y.detach().cpu().numpy())
        y_pred.append(preds.detach().cpu().numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    acc = float((y_true == y_pred).mean())
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted"))
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

    return {
        "loss": float(total_loss / max(total_count, 1)),
        "acc": acc,
        "balanced_acc": bal_acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "cm": cm,
    }


# -------------------------
# Args
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", type=str, default="pose",
                   choices=["pose", "full", "noface", "hands"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wandb_project", type=str, default="ablation-landmark")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--split_path", type=str, default=str(DATA_DIR / "splits" / "split.json"))
    p.add_argument("--no_wandb", action="store_true")
    # early stopping
    p.add_argument("--early_stop_patience", type=int, default=10)
    p.add_argument("--min_delta", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # fairness seeds
    train_seed = DEFAULT_SEED
    seed(train_seed)

    device = get_device(prefer_gpu=True)
    print(f"Device: {device}")

    # load split.json
    split_path = Path(args.split_path)
    split_data = read_json(split_path)

    split_seed = int(split_data.get("meta", {}).get("seed", DEFAULT_SEED))
    label2idx = split_data["label2idx"]
    idx2label = {int(k): v for k, v in split_data["idx2label"].items()}
    num_classes = len(label2idx)

    train_items = split_data["splits"]["train"]
    val_items = split_data["splits"]["val"]
    test_items = split_data["splits"]["test"]

    print(f"Total train: {len(train_items)}, val: {len(val_items)}, test: {len(test_items)}")

    # dataset loaders
    train_ds = SignDataset(train_items, variant=args.variant)
    val_ds = SignDataset(val_items, variant=args.variant)
    test_ds = SignDataset(test_items, variant=args.variant)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # infer T & L for naming/metadata
    T_in, L_in = infer_T_L_from_variant(args.variant)

    # model
    model = ResNet2DSign(num_classes=num_classes, in_channels=4)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- W&B init (rapi) ---
    run_name = args.run_name or args.variant  # clean for ablation landmark

    group = f"landmark_T{T_in}_seed{train_seed}_split{split_seed}_lr{fmt_lr(args.lr)}_bs{args.batch_size}"
    tags = ["ablation", "landmark", args.variant]
    job_type = "ablation_landmark"

    wandb_run = None
    if not args.no_wandb:
        config = {
            "variant": args.variant,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "num_classes": num_classes,
            "train_seed": train_seed,
            "split_seed": split_seed,
            "T": T_in,
            "L": L_in,
            "model": "ResNet2DSign(resnet18)",
            "early_stop_patience": args.early_stop_patience,
            "min_delta": args.min_delta,
        }
        wandb_run = wandb_init(
            project=args.wandb_project,
            config=config,
            name=run_name,
            group=group,
            tags=tags,
            job_type=job_type,
        )

    # checkpoints
    ckpt_dir = ensure_dir(Path("checkpoints") / args.variant)
    ckpt_path = ckpt_dir / f"{args.variant}_r18_lr{fmt_lr(args.lr)}_bs{args.batch_size}_seed{train_seed}_T{T_in}_L{L_in}__best.pt"

    best_val_acc = -1.0
    best_epoch = -1
    wait = 0

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, device)
        val_stats = evaluate_basic(model, val_loader, device)

        msg = (
            f"Epoch {epoch:03d} | "
            f"train_loss={train_stats['loss']:.4f}, train_acc={train_stats['acc']:.4f} | "
            f"val_loss={val_stats['loss']:.4f}, val_acc={val_stats['acc']:.4f}"
        )
        print(msg)

        # log per-epoch
        if wandb_run is not None:
            wandb_log({
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_acc": train_stats["acc"],
                "val_loss": val_stats["loss"],
                "val_acc": val_stats["acc"],
                "early_stop/wait": wait,
                "early_stop/best_epoch": best_epoch,
                "early_stop/patience": args.early_stop_patience,
            }, step=epoch)

        improved = (val_stats["acc"] > best_val_acc + args.min_delta)
        if improved:
            best_val_acc = float(val_stats["acc"])
            best_epoch = epoch
            wait = 0

            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": best_val_acc,
                "config": vars(args),
                "label2idx": label2idx,
                "idx2label": idx2label,
                "train_seed": train_seed,
                "split_seed": split_seed,
                "T": T_in,
                "L": L_in,
            }, ckpt_path)

            print(f"  🔥 New best val_acc={best_val_acc:.4f} | checkpoint saved to {ckpt_path}")

            # log artifact hanya saat new best
            if wandb_run is not None:
                wandb_log_artifact_file(
                    run=wandb_run,
                    file_path=ckpt_path,
                    artifact_name=f"ckpt-{args.variant}",
                    artifact_type="model",
                    metadata={
                        "variant": args.variant,
                        "best_epoch": best_epoch,
                        "best_val_acc": best_val_acc,
                        "train_seed": train_seed,
                        "split_seed": split_seed,
                        "T": T_in,
                        "L": L_in,
                        "lr": args.lr,
                        "batch_size": args.batch_size,
                    },
                    aliases=["best"],
                )
        else:
            wait += 1

        # early stopping
        if wait >= args.early_stop_patience:
            print(f"⏹️ Early stopping at epoch {epoch} (best_epoch={best_epoch}, best_val_acc={best_val_acc:.4f})")
            if wandb_run is not None:
                wandb_log({"early_stop/triggered": 1, "early_stop/stop_epoch": epoch}, step=epoch)
            break

    # --- load best ckpt & test evaluation ---
    best_ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state"])

    test_stats = evaluate_test_full(model, test_loader, device, num_classes)

    print(f"\n✅ BEST CHECKPOINT TEST | test_loss={test_stats['loss']:.4f}, test_acc={test_stats['acc']:.4f}")
    print(f"   test_bal_acc={test_stats['balanced_acc']:.4f} | f1_macro={test_stats['f1_macro']:.4f} | f1_weighted={test_stats['f1_weighted']:.4f}")

    # save confusion matrix
    reports_dir = ensure_dir(Path("reports") / args.variant)
    cm_path = reports_dir / "confusion_matrix_test.png"
    cmn_path = reports_dir / "confusion_matrix_test_norm_true.png"

    save_confusion_matrix_png(test_stats["cm"], idx2label, cm_path, normalize_true=False)
    save_confusion_matrix_png(test_stats["cm"], idx2label, cmn_path, normalize_true=True)

    # log final metrics + images into W&B
    if wandb_run is not None:
        wandb_log({
            "best_epoch": best_ckpt["epoch"],
            "best_val_acc": best_ckpt["val_acc"],
            "test_loss": test_stats["loss"],
            "test_acc": test_stats["acc"],
            "test_balanced_acc": test_stats["balanced_acc"],
            "test_f1_macro": test_stats["f1_macro"],
            "test_f1_weighted": test_stats["f1_weighted"],
        })

        wandb_log_image("test/confusion_matrix", cm_path, caption=f"CM test ({args.variant})")
        wandb_log_image("test/confusion_matrix_norm_true", cmn_path, caption=f"CM test norm-true ({args.variant})")

        # summary untuk tabel ablation
        wandb_set_summary({
            "variant": args.variant,
            "T": T_in,
            "L": L_in,
            "train_seed": train_seed,
            "split_seed": split_seed,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "best_epoch": best_ckpt["epoch"],
            "best_val_acc": float(best_ckpt["val_acc"]),
            "test_loss": test_stats["loss"],
            "test_acc": test_stats["acc"],
            "test_balanced_acc": test_stats["balanced_acc"],
            "test_f1_macro": test_stats["f1_macro"],
            "test_f1_weighted": test_stats["f1_weighted"],
        })

    if wandb_run is not None:
        wandb_finish()


if __name__ == "__main__":
    main()
