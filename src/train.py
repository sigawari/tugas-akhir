# train.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
    wandb_is_available,
    wandb_log_image,
    wandb_set_summary,
)
from models import Basic2DCNN, ResNet18, ResNet34, ResNet50


# -------------------------
# Dataset
# -------------------------
class SignDataset(Dataset):
    """Dataset untuk sign language berbasis NPY + split index.
    Return x: (C=4, H=T, W=L)
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
        assert D % 4 == 0, f"D={D} tidak habis dibagi 4, cek build_dataset.py"
        L = D // 4

        seq = seq.reshape(T, L, 4)          # (T, L, 4)
        seq = np.transpose(seq, (2, 0, 1))  # (4, T, L)

        x_tensor = torch.from_numpy(seq).float()
        y_tensor = torch.tensor(label, dtype=torch.long)

        return {"x": x_tensor, "y": y_tensor, "word": word, "index": seq_idx}


# -------------------------
# Metrics helpers (tanpa sklearn)
# -------------------------
def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


def confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(targets.view(-1), preds.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm


def f1_from_confusion(cm: torch.Tensor) -> Tuple[float, float]:
    # macro F1 dan weighted F1 dari confusion matrix
    cm = cm.float()
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
    # rata-rata recall per kelas
    cm = cm.float()
    tp = torch.diag(cm)
    fn = cm.sum(1) - tp
    recall = tp / (tp + fn + 1e-12)
    return recall.mean().item()


def plot_confusion_matrix_image(cm: torch.Tensor, idx2label: Dict[int, str], title: str) -> "Any":
    import matplotlib.pyplot as plt

    cm_np = cm.cpu().numpy()
    labels = [idx2label[i] for i in range(len(idx2label))]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_np, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True",
        xlabel="Pred",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_np.max() / 2.0 if cm_np.max() > 0 else 0.5
    for i in range(cm_np.shape[0]):
        for j in range(cm_np.shape[1]):
            ax.text(j, i, str(cm_np[i, j]),
                    ha="center", va="center",
                    color="white" if cm_np[i, j] > thresh else "black")
    fig.tight_layout()
    return fig


# -------------------------
# Train / Eval
# -------------------------
def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    device: torch.device) -> Dict[str, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_acc, total_count = 0.0, 0.0, 0

    for batch in loader:
        x = batch["x"].to(device)  # (B,4,T,L)
        y = batch["y"].to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_from_logits(logits.detach(), y.detach()) * bs
        total_count += bs

    return {"loss": total_loss / total_count, "acc": total_acc / total_count}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_acc, total_count = 0.0, 0.0, 0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        logits = model(x)
        loss = criterion(logits, y)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_from_logits(logits, y) * bs
        total_count += bs

    return {"loss": total_loss / total_count, "acc": total_acc / total_count}


@torch.no_grad()
def evaluate_test_full(model: nn.Module, loader: DataLoader, device: torch.device,
                       num_classes: int) -> Dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss, total_count = 0.0, 0
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        logits = model(x)
        loss = criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_count += bs

    preds_t = torch.cat(all_preds, dim=0)
    targets_t = torch.cat(all_targets, dim=0)

    cm = confusion_matrix(preds_t, targets_t, num_classes)
    acc = (preds_t == targets_t).float().mean().item()
    bal_acc = balanced_accuracy_from_confusion(cm)
    f1_macro, f1_weighted = f1_from_confusion(cm)

    return {
        "loss": total_loss / total_count,
        "acc": acc,
        "balanced_acc": bal_acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "cm": cm,
    }


def build_model(model_name: str, num_classes: int, in_channels: int = 4) -> nn.Module:
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


def model_to_project(model_name: str) -> str:
    # 4 project W&B (satu per model)
    model_name = model_name.lower()
    if model_name in ("cnn2d", "basic2dcnn", "cnn"):
        return "ablation-model-cnn2d"
    if model_name in ("resnet18", "r18"):
        return "ablation-model-resnet18"
    if model_name in ("resnet34", "r34"):
        return "ablation-model-resnet34"
    if model_name in ("resnet50", "r50"):
        return "ablation-model-resnet50"
    return "ablation-model-unknown"


# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", type=str, default="pose",
                   choices=["pose", "full", "noface", "hands"])
    p.add_argument("--model", type=str, default="resnet18",
                   choices=["cnn2d", "resnet18", "resnet34", "resnet50"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--split_path", type=str, default=str(DATA_DIR / "splits" / "split.json"))
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--scheduler", type=str, default="plateau", choices=["none", "plateau", "cosine"])
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default=None)  # optional override
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # fairness
    train_seed = DEFAULT_SEED
    seed(train_seed)

    device = get_device(prefer_gpu=True)
    print(f"Device: {device}")

    split_path = Path(args.split_path)
    split_data = read_json(split_path)

    split_seed = split_data.get("meta", {}).get("seed", None)
    label2idx = split_data["label2idx"]
    idx2label = {int(k): v for k, v in split_data["idx2label"].items()}
    num_classes = len(label2idx)

    train_items = split_data["splits"]["train"]
    val_items = split_data["splits"]["val"]
    test_items = split_data["splits"]["test"]
    print(f"Total train: {len(train_items)}, val: {len(val_items)}, test: {len(test_items)}")

    train_ds = SignDataset(train_items, variant=args.variant)
    val_ds = SignDataset(val_items, variant=args.variant)
    test_ds = SignDataset(test_items, variant=args.variant)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # infer T & L dari 1 sample (buat run name + fairness)
    sample0 = train_ds[0]["x"]  # (4,T,L)
    _, T_in, L_in = sample0.shape

    model = build_model(args.model, num_classes=num_classes, in_channels=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- load scheduler (optional) ---
    scheduler = None
    if hasattr(args, "scheduler"):
        if args.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
            )
        elif args.scheduler and args.scheduler != "none":
            # fallback scheduler names can be added here
            pass

    # W&B identity
    project = args.wandb_project or model_to_project(args.model)

    group = f"split{split_seed}_seed{train_seed}_T{T_in}"
    tags = ["ablation", "model", "landmark", args.model, args.variant]

    auto_name = f"{args.variant}_{args.model}_lr{args.lr:.0e}_bs{args.batch_size}_seed{train_seed}_T{T_in}_L{L_in}"
    run_name = args.run_name or auto_name

    wandb_run = None
    if (not args.no_wandb) and wandb_is_available():
        config = {
            "variant": args.variant,
            "model": args.model,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "patience": args.patience,
            "num_classes": num_classes,
            "train_seed": train_seed,
            "split_seed": split_seed,
            "T": T_in,
            "L": L_in,
        }
        wandb_run = wandb_init(
            project=project,
            config=config,
            name=run_name,
            group=group,
            tags=tags,
            job_type="ablation_model_with_full_landmark",
            notes=(
                "Ablation study using full landmark input with a fixed training setup across models. "
                "Weight decay is set to 1e-3 with dropout rate 0.5, and early stopping is enabled "
                "with patience of 5 epochs. This configuration serves as a baseline setup "
                "for fair model comparison prior to hyperparameter tuning."
            )
        )
        print("wandb_run:", wandb_run)

    # checkpoints
    ckpt_dir = ensure_dir(Path("checkpoints") / args.model / args.variant)
    ckpt_path = ckpt_dir / f"{run_name}__best.pt"

    best_val_loss = float("inf")
    best_epoch = -1
    wait = 0

    # training loop
    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, device)
        val_stats = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_stats['loss']:.4f}, train_acc={train_stats['acc']:.4f} | "
            f"val_loss={val_stats['loss']:.4f}, val_acc={val_stats['acc']:.4f}"
        )

        # log per-epoch (ini yang bikin grafik muncul di W&B)
        if wandb_run is not None:
            wandb_log({
                "epoch": epoch,
                "train/loss": train_stats["loss"],
                "train/acc": train_stats["acc"],
                "val/loss": val_stats["loss"],
                "val/acc": val_stats["acc"],
                "early_stop/wait": wait,
                "early_stop/patience": args.patience,
            }, step=epoch)

        # best checkpoint
        improved = val_stats["loss"] < best_val_loss - 1e-8
        if improved:
            best_val_loss = float(val_stats["loss"])
            best_epoch = epoch
            wait = 0

            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": vars(args),
                "label2idx": label2idx,
                "idx2label": idx2label,
                "train_seed": train_seed,
                "split_seed": split_seed,
                "T": T_in,
                "L": L_in,
            }, ckpt_path)

            print(f"  🔥 New best val_loss={best_val_loss:.4f} | checkpoint saved to {ckpt_path}")

            if wandb_run is not None:
                wandb_log({
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "early_stop/best_epoch": best_epoch,
                }, step=epoch)
        else:
            wait += 1

        # scheduler step (after val)
        if scheduler is not None:
            if getattr(args, "scheduler", None) == "plateau":
                scheduler.step(val_stats["loss"])
            else:
                scheduler.step()

        # early stopping
        if wait >= args.patience:
            print(f"⏹️ Early stopping at epoch {epoch} (best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f})")
            if wandb_run is not None:
                wandb_log({
                    "early_stop/triggered": 1,
                    "early_stop/stop_epoch": epoch,
                }, step=epoch)
            break

    # load best & test
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    test_stats = evaluate_test_full(model, test_loader, device, num_classes)
    print(
        f"\n✅ BEST CHECKPOINT TEST | test_loss={test_stats['loss']:.4f}, test_acc={test_stats['acc']:.4f}\n"
        f"   test_bal_acc={test_stats['balanced_acc']:.4f} | f1_macro={test_stats['f1_macro']:.4f} | f1_weighted={test_stats['f1_weighted']:.4f}"
    )

    # confusion matrix image
    cm = test_stats["cm"]
    fig = plot_confusion_matrix_image(cm, idx2label, title=f"CM Test | {args.model} | {args.variant}")

    if wandb_run is not None:
        # log scalar metrics
        wandb_log({
            "test_loss": test_stats["loss"],
            "test_acc": test_stats["acc"],
            "test_balanced_acc": test_stats["balanced_acc"],
            "test_f1_macro": test_stats["f1_macro"],
            "test_f1_weighted": test_stats["f1_weighted"],
        })

        # log image
        wandb_log_image("test/confusion_matrix", fig, step=best_epoch)

        # summary biar gampang bikin tabel
        wandb_set_summary({
            "variant": args.variant,
            "T": T_in,
            "L": L_in,
            "train_seed": train_seed,
            "split_seed": split_seed,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "test_loss": test_stats["loss"],
            "test_acc": test_stats["acc"],
            "test_balanced_acc": test_stats["balanced_acc"],
            "test_f1_macro": test_stats["f1_macro"],
            "test_f1_weighted": test_stats["f1_weighted"],
        })

        wandb_finish()


if __name__ == "__main__":
    main()
