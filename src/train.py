# train_resnet.py
# Main training script (Colab-ready).
# Loads config YAML, prepares dataset, initializes model & Trainer.
# Run: python train_resnet.py --config configs/full.yaml

# train.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any

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
)
from models import ResNet2DSign
from metrics import accuracy_from_logits


# Dataset
class SignDataset(Dataset):
    """Dataset untuk sign language berbasis NPY + split index.

    Menghasilkan tensor bentuk (C=4, H=T, W=L) untuk ResNet2D.
    """

    def __init__(self, items: List[Dict[str, Any]], variant: str) -> None:
        super().__init__()
        self.items = items
        self.variant = variant

        # Cache X.npy per kata
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

        # ---- reshape ke (C=4, H=T, W=L) ----
        T, D = seq.shape
        assert D % 4 == 0, f"D={D} tidak habis dibagi 4, cek build_dataset.py"
        L = D // 4

        seq = seq.reshape(T, L, 4)          # (T, L, 4)
        seq = np.transpose(seq, (2, 0, 1))  # (4, T, L)

        x_tensor = torch.from_numpy(seq).float()       # (4, T, L)
        y_tensor = torch.tensor(label, dtype=torch.long)

        return {
            "x": x_tensor,
            "y": y_tensor,
            "word": word,
            "index": seq_idx,
        }

# Training loop------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_count = 0

    criterion = nn.CrossEntropyLoss()

    for batch in loader:
        x = batch["x"].to(device)     # (B, T, D)
        y = batch["y"].to(device)     # (B,)

        optimizer.zero_grad()
        logits = model(x)             # (B, num_classes)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        acc = accuracy_from_logits(logits.detach(), y.detach())

        total_loss += loss.item() * batch_size
        total_acc += acc * batch_size
        total_count += batch_size

    avg_loss = total_loss / total_count
    avg_acc = total_acc / total_count

    return {"loss": avg_loss, "acc": avg_acc}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_count = 0

    criterion = nn.CrossEntropyLoss()

    for batch in loader:
        x = batch["x"].to(device)    # (B, T, D)
        y = batch["y"].to(device)

        logits = model(x)
        loss = criterion(logits, y)

        batch_size = y.size(0)
        acc = accuracy_from_logits(logits, y)

        total_loss += loss.item() * batch_size
        total_acc += acc * batch_size
        total_count += batch_size

    avg_loss = total_loss / total_count
    avg_acc = total_acc / total_count

    return {"loss": avg_loss, "acc": avg_acc}


# Main script--------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--variant", type=str, default="pose",
                        choices=["pose", "full", "noface", "hands"],
                        help="Variant fitur yang digunakan.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--wandb_project", type=str, default="sign-resnet18")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--split_path", type=str,
                        default=str(DATA_DIR / "splits" / "split.json"))
    parser.add_argument("--no_wandb", action="store_true",
                        help="Matikan logging ke Weights & Biases.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Seed & device
    seed(DEFAULT_SEED)
    device = get_device(prefer_gpu=True)
    print(f"Device: {device}")

    # Load split.json
    split_path = Path(args.split_path)
    split_data = read_json(split_path)
    label2idx = split_data["label2idx"]
    num_classes = len(label2idx)

    train_items = split_data["splits"]["train"]
    val_items = split_data["splits"]["val"]
    test_items = split_data["splits"]["test"]

    print(f"Total train: {len(train_items)}, val: {len(val_items)}, test: {len(test_items)}")

    # Dataset & DataLoader
    train_ds = SignDataset(train_items, variant=args.variant)
    val_ds = SignDataset(val_items, variant=args.variant)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # Model
    model = ResNet2DSign(num_classes=num_classes, in_channels=4)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Wandb init
    wandb_run = None
    if not args.no_wandb:
        config = {
            "variant": args.variant,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "num_classes": num_classes,
            "seed": DEFAULT_SEED,
            "model": "ResNet2DSign",
        }
        wandb_run = wandb_init(
            project=args.wandb_project,
            config=config,
            name=args.run_name,
        )

    best_val_acc = 0.0
    ckpt_dir = ensure_dir(Path("checkpoints"))
    ckpt_path = ckpt_dir / f"resnet18_{args.variant}_best.pt"

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_stats = evaluate(model, val_loader, device)

        msg = (
            f"Epoch {epoch:03d} | "
            f"train_loss={train_stats['loss']:.4f}, train_acc={train_stats['acc']:.4f} | "
            f"val_loss={val_stats['loss']:.4f}, val_acc={val_stats['acc']:.4f}"
        )
        print(msg)

        # Log ke wandb
        if wandb_run is not None:
            wandb_log({
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_acc": train_stats["acc"],
                "val_loss": val_stats["loss"],
                "val_acc": val_stats["acc"],
            }, step=epoch)

        # Save best model
        if val_stats["acc"] > best_val_acc:
            best_val_acc = val_stats["acc"]
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": best_val_acc,
                "config": vars(args),
                "label2idx": label2idx,
            }, ckpt_path)
            print(f"  🔥 New best val_acc={best_val_acc:.4f} | checkpoint saved to {ckpt_path}")

    if wandb_run is not None:
        wandb_finish()


if __name__ == "__main__":
    main()
