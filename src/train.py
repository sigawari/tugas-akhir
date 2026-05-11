from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from models import Basic2DCNN, ResNet18, ResNet34, ResNet50
from utils import (
    DEFAULT_SEED,
    seed,
    get_device,
    ensure_dir,
    wandb_init,
    wandb_log,
    wandb_finish,
)
from dataset import create_dataloaders  # file 10_dataset.py


# =========================
# MODEL FACTORY
# =========================

def build_model(model_name: str, num_classes: int, in_channels: int = 4) -> nn.Module:
    model_name = model_name.lower()

    if model_name in ("cnn2d", "basic2dcnn", "cnn"):
        return Basic2DCNN(num_classes=num_classes, in_channels=in_channels)

    # if model_name in ("cnn2d_residual", "residualcnn", "custom_resnet"):
    #     return CNN2DResidual(num_classes=num_classes, in_channels=in_channels)

    if model_name in ("resnet18", "r18"):
        return ResNet18(num_classes=num_classes, in_channels=in_channels)

    if model_name in ("resnet34", "r34"):
        return ResNet34(num_classes=num_classes, in_channels=in_channels)

    if model_name in ("resnet50", "r50"):
        return ResNet50(num_classes=num_classes, in_channels=in_channels)

    raise ValueError(f"Unknown model: {model_name}")


# =========================
# SCHEDULER FACTORY
# =========================

def build_scheduler(optimizer, scheduler_name: str, epochs: int):
    scheduler_name = scheduler_name.lower()

    if scheduler_name == "none":
        return None

    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=5e-6,
        )

    raise ValueError(f"Unknown scheduler: {scheduler_name}")

# =========================
# TRAIN / VAL LOOP
# =========================

def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device | str,
) -> Tuple[float, float]:
    """
    Return:
      avg_loss, avg_acc
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in loader:
        x = batch["x"].to(device)   # (B, 4, T, L)
        y = batch["y"].to(device)   # (B,)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, y)

            if is_train:
                loss.backward()
                optimizer.step()

        preds = torch.argmax(logits, dim=1)
        correct = (preds == y).sum().item()
        bsz = y.size(0)

        total_loss += loss.item() * bsz
        total_correct += correct
        total_count += bsz

    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)
    return avg_loss, avg_acc


# =========================
# EARLY STOPPING
# =========================

class EarlyStopping:
    def __init__(self, patience: int = 5, mode: str = "min") -> None:
        self.patience = patience
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """
        Return True if improved.
        """
        if self.best_score is None:
            self.best_score = score
            return True

        improved = score < self.best_score if self.mode == "min" else score > self.best_score

        if improved:
            self.best_score = score
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True

        return False


# =========================
# CHECKPOINT
# =========================

def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    config: Dict[str, Any],
) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    torch.save(
        {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


# =========================
# ARGPARSE
# =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    #Is there any cnn2d_residual?
    p.add_argument("--model", type=str, default="resnet50",
                   choices=["cnn2d", "resnet18", "resnet34", "resnet50"])

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--scheduler", type=str, default="cosine",
                   choices=["none", "plateau", "cosine"])

    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)

    p.add_argument("--split_path", type=str, default=None)

    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.2)

    p.add_argument("--jitter_prob", type=float, default=0.5)
    p.add_argument("--jitter_std", type=float, default=0.01)
    p.add_argument("--mask_prob", type=float, default=0.5)
    p.add_argument("--mask_ratio", type=float, default=0.05)

    p.add_argument("--use_delta", type=int, default=1)

    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--wandb_project", type=str, default="slr-resnet-setupB-2-half")
    p.add_argument("--no_wandb", action="store_true")


    return p.parse_args()


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
        split_path = Path(__file__).resolve().parent.parent / "dataset" / "splits" / "split_80_20.json"
    print(f"📌 TRAINING MENGGUNAKAN SPLIT: {split_path}")  # <-- TAMBAH INI

    train_loader, val_loader, split_data = create_dataloaders(
        split_path=split_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        train_augment=True,
        jitter_prob=args.jitter_prob,
        jitter_std=args.jitter_std,
        mask_prob=args.mask_prob,
        mask_ratio=args.mask_ratio,
        use_delta=bool(args.use_delta),
    )

    num_classes = split_data["meta"]["num_classes"]

    in_channels = 4 if bool(args.use_delta) else 2

    model = build_model(
        model_name=args.model,
        num_classes=num_classes,
        in_channels=in_channels,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = build_scheduler(optimizer, args.scheduler, args.epochs)
    early_stopper = EarlyStopping(patience=args.patience, mode="min")

    run_name = args.run_name
    if run_name is None:
        feat_tag = "xy_dxdy" if bool(args.use_delta) else "xy"
        run_name = f"{args.model}_{feat_tag}_lr{args.lr}_bs{args.batch_size}_wd{args.weight_decay}_sc{args.scheduler}"

    save_root = Path(args.save_dir) / args.model / "splits"
    ensure_dir(save_root)
    ckpt_best = save_root / f"{run_name}__best.pt"
    ckpt_last = save_root / f"{run_name}__last.pt"

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb_init(
            project=args.wandb_project,
            config=vars(args),
            name=run_name,
        )

    best_val_loss = float("inf")
    best_epoch = -1

    print("\n=== TRAINING START ===")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples  : {len(val_loader.dataset)}")
    print(f"Num classes  : {num_classes}")
    print(f"Checkpoint   : {ckpt_best}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
        )

        # scheduler update
        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        improved = early_stopper.step(val_loss)

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(
                ckpt_best,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                config=vars(args),
            )

        save_checkpoint(
            ckpt_last,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_loss=best_val_loss,
            config=vars(args),
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={current_lr:.6f}"
        )

        if use_wandb:
            wandb_log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "lr": current_lr,
                    "best_val_loss": best_val_loss,
                },
                step=epoch,
            )

        if early_stopper.should_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    print("\n=== TRAINING FINISHED ===")
    print(f"Best epoch    : {best_epoch}")
    print(f"Best val loss : {best_val_loss:.4f}")
    print(f"Best ckpt     : {ckpt_best}")
    print(f"Last ckpt     : {ckpt_last}")

    if use_wandb:
        wandb_log(
            {
                "final/best_epoch": best_epoch,
                "final/best_val_loss": best_val_loss,
            }
        )
        wandb_finish()


if __name__ == "__main__":
    main()