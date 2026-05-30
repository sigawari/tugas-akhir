import yaml
import logging
import random
import numpy as np
import torch
from pathlib import Path
import wandb


def get_root_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def load_config(config_name: str = "experiment.yaml") -> dict:
    root = get_root_dir()
    config_path = root / "configs" / config_name
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["root_dir"] = root
    return config


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_path(path_str: str) -> Path:
    return get_root_dir() / path_str


# ── W&B helpers ──────────────────────────────────────────────────────────────

def init_wandb(cfg: dict, model_name: str, output_label: str, use_delta: bool):
    """Inisialisasi W&B run. Kalau wandb tidak terinstall, skip tanpa error.

    Parameters
    ----------
    cfg          : config dict dari load_config("train.yaml")
    model_name   : nama arsitektur, e.g. "resnet18"
    output_label : label run, e.g. "resnet18_delta"
    use_delta    : apakah pakai fitur delta

    Returns
    -------
    run object atau None kalau wandb tidak tersedia
    """
    try:
        import wandb
    except ImportError:
        logging.getLogger("utils").warning(
            "wandb tidak terinstall — skip logging. "
            "Install dengan: pip install wandb"
        )
        return None

    run = wandb.init(
        project="slr-try-newpipeline",
        name=output_label,
        config={
            # Model
            "model":         model_name,
            "input_channels": cfg["model"]["input_channels"],
            "num_classes":    cfg["model"]["num_classes"],
            "dropout":        cfg["model"].get("dropout", 0.5),
            "use_delta":      use_delta,
            # Training
            "epochs":         cfg["training"]["epochs"],
            "batch_size":     cfg["training"]["batch_size"],
            "lr":             cfg["training"]["learning_rate"],
            "weight_decay":   cfg["training"]["weight_decay"],
            "patience":       cfg["training"]["early_stopping_patience"],
            "optimizer":      cfg["training"].get("optimizer", "adamw"),
            "scheduler":      cfg["training"].get("scheduler", "cosine"),
            "seed":           cfg["training"]["seed"],
            # Split
            "n_total":        cfg["split"]["n_total"],
            "n_val":          cfg["split"]["n_val"],
            # Augmentation
            "aug_speed_min":  cfg.get("augmentation", {}).get(
                                  "temporal_resample", {}).get("speed_min", 0.85),
            "aug_speed_max":  cfg.get("augmentation", {}).get(
                                  "temporal_resample", {}).get("speed_max", 1.15),
            "aug_jitter_std": cfg.get("augmentation", {}).get(
                                  "jitter", {}).get("std", 0.01),
            "aug_mask_prob":  cfg.get("augmentation", {}).get(
                                  "mask", {}).get("prob", 0.1),
        },
        reinit=True,
    )
    return run


def log_epoch_wandb(run, epoch: int, tr_loss: float, tr_acc: float,
                    val_loss: float, val_acc: float, val_f1: float,
                    lr: float = None):
    """Log satu epoch ke W&B."""
    if run is None:
        return
    payload = {
        "epoch":      epoch,
        "train/loss": tr_loss,
        "train/acc":  tr_acc,
        "val/loss":   val_loss,
        "val/acc":    val_acc,
        "val/f1":     val_f1,
    }
    if lr is not None:
        payload["train/lr"] = lr
    run.log(payload)


def finish_wandb(run, final_acc: float, final_f1: float,
                 best_epoch: int, cm_labels: list, cm: list):
    """Log hasil akhir dan tutup run W&B."""
    if run is None:
        return
    import wandb
    run.summary["final/val_acc"]    = final_acc
    run.summary["final/val_f1"]     = final_f1
    run.summary["final/best_epoch"] = best_epoch

    # Confusion matrix sebagai W&B artifact
    run.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=[row_i for row_i, row in enumerate(cm) for _ in row for _ in [None] * row[_] if row[_] > 0],
            preds=None,
            class_names=cm_labels,
        )
    }) if False else None   # placeholder — lihat log_confusion_matrix_wandb

    run.finish()


def log_confusion_matrix_wandb(run, all_preds: list, all_labels: list,
                                class_names: list):
    """Log confusion matrix yang benar ke W&B."""
    if run is None:
        return                    # ← ini benar, keluar kalau no wandb
    run.log({                     # ← ini harus di luar if, tapi sekarang ter-indent salah
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_labels,
            preds=all_preds,
            class_names=class_names,
        )
    })