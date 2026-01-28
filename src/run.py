# run_manual.py
from __future__ import annotations

import os
import sys
import subprocess
from datetime import datetime

# =========================
# PATHS
# =========================
PY = sys.executable  # python dari venv aktif
HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "train.py")

# =========================
# CONFIG 
# =========================
PROJECT = "ablation-hyperparameter"

# Toggle
DRY_RUN = False             
CONTINUE_ON_ERROR = False    # True: kalau 1 run gagal, lanjut run berikutnya

SPLIT_PATH = None

# Daftar eksperimen 
RUNS = [
    {
        "model": "resnet34",     # opsi: cnn2d | resnet18 | resnet34 | resnet50
        "variant": "full",       # opsi: pose | hands | noface | full
        "epochs": 50,
        "lr": 1e-5,
        "batch_size": 16,
        "patience": 5,           # early stopping patience
        "weight_decay": 1e-4,
        "scheduler": "plateau",  # opsi: none | plateau | cosine
        "no_wandb": False,       
        "run_name": "resnet34_1e5",
    },

    # kalau mau bandingin 2 model dengan setup sama
    # {
    #     "model": "cnn2d",
    #     "variant": "full",
    #     "epochs": 50,
    #     "lr": 1e-4,
    #     "batch_size": 16,
    #     "patience": 5,
    #     "weight_decay": 1e-3,
    #     "scheduler": "plateau",
    #     "no_wandb": False,
    # },
]


def _fmt_sci(x: float) -> str:
    # 1e-4 -> "1e-04"
    return f"{x:.0e}".replace("+", "")


def auto_run_name(cfg: dict) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lr = _fmt_sci(cfg["lr"])
    wd = _fmt_sci(cfg["weight_decay"])
    return (
        f"{cfg['variant']}_{cfg['model']}"
        f"_lr{lr}_bs{cfg['batch_size']}"
        f"_wd{wd}_sch{cfg['scheduler']}"
        f"_p{cfg['patience']}_e{cfg['epochs']}"
        f"_{ts}"
    )


def build_cmd(cfg: dict) -> list[str]:
    run_name = cfg.get("run_name") or auto_run_name(cfg)

    cmd = [
        PY, TRAIN,
        "--variant", cfg["variant"],
        "--model", cfg["model"],
        "--epochs", str(cfg["epochs"]),
        "--lr", str(cfg["lr"]),
        "--batch_size", str(cfg["batch_size"]),
        "--patience", str(cfg["patience"]),
        "--wandb_project", PROJECT,
        "--run_name", run_name,
        "--weight_decay", str(cfg["weight_decay"]),
        "--scheduler", cfg["scheduler"],
    ]

    if cfg.get("no_wandb", False):
        cmd.append("--no_wandb")

    # Split path optional (kalau kamu set SPLIT_PATH)
    if SPLIT_PATH:
        cmd += ["--split_path", str(SPLIT_PATH)]

    return cmd


def main() -> None:
    failures = 0

    for i, cfg in enumerate(RUNS, start=1):
        cmd = build_cmd(cfg)

        print(f"\n=== RUN {i}/{len(RUNS)} ===")
        print(" ".join(cmd))

        if DRY_RUN:
            continue

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            failures += 1
            print(f"[ERROR] run gagal (exit={e.returncode}) -> model={cfg['model']} variant={cfg['variant']}")
            if not CONTINUE_ON_ERROR:
                raise

    if failures:
        print(f"\nSelesai dengan {failures} run gagal.")
    else:
        print("\nSelesai, semua run sukses.")


if __name__ == "__main__":
    main()
