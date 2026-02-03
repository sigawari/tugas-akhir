# run_ablation_landmark.py
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import argparse
from datetime import datetime

PY = sys.executable  # pakai python dari venv aktif
TRAIN = str(Path(__file__).parent / "train.py")

VARIANTS = ["pose", "hands", "noface", "full"]
MODELS = ["cnn2d", "resnet18", "resnet34", "resnet50"]

EPOCHS = 50
LR = 1e-4
BS = 16
PATIENCE = 10

DEFAULT_PROJECT = "ablation-landmark"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split_path", type=str, default=None, help="Path ke split.json (disarankan untuk reproducibility)")
    p.add_argument("--wandb_project", type=str, default=DEFAULT_PROJECT)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--batch_size", type=int, default=BS)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--scheduler", type=str, default="plateau", choices=["none", "plateau", "cosine"])
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--timestamp", action="store_true", help="Tambahkan timestamp ke run_name")
    return p.parse_args()


def _fmt_sci(x: float) -> str:
    return f"{x:.0e}".replace("+", "")


def build_cmd(args: argparse.Namespace, model: str, variant: str) -> list[str]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S") if args.timestamp else None
    lr_s = _fmt_sci(args.lr)
    wd_s = _fmt_sci(args.weight_decay)

    run_name = f"{variant}_{model}_lr{lr_s}_bs{args.batch_size}_wd{wd_s}_sch{args.scheduler}_p{args.patience}_e{args.epochs}"
    if ts:
        run_name += f"_{ts}"

    cmd = [
        PY, TRAIN,
        "--model", model,
        "--variant", variant,
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--batch_size", str(args.batch_size),
        "--patience", str(args.patience),
        "--weight_decay", str(args.weight_decay),
        "--scheduler", args.scheduler,
        "--wandb_project", args.wandb_project,
        "--run_name", run_name,
    ]

    if args.no_wandb:
        cmd.append("--no_wandb")

    if args.split_path:
        cmd += ["--split_path", args.split_path]

    return cmd


def main() -> None:
    args = parse_args()

    if args.split_path is not None:
        sp = Path(args.split_path)
        if not sp.is_file():
            raise FileNotFoundError(f"split_path tidak ditemukan: {sp}")
        print(f"[INFO] Using split_path: {sp}")

    for model in MODELS:
        for variant in VARIANTS:
            cmd = build_cmd(args, model=model, variant=variant)
            print("\n=== RUN:", " ".join(cmd), "===\n")
            if args.dry_run:
                continue
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
