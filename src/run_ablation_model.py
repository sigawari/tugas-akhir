# run_ablation_model_full.py
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PY = sys.executable
TRAIN = str(Path(__file__).parent / "train.py")

MODELS = ["cnn2d", "resnet18", "resnet34", "resnet50"]

# Fixed setup untuk “Ablation Model (Full landmark)”
VARIANT = "full"
PROJECT = "ablation-model"
EPOCHS = 50
LR = 1e-4
WD = 1e-3
BS = 16
PATIENCE = 5

def main() -> None:
    for model in MODELS:
        run_name = f"full_{model}_lr{LR:.0e}_bs{BS}_wde-3_T30"  
        cmd = [
            PY, TRAIN,
            "--variant", VARIANT,
            "--model", model,
            "--epochs", str(EPOCHS),
            "--lr", str(LR),
            "--batch_size", str(BS),
            "--patience", str(PATIENCE),
            "--wandb_project", PROJECT,
            "--run_name", run_name,
            "--weight_decay", str(WD),
            "--scheduler", "plateau",
        ]

        print("\n=== RUN:", " ".join(cmd), "===\n")
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
