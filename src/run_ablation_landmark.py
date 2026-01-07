# run_ablation_landmark.py
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PY = sys.executable  # pakai python dari venv aktif
TRAIN = str(Path(__file__).parent / "train.py")

VARIANTS = ["pose", "hands", "noface", "full"]
MODELS = ["cnn2d", "resnet18", "resnet34", "resnet50"]

EPOCHS = 50
LR = 1e-4
BS = 16
PATIENCE = 10

def main() -> None:
    for model in MODELS:
        for variant in VARIANTS:
            cmd = [
                PY, TRAIN,
                "--model", model,
                "--variant", variant,
                "--epochs", str(EPOCHS),
                "--lr", str(LR),
                "--batch_size", str(BS),
                "--patience", str(PATIENCE),
            ]

            print("\n=== RUN:", " ".join(cmd), "===\n")
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
