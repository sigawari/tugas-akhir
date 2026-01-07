import subprocess
import sys
from pathlib import Path

VARIANTS = ["pose", "hands", "noface", "full"]

def main():
    # asumsi script ini dijalankan dari folder src/
    script_dir = Path(__file__).resolve().parent
    train_py = script_dir / "train.py"

    if not train_py.exists():
        raise FileNotFoundError(f"train.py tidak ditemukan di: {train_py}")

    # samakan hyperparameter untuk fairness
    epochs = 50
    lr = 1e-4
    bs = 16

    for v in VARIANTS:
        cmd = [
            sys.executable, str(train_py),
            "--variant", v,
            "--epochs", str(epochs),
            "--lr", str(lr),
            "--batch_size", str(bs),
            "--wandb_project", "ablation-landmark",
        ]
        print("\n=== RUN:", " ".join(cmd), "===")
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
