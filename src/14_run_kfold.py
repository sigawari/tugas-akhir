import subprocess
import sys
import os
from pathlib import Path

#  KONFIGURASI GLOBAL (STANDARISASI)
PROJECT_NAME = "slr-kfold-comparison"
USE_DELTA    = 1
EPOCHS       = 50
LR           = 1e-4         
BATCH_SIZE   = 16           
FOLDS        = range(1, 6)   # Fold 1 sampai 5

#  Daftar model yang akan diuji
MODELS = ["cnn2d", "resnet18", "resnet34", "resnet50"]

#  SETUP PATH
BASE_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
SPLIT_DIR    = PROJECT_ROOT / "dataset" / "splits" / "kfold"
REPORT_DIR   = PROJECT_ROOT / "reports"
CKPT_DIR     = BASE_DIR / "checkpoints"
PYTHON_EXEC  = sys.executable

# Set environment variable agar W&B fallback aman
os.environ["WANDB_PROJECT"] = PROJECT_NAME

#  MAIN PIPELINE
for model_name in MODELS:
    print(f"\n{'='*80}")
    print(f" MEMULAI MODEL: {model_name.upper()} | LR: {LR} | BS: {BATCH_SIZE} | Δ: {USE_DELTA}")
    print(f"{'='*80}")

    # Folder checkpoint spesifik per model (sesuai default train.py)
    model_ckpt_dir = CKPT_DIR / model_name / "splits"
    os.makedirs(model_ckpt_dir, exist_ok=True)

    for fold_id in FOLDS:
        split_json = f"split_fold_{fold_id}.json"
        split_path = SPLIT_DIR / split_json

        if not split_path.exists():
            print(f"⚠️  Skip fold {fold_id}: File {split_path} tidak ditemukan.")
            continue

        # 🏷️ NAMA RUN UNIK & INFORMATIF (Kunci anti-bentrok)
        # Contoh: resnet34_lr0.0001_bs16_fold3
        unique_run_name = f"{model_name}_lr{LR}_bs{BATCH_SIZE}_fold{fold_id}"

        print(f"\n   ▶️  Running: {unique_run_name}")

        # 1️⃣ TRAIN
        cmd_train = [
            PYTHON_EXEC, str(BASE_DIR / "train.py"),
            "--model", model_name,
            "--use_delta", str(USE_DELTA),
            "--epochs", str(EPOCHS),
            "--split_path", str(split_path),
            "--run_name", unique_run_name,          
            "--wandb_project", PROJECT_NAME,         
            "--lr", str(LR),                        
            "--batch_size", str(BATCH_SIZE),         
            "--seed", "42",
        ]

        try:
            subprocess.run(cmd_train, check=True, cwd=BASE_DIR)
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Training gagal untuk {unique_run_name}. Lanjut ke fold berikutnya...")
            continue

        # 2️⃣ EVALUASI
        best_ckpt_path = model_ckpt_dir / f"{unique_run_name}__best.pt"
        if not best_ckpt_path.exists():
            print(f"   ⚠️ Checkpoint tidak ditemukan. Skip evaluasi.")
            continue

        cmd_eval = [
            PYTHON_EXEC, str(BASE_DIR / "12_eval.py"),
            "--model", model_name,
            "--use_delta", str(USE_DELTA),
            "--split_path", str(split_path),
            "--ckpt_path", str(best_ckpt_path),
            "--report_dir", str(REPORT_DIR / unique_run_name)
        ]
        subprocess.run(cmd_eval, check=True, cwd=BASE_DIR)
        print(f"   ✅ Selesai: {unique_run_name}")

    print(f"\n💤 Model {model_name.upper()} selesai. Lanjut model berikutnya...\n")

print("🎉 SEMUA EKSPERIMEN SELESAI! Cek dashboard W&B: wandb.ai")