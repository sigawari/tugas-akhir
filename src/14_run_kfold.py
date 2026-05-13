import subprocess
import sys
import json
import numpy as np
import os
from pathlib import Path
import wandb

#  Konfigurasi W&B (Fallback aman jika CLI tidak terbaca)
os.environ["WANDB_PROJECT"] = "kfoldtrainingslr"
os.environ["WANDB_RUN_GROUP"] = "kfold_cross_validation"

BASE_DIR = Path(__file__).resolve().parent  # ta-code/src/
PROJECT_ROOT = BASE_DIR.parent              # ta-code/

SPLIT_DIR  = PROJECT_ROOT / "dataset" / "splits" / "kfold"
REPORT_DIR = PROJECT_ROOT / "reports"

FOLDS      = [f"split_fold_{i}.json" for i in range(1, 6)]
MODEL      = "resnet34"
USE_DELTA  = 1
EPOCHS     = 50

CKPT_DIR = BASE_DIR / "checkpoints" / MODEL / "splits"
PYTHON_EXEC = sys.executable

results = []

for fold_json in FOLDS:
    fold_id = fold_json.replace("split_fold_", "").replace(".json", "")
    split_path = SPLIT_DIR / fold_json
    
    if not split_path.is_file():
        print(f"❌ File split tidak ditemukan: {split_path}")
        continue

    print(f"\n{'='*60}")
    print(f"🔄 MEMULAI FOLD {fold_id}")
    print(f"📂 Split Path: {split_path}")
    print(f"{'='*60}")

    # ✅ Gunakan --run_name agar match dengan train.py & nama checkpoint
    fold_run_name = f"fold_{fold_id}"

    # 1️⃣ TRAIN
    cmd_train = [
        PYTHON_EXEC, str(BASE_DIR / "train.py"),
        "--model", MODEL,
        "--use_delta", str(USE_DELTA),
        "--epochs", str(EPOCHS),
        "--split_path", str(split_path),
        "--run_name", fold_run_name,       # ✅ Match train.py
        "--seed", "42",
        "--wandb_project", "kfoldtrainingslr", # ✅ Match train.py (override default)
        # ❌ Hapus --project, --group, --name agar tidak bentrok argparse
    ]
    subprocess.run(cmd_train, check=True, cwd=BASE_DIR)

    # 2️⃣ EVAL
    # train.py menyimpan checkpoint dengan nama: {run_name}__best.pt
    best_ckpt = CKPT_DIR / f"{fold_run_name}__best.pt"
    
    print(f"🔍 Mencari checkpoint di: {best_ckpt}")
    if not best_ckpt.exists():
        print(f"⚠️ Checkpoint tidak ditemukan. Mencari fallback...")
        fallbacks = list(CKPT_DIR.glob(f"*{MODEL}*fold{fold_id}*__best.pt"))
        if fallbacks:
            best_ckpt = max(fallbacks, key=lambda p: p.stat().st_mtime)
            print(f"✅ Ditemukan fallback: {best_ckpt}")
        else:
            print("❌ Tidak ada checkpoint sama sekali. Skip fold ini.")
            continue

    cmd_eval = [
        PYTHON_EXEC, str(BASE_DIR / "12_eval.py"),
        "--model", MODEL,
        "--use_delta", str(USE_DELTA),
        "--split_path", str(split_path),
        "--ckpt_path", str(best_ckpt),
        "--report_dir", str(REPORT_DIR / f"fold_{fold_id}")
    ]
    subprocess.run(cmd_eval, check=True, cwd=BASE_DIR)

    # 3️⃣ BACA HASIL
    metrics_folder = REPORT_DIR / f"fold_{fold_id}"
    metrics_files = list(metrics_folder.glob("metrics_*.json"))
    if not metrics_files:
        print(f"⚠️ File metrics tidak ditemukan di {metrics_folder}")
        continue

    metrics_file = max(metrics_files, key=lambda p: p.stat().st_mtime)
    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    results.append({
        "fold": fold_id,
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "balanced_acc": metrics["balanced_accuracy"]
    })
    print(f"✅ Fold {fold_id} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1_macro']:.4f}")

# 📊 FINAL REPORT & W&B SUMMARY LOGGING
if results:
    accs = np.array([r["accuracy"] for r in results])
    f1s = np.array([r["f1_macro"] for r in results])
    bal_accs = np.array([r["balanced_acc"] for r in results])
    
    print(f"\n{'='*60}")
    print("📈 HASIL STRATIFIED 5-FOLD CROSS-VALIDATION")
    print(f"{'='*60}")
    print(f"Accuracy     : {accs.mean():.4f} ± {accs.std():.4f}")
    print(f"F1-Macro     : {f1s.mean():.4f} ± {f1s.std():.4f}")
    print(f"Balanced Acc : {bal_accs.mean():.4f} ± {bal_accs.std():.4f}")
    print(f"📁 Laporan tersimpan di: {REPORT_DIR}/fold_*/")

    # 🌐 UPLOAD SUMMARY KE W&B (Run terpisah agar tidak ganggu training runs)
    try:
        summary_run = wandb.init(
            project="kfoldtrainingslr",
            group="kfold_cross_validation",
            name="cv_summary_report",
            job_type="cross_validation_summary"
        )
        summary_run.log({
            "cv/accuracy_mean": accs.mean(),
            "cv/accuracy_std": accs.std(),
            "cv/f1_macro_mean": f1s.mean(),
            "cv/f1_macro_std": f1s.std(),
            "cv/balanced_acc_mean": bal_accs.mean(),
            "cv/balanced_acc_std": bal_accs.std(),
            "cv/fold_accuracies": accs.tolist(),
            "cv/fold_f1_scores": f1s.tolist()
        })
        summary_run.finish()
        print("✅ CV Summary berhasil di-upload ke WandB: kfoldtrainingslr")
    except Exception as e:
        print(f"️ Gagal log summary ke WandB: {e}")
        print("💡 Pastikan sudah run `wandb login` di terminal.")
else:
    print("❌ Pipeline gagal. Cek error di atas.")