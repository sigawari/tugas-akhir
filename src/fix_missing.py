# 17_fix_missing_folds.py
import subprocess, sys, re
from pathlib import Path

BASE_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
CKPTS_ROOT   = BASE_DIR / "checkpoints"
SPLIT_DIR    = PROJECT_ROOT / "dataset" / "splits" / "kfold"
REPORT_DIR   = PROJECT_ROOT / "reports"

MODELS = ["cnn2d", "resnet18", "resnet34", "resnet50"]
FOLDS  = range(1, 6)
FEATS  = ["xy", "xy_dxdy"]

print("🔍 Memindai file metrics yang hilang...")
missing = []

for model in MODELS:
    for feat in FEATS:
        for fold in FOLDS:
            # Cek apakah JSON sudah ada
            pattern = f"metrics_{model}_{feat}_*fold{fold}*.json"
            if not list(REPORT_DIR.glob(pattern)):
                # Cari checkpoint yang sesuai
                ckpt_dir = CKPTS_ROOT / model / "splits"
                ckpt_pattern = f"{model}_{feat}_*fold{fold}__best.pt"
                ckpts = list(ckpt_dir.glob(ckpt_pattern))
                split_path = SPLIT_DIR / f"split_fold_{fold}.json"
                
                if ckpts and split_path.exists():
                    missing.append({
                        "model": model, "feat": feat, "fold": fold,
                        "ckpt": ckpts[0], "split": split_path
                    })

if not missing:
    print("✅ Semua 40 file metrics lengkap!")
    sys.exit(0)

print(f"⚠️  Ditemukan {len(missing)} file metrics hilang. Mengevaluasi ulang...\n")
for m in missing:
    print(f"🔄 {m['model']:8} | {m['feat']:8} | Fold {m['fold']}")
    cmd = [
        sys.executable, str(BASE_DIR / "12_eval.py"),
        "--model", m["model"],
        "--ckpt_path", str(m["ckpt"]),
        "--split_path", str(m["split"]),
        "--report_dir", str(REPORT_DIR)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE_DIR)
    if result.returncode == 0:
        print("✅ Berhasil\n")
    else:
        print(f"❌ Gagal. Error:\n{result.stderr[-300:]}\n")

print("🎉 Perbaikan selesai. Jalankan kembali: python 15_eval_kfold.py")