# run_eval_kfold.py
import subprocess
import sys
from pathlib import Path

MODELS = ["cnn2d", "resnet18", "resnet34", "resnet50"]

# splits naik 1 parent dari folder script
SPLITS_DIR = Path(__file__).resolve().parent.parent / "dataset" / "splits" / "kfold"

# checkpoints tidak perlu naik parent (relative ke cwd saat menjalankan script)
CKPTS_ROOT = Path("checkpoints")

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("🚀 Mulai evaluasi k-fold otomatis...\n")

for model in MODELS:
    ckpt_dir = CKPTS_ROOT / model / "splits"
    if not ckpt_dir.exists():
        print(f"⚠️  Folder checkpoint tidak ditemukan: {ckpt_dir}")
        continue

    for fold in range(1, 6):
        # Cari checkpoint fold ini
        ckpt_candidates = list(ckpt_dir.glob(f"{model}_*fold{fold}__best.pt"))
        if not ckpt_candidates:
            print(f"⚠️  Checkpoint tidak ditemukan: {model} fold {fold}")
            continue

        ckpt_path = ckpt_candidates[0]
        split_path = SPLITS_DIR / f"split_fold_{fold}.json"

        if not split_path.exists():
            print(f"⚠️  Split JSON tidak ditemukan: {split_path} (Lewati fold {fold})")
            continue

        print(f"🔄 Evaluating: {model} | Fold {fold} | CKPT: {ckpt_path.name}")
        
        cmd = [
            sys.executable, "12_eval.py",
            "--model", model,
            "--ckpt_path", str(ckpt_path),
            "--split_path", str(split_path),
            "--use_delta", "1",
            "--report_dir", "reports"
        ]
        
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"❌ Gagal mengevaluasi {model} fold {fold}. Cek log di atas.")
        else:
            print(f"✅ Selesai: {model} fold {fold}\n")

print("🏁 Evaluasi k-fold selesai. Cek folder `reports/` untuk metrics_*.json")