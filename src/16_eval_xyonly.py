import subprocess, sys
from pathlib import Path

MODELS = ["cnn2d", "resnet18", "resnet34", "resnet50"]
CKPTS_ROOT = Path("checkpoints")
SPLITS_DIR = Path("../dataset/splits/kfold")
REPORT_DIR = Path("reports")

print("🚀 Evaluasi k-fold untuk xy (use_delta=0)...\n")
for model in MODELS:
    ckpt_dir = CKPTS_ROOT / model / "splits"
    if not ckpt_dir.exists(): continue
    
    for fold in range(1, 6):
        split_path = SPLITS_DIR / f"split_fold_{fold}.json"
        if not split_path.exists(): continue
        
        ckpt_candidates = list(ckpt_dir.glob(f"{model}_*fold{fold}__best.pt"))
        if not ckpt_candidates: continue
        
        cmd = [
            sys.executable, "12_eval.py",
            "--model", model,
            "--ckpt_path", str(ckpt_candidates[0]),
            "--split_path", str(split_path),
            "--use_delta", "0",
            "--report_dir", "reports"
        ]
        print(f"🔄 {model} | Fold {fold} | xy")
        subprocess.run(cmd, check=False)

print("\n✅ Evaluasi xy selesai. File metrics_*.json baru tersimpan di reports/")