# scripts/run_all_experiments.py
"""
Jalankan semua 6 eksperimen ablasi secara otomatis.

Cara pakai:
    python scripts/run_all_experiments.py
    python scripts/run_all_experiments.py --dry-run   # cek konfigurasi saja
    python scripts/run_all_experiments.py --id 1 3 5  # jalankan id tertentu saja

Output per eksperimen:
    outputs/checkpoints/<label>_best.pt
    outputs/logs/train_<label>.json
    outputs/plots/loss_<label>.png
    outputs/logs/experiment_summary.json  ← ringkasan semua eksperimen
"""

import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.utils import load_config, get_logger

logger = get_logger("RunAll")


def parse_args():
    p = argparse.ArgumentParser(description="Run semua eksperimen ablasi")
    p.add_argument("--dry-run", action="store_true",
                   help="Tampilkan konfigurasi tanpa menjalankan training")
    p.add_argument("--id", nargs="+", type=int, default=None,
                   help="Jalankan hanya eksperimen dengan ID tertentu (contoh: --id 1 3)")
    p.add_argument("--config-train",      default="train.yaml")
    p.add_argument("--config-experiment", default="experiment.yaml")
    return p.parse_args()


def main():
    args    = parse_args()
    cfg_exp = load_config(args.config_experiment)
    experiments = cfg_exp["experiments"]

    # filter by id kalau ada
    if args.id:
        experiments = [e for e in experiments if e["id"] in args.id]
        if not experiments:
            logger.error(f"Tidak ada eksperimen dengan ID {args.id}")
            sys.exit(1)

    logger.info(f"Total eksperimen: {len(experiments)}")
    logger.info("-" * 55)

    summary = []

    for exp in experiments:
        label     = exp["label"]
        model     = exp["model"]
        use_delta = exp["use_delta"]

        logger.info(f"\n[{exp['id']}/6] {label}")
        logger.info(f"  model={model}  use_delta={use_delta}")

        if args.dry_run:
            logger.info("  → dry-run, skip.")
            continue

        start_time = datetime.now()

        # Jalankan run_train.py dengan override argumen
        cmd = [
            sys.executable,
            str(ROOT_DIR / "scripts" / "run_train.py"),
            "--model",     model,
            "--use-delta", str(use_delta),
            "--label",     label,
            "--config",    args.config_train,
        ]

        logger.info(f"  CMD: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=ROOT_DIR)

        elapsed = (datetime.now() - start_time).total_seconds()

        if result.returncode != 0:
            logger.error(f"  ✗ Eksperimen {label} GAGAL (returncode={result.returncode})")
            summary.append({
                "id": exp["id"], "label": label,
                "status": "FAILED", "elapsed_sec": elapsed
            })
            continue

        # Baca hasil dari log yang disimpan run_train.py
        log_path = ROOT_DIR / "outputs" / "logs" / f"train_{label}.json"
        result_data = {}
        if log_path.exists():
            with open(log_path) as f:
                log = json.load(f)
            final = log.get("final", {})
            result_data = {
                "accuracy":   final.get("accuracy"),
                "f1_macro":   final.get("f1_macro"),
                "best_epoch": final.get("best_epoch"),
            }
            logger.info(
                f"  ✓ Selesai — "
                f"Acc={result_data['accuracy']:.4f}  "
                f"F1={result_data['f1_macro']:.4f}  "
                f"BestEpoch={result_data['best_epoch']}  "
                f"({elapsed:.0f}s)"
            )

        summary.append({
            "id":          exp["id"],
            "label":       label,
            "model":       model,
            "use_delta":   use_delta,
            "status":      "OK",
            "elapsed_sec": round(elapsed),
            **result_data,
        })

    # Simpan ringkasan semua eksperimen
    if not args.dry_run and summary:
        summary_path = ROOT_DIR / "outputs" / "logs" / "experiment_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump({
                "run_date":    datetime.now().isoformat(),
                "experiments": summary,
            }, f, indent=2, ensure_ascii=False)

        logger.info("\n" + "=" * 55)
        logger.info("  RINGKASAN EKSPERIMEN")
        logger.info("=" * 55)
        logger.info(f"  {'Label':<25} {'Acc':>6} {'F1':>6} {'Epoch':>6} {'Status'}")
        logger.info("  " + "-" * 53)
        for s in summary:
            acc   = f"{s['accuracy']:.4f}"   if s.get("accuracy")   else "  -   "
            f1    = f"{s['f1_macro']:.4f}"   if s.get("f1_macro")   else "  -   "
            epoch = f"{s['best_epoch']}"     if s.get("best_epoch") else "  -"
            logger.info(f"  {s['label']:<25} {acc:>6} {f1:>6} {epoch:>6}  {s['status']}")
        logger.info("=" * 55)
        logger.info(f"  Ringkasan → {summary_path}")


if __name__ == "__main__":
    main()