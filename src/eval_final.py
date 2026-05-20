#!/usr/bin/env python3
"""
Evaluasi K-Fold + Agregasi Otomatis (All-in-One)
Menggabungkan logika 16_eval_kfold.py dan 15_aggregate_kfold.py
"""
import subprocess, sys, json, re, os
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# ==========================
# 1. KONFIGURASI
# ==========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR     = Path(__file__).resolve().parent
CKPTS_ROOT   = BASE_DIR / "checkpoints"
SPLIT_DIR    = PROJECT_ROOT / "dataset" / "splits" / "kfold"
REPORT_DIR   = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PYTHON_EXEC  = sys.executable

MODELS = ["cnn2d", "resnet18", "resnet34", "resnet50"]
FOLDS  = range(1, 6)

# ==========================
# 2. LOOP EVALUASI
# ==========================
print("🚀 Memulai evaluasi k-fold otomatis...\n")
for model in MODELS:
    for fold in FOLDS:
        split_path = SPLIT_DIR / f"split_fold_{fold}.json"
        if not split_path.exists():
            print(f"⚠️  Skip fold {fold}: {split_path.name} tidak ditemukan.")
            continue

        # Cari checkpoint di folder delta0 & delta1
        found = False
        for delta_dir in [f"delta{d}" for d in [0, 1]]:
            ckpt_dir = CKPTS_ROOT / model / delta_dir / "splits"
            ckpt_candidates = list(ckpt_dir.glob(f"{model}_delta{delta_dir[-1]}*_fold{fold}__best.pt"))
            if not ckpt_candidates:
                continue

            ckpt_path = ckpt_candidates[0]
            cmd = [
                PYTHON_EXEC, str(BASE_DIR / "12_eval.py"),
                "--model", model,
                "--ckpt_path", str(ckpt_path),
                "--split_path", str(split_path),
                "--report_dir", "reports"
            ]
            print(f"🔄 Evaluating: {model} | Fold {fold} | CKPT: {ckpt_path.name}")
            result = subprocess.run(cmd, check=False, cwd=BASE_DIR)
            if result.returncode == 0:
                print(f"✅ Selesai: {model} fold {fold}\n")
                found = True
            else:
                print(f"❌ Gagal: {model} fold {fold}\n")
            break # Hanya evaluasi 1 checkpoint per fold/model
        if not found:
            print(f"⚠️  Tidak ada checkpoint untuk {model} fold {fold}\n")

print("📦 Evaluasi selesai. Memuat metrik untuk agregasi...\n")

# ==========================
# 3. AGREGASI (Adaptasi dari 15_aggregate_kfold.py)
# ==========================
def load_all_metrics():
    metrics = []
    for json_file in REPORT_DIR.glob("metrics_*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        match = re.search(r'(cnn2d|resnet18|resnet34|resnet50)', json_file.stem)
        model = match.group(1) if match else "unknown"
        fold_match = re.search(r'fold(\d+)', json_file.stem)
        fold = int(fold_match.group(1)) if fold_match else 0
        features = "xy" if data.get("use_delta") is False else "xy_dxdy"
        metrics.append({
            "config": {"model": model, "features": features, "fold": fold},
            "accuracy": data["accuracy"], "f1_macro": data["f1_macro"],
            "per_class": data["per_class"], "confusion_matrix": np.array(data["confusion_matrix"])
        })
    return metrics

def aggregate_by_config(metrics_list):
    grouped = defaultdict(list)
    for m in metrics_list:
        key = (m["config"]["model"], m["config"]["features"])
        grouped[key].append(m)
    results = {}
    for (model, features), folds in grouped.items():
        if len(folds) < 3:
            print(f"⚠️  {model}/{features}: hanya {len(folds)} fold, butuh minimal 3! (Skip)")
            continue
        acc_vals = [f["accuracy"] for f in folds]
        f1_vals = [f["f1_macro"] for f in folds]
        results[(model, features)] = {
            "accuracy_mean": np.mean(acc_vals), "accuracy_std": np.std(acc_vals),
            "f1_mean": np.mean(f1_vals), "f1_std": np.std(f1_vals),
            "folds": folds
        }
    return results

metrics_list = load_all_metrics()
print(f"✅ Ditemukan {len(metrics_list)} file metrik")
results = aggregate_by_config(metrics_list)
print(f"✅ {len(results)} konfigurasi siap dianalisis")

# Tabel LaTeX
latex_rows = []
for (m, f), s in sorted(results.items()):
    acc = f"{s['accuracy_mean']*100:.2f} $\\pm$ {s['accuracy_std']*100:.2f}"
    f1  = f"{s['f1_mean']*100:.2f} $\\pm$ {s['f1_std']*100:.2f}"
    latex_rows.append(f"{m} & {f} & {acc} & {f1} \\\\ \\hline")
latex_code = r"""\begin{table}[H]
  \centering
  \caption{Hasil evaluasi K-Fold Cross-Validation (Mean $\pm$ Std Dev)}
  \label{tab:kfold-results}
  \begin{tabular}{|l|l|c|c|}
  \hline
  \textbf{Model} & \textbf{Fitur} & \textbf{Akurasi (\%)} & \textbf{F1-Score (\%)} \\ \hline
""" + "\n".join(latex_rows) + r"""
  \end{tabular}
\end{table}"""
out_dir = REPORT_DIR / "aggregated"
out_dir.mkdir(exist_ok=True)
(out_dir / "table_kfold_results.tex").write_text(latex_code, encoding="utf-8")

# Plot Boxplot
import seaborn as sns
df = pd.DataFrame([{
    "config": f"{m['config']['model']} ({m['config']['features']})",
    "accuracy": m["accuracy"] * 100
} for m in metrics_list])
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x="config", y="accuracy", hue="config", palette="Set2", legend=False)
plt.ylabel("Akurasi (%)"); plt.title("Distribusi Akurasi per Lipatan (K-Fold)"); plt.xticks(rotation=45)
plt.tight_layout(); plt.savefig(out_dir / "boxplot_accuracy_by_fold.png", dpi=300)

print(f"\n📊 RINGKASAN:")
print(f"{'Model':<12} {'Fitur':<10} {'Akurasi (%)':<15} {'F1-Score (%)'}")
print("-"*55)
for (m, f), s in sorted(results.items()):
    print(f"{m:<12} {f:<10} {s['accuracy_mean']*100:.2f} ± {s['accuracy_std']*100:.2f:<12} {s['f1_mean']*100:.2f} ± {s['f1_std']*100:.2f}")
print(f"\n✅ Output tersimpan di: {out_dir}")