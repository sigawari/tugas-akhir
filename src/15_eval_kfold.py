import subprocess, sys, json, re
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# ==========================
# 1. KONFIGURASI & PATH
# ==========================
BASE_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
CKPTS_ROOT   = BASE_DIR / "checkpoints"
SPLIT_DIR    = PROJECT_ROOT / "dataset" / "splits" / "kfold"
REPORT_DIR   = PROJECT_ROOT / "reports"
OUTPUT_DIR   = REPORT_DIR / "aggregated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PYTHON_EXEC  = sys.executable

MODELS = ["cnn2d", "resnet18", "resnet34", "resnet50"]
FOLDS  = range(1, 6)

# ==========================
# 2. EVALUASI OTOMATIS (FIXED)
# ==========================
def run_evaluation():
    print("Memulai evaluasi k-fold otomatis (xy & xy_dxdy)...\n")
    eval_count = 0
    failed_list = []

    for model in MODELS:
        ckpt_dir = CKPTS_ROOT / model / "splits"
        if not ckpt_dir.exists(): continue

        for fold in FOLDS:
            for feat_tag in ["xy", "xy_dxdy"]:
                pattern = f"{model}_{feat_tag}_*fold{fold}__best.pt"
                ckpt_candidates = list(ckpt_dir.glob(pattern))
                if not ckpt_candidates: continue

                ckpt_path = ckpt_candidates[0]
                split_path = SPLIT_DIR / f"split_fold_{fold}.json"
                if not split_path.exists(): continue

                print(f"🔄 {model:8} | {feat_tag:8} | Fold {fold} ...", end=" ")
                
                cmd = [
                    PYTHON_EXEC, str(BASE_DIR / "12_eval.py"),
                    "--model", model, "--ckpt_path", str(ckpt_path),
                    "--split_path", str(split_path), "--report_dir", str(REPORT_DIR)
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE_DIR)
                if result.returncode == 0:
                    eval_count += 1
                    print("✅")
                else:
                    print("❌")
                    failed_list.append(f"{model}/{feat_tag}/fold{fold}")
                    # Tampilkan 3 baris error terakhir untuk debugging
                    err_lines = result.stderr.strip().split('\n')[-3:]
                    for line in err_lines:
                        print(f"   ⚠️  {line.strip()}")

    print(f"\n📦 Evaluasi selesai. {eval_count}/40 berhasil.")
    if failed_list:
        print(f"⚠️  {len(failed_list)} fold gagal:")
        for f in failed_list: print(f"   - {f}")

# ==========================
# 3. AGREGASI & ANALISIS
# ==========================
def extract_config_from_filename(filename: str):
    clean = filename.replace("metrics_", "").replace(".json", "")
    model_match = re.search(r'(cnn2d|resnet18|resnet34|resnet50)', clean)
    model = model_match.group(1) if model_match else "unknown"
    fold_match = re.search(r'fold(\d+)', clean)
    fold = int(fold_match.group(1)) if fold_match else 0

    # Baca use_delta dari JSON untuk tentukan fitur (xy vs xy_dxdy)
    json_path = REPORT_DIR / f"{filename}.json"
    features = "xy_dxdy"
    if json_path.exists():
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            if data.get("use_delta") is False:
                features = "xy"
        except Exception:
            pass
    return {"model": model, "features": features, "fold": fold}

def load_all_metrics():
    metrics_list = []
    seen_keys = set()  
    
    # ✅ Scan REKURSIF (baca root & subfolder)
    for json_file in REPORT_DIR.rglob("metrics_*.json"):
        # ✅ Filter ketat: abaikan file aneh/lama
        if not re.match(r'metrics_(cnn2d|resnet18|resnet34|resnet50)', json_file.name):
            continue
            
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Baca fitur langsung dari konten JSON
            features = "xy" if data.get("use_delta") is False else "xy_dxdy"
            
            # Parse model & fold dari nama file
            model_match = re.search(r'(cnn2d|resnet18|resnet34|resnet50)', json_file.stem)
            fold_match = re.search(r'fold(\d+)', json_file.stem)
            
            if not model_match or not fold_match:
                continue
                
            model = model_match.group(1)
            fold = int(fold_match.group(1))
            
            # ✅ Deduplikasi: jika (model, fold, fitur) sudah ada, skip
            key = (model, fold, features)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            
            metrics_list.append({
                "config": {"model": model, "features": features, "fold": fold},
                "accuracy": data["accuracy"],
                "f1_macro": data["f1_macro"],
                "per_class": data["per_class"],
                "confusion_matrix": np.array(data["confusion_matrix"])
            })
        except Exception as e:
            continue  # Skip file corrupt/error baca
    return metrics_list
         
def aggregate_by_config(metrics_list):
    grouped = defaultdict(list)
    for m in metrics_list:
        key = (m["config"]["model"], m["config"]["features"])
        grouped[key].append(m)

    results = {}
    for (model, features), folds in grouped.items():
        if len(folds) < 3:
            print(f"⚠️  {model}/{features}: hanya {len(folds)} fold (<3). Dilewati demi validitas statistik.")
            continue

        acc_vals = [f["accuracy"] for f in folds]
        f1_vals = [f["f1_macro"] for f in folds]
        cm_sum = np.sum([f["confusion_matrix"] for f in folds], axis=0)

        results[(model, features)] = {
            "accuracy_mean": np.mean(acc_vals), "accuracy_std": np.std(acc_vals),
            "f1_mean": np.mean(f1_vals), "f1_std": np.std(f1_vals),
            "cm_aggregated": cm_sum, "folds": folds
        }
    return results

# ==========================
# 4. VISUALISASI & LAPORAN
# ==========================
def generate_latex_table(results):
    rows = []
    for (m, f), s in sorted(results.items()):
        acc = f"{s['accuracy_mean']*100:.2f} $\\pm$ {s['accuracy_std']*100:.2f}"
        f1  = f"{s['f1_mean']*100:.2f} $\\pm$ {s['f1_std']*100:.2f}"
        rows.append(f"{m} & {f} & {acc} & {f1} \\\\ \\hline")

    latex = r"""\begin{table}[H]
  \centering
  \caption{Hasil Evaluasi K-Fold Cross-Validation (Mean $\pm$ Std Dev)}
  \label{tab:kfold-results}
  \begin{tabular}{|l|l|c|c|}
  \hline
  \textbf{Model} & \textbf{Fitur} & \textbf{Akurasi (\%)} & \textbf{F1-Score (\%)} \\ \hline
""" + "\n".join(rows) + r"""
  \end{tabular}
\end{table}"""
    out_path = OUTPUT_DIR / "table_kfold_results.tex"
    out_path.write_text(latex, encoding="utf-8")
    print(f"✅ Tabel LaTeX tersimpan: {out_path}")

def plot_barchart(results):
    if not results: return
    models = sorted(set(k[0] for k in results.keys()))
    features = sorted(set(k[1] for k in results.keys()))

    x_pos = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, feat in enumerate(features):
        means, stds = [], []
        for m in models:
            if (m, feat) in results:
                means.append(results[(m, feat)]["accuracy_mean"] * 100)
                stds.append(results[(m, feat)]["accuracy_std"] * 100)
            else:
                means.append(0); stds.append(0)
        positions = x_pos + (i - len(features)/2) * width
        ax.bar(positions, means, yerr=stds, capsize=5, width=width, label=feat, alpha=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.replace("resnet", "ResNet-").upper() if "resnet" in m else m.upper() for m in models])
    ax.set_ylabel("Akurasi (%)"); ax.set_title("Perbandingan Akurasi Model (K-Fold CV)")
    ax.legend(title="Fitur"); ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "bar_accuracy_kfold.png", dpi=300); plt.close()
    print(f"✅ Bar chart tersimpan")

def plot_boxplot(metrics_list):
    if not metrics_list:
        print("⚠️  Tidak ada data untuk boxplot")
        return
    data_for_plot = [{"config": f"{m['config']['model']} ({m['config']['features']})",
                      "accuracy": m["accuracy"] * 100} for m in metrics_list]
    df = pd.DataFrame(data_for_plot)
    plt.figure(figsize=(12, 6))
    try:
        import seaborn as sns
        sns.boxplot(data=df, x="config", y="accuracy", hue="config", palette="Set2", legend=False)
    except ImportError:
        df.boxplot(column="accuracy", by="config", ax=plt.gca())
    plt.ylabel("Akurasi (%)"); plt.title("Distribusi Akurasi per Lipatan (K-Fold)")
    plt.xticks(rotation=45, ha="right"); plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "boxplot_accuracy_by_fold.png", dpi=300); plt.close()
    print(f"✅ Boxplot tersimpan")

def plot_aggregated_cm(results):
    if not results: return
    # Ambil konfigurasi dengan akurasi tertinggi
    best_config = max(results.keys(), key=lambda k: results[k]["accuracy_mean"])
    
    cm = results[best_config]["cm_aggregated"]
    cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-10)
    labels = ["belum", "hati_hati", "hobi", "izin", "maaf", 
              "sahabat", "teman", "terima_kasih", "tidak_punya", "ulang"]

    plt.figure(figsize=(10, 8))
    try:
        import seaborn as sns
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
    except ImportError:
        plt.imshow(cm_norm, cmap="Blues"); plt.colorbar()
    plt.ylabel("True Label"); plt.xlabel("Predicted Label")
    plt.title(f"Confusion Matrix Teragregasi\n{best_config[0]} | {best_config[1]}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"cm_aggregated_{best_config[0]}_{best_config[1]}.png", dpi=300)
    plt.close()
    print(f"✅ Aggregated CM tersimpan untuk model terbaik: {best_config}")

# ==========================
# 5. MAIN
# ==========================
def main():
    run_evaluation()

    print("\n📂 Memuat metrik untuk agregasi...")
    metrics_list = load_all_metrics()
    print(f"✅ Ditemukan {len(metrics_list)} file metrik")

    results = aggregate_by_config(metrics_list)
    print(f"✅ {len(results)} konfigurasi siap dianalisis")

    if not results:
        print("⚠️  Tidak ada data valid. Pastikan training & evaluasi berjalan sukses.")
        return

    generate_latex_table(results)
    plot_barchart(results)
    plot_boxplot(metrics_list)
    plot_aggregated_cm(results)

    print("\n📊 RINGKASAN HASIL (Mean ± Std):")
    print(f"{'Model':<12} {'Fitur':<10} {'Akurasi (%)':<15} {'F1-Score (%)':<15}")
    print("-" * 55)
    for (m, f), s in sorted(results.items()):
        acc = f"{s['accuracy_mean']*100:.2f} ± {s['accuracy_std']*100:.2f}"
        f1  = f"{s['f1_mean']*100:.2f} ± {s['f1_std']*100:.2f}"
        print(f"{m:<12} {f:<10} {acc:<15} {f1:<15}")
    print(f"\n✅ Semua output tersimpan di: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()