"""
15_aggregate_kfold.py
Mengagregasi hasil evaluasi 5-fold menjadi tabel ringkasan dan visualisasi untuk Bab 4.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--report_dir", type=str, default="reports")
    p.add_argument("--output_dir", type=str, default="reports/aggregated")
    p.add_argument("--models", nargs="+", default=["cnn2d", "resnet18", "resnet34", "resnet50"])
    p.add_argument("--features", nargs="+", default=["xy", "xy_dxdy"])
    return p.parse_args()

def extract_config_from_filename(filename: str):
    """
    Parse filename: metrics_resnet18_xy_dxdy_lr0.0001_bs16_fold3__best
    Return: dict with model, features, fold, etc.
    """
    # Implementasi parsing sesuai naming convention kamu
    # Contoh sederhana:
    parts = filename.replace("metrics_", "").replace("__best", "").split("_")
    # ... (sesuaikan dengan pola nama file kamu)
    return {"model": parts[0], "features": "_".join(parts[1:3]), "fold": int(parts[-1].replace("fold", ""))}

def load_all_metrics(report_dir: Path):
    """Load semua file metrics_*.json dari direktori laporan."""
    metrics_list = []
    for json_file in report_dir.glob("metrics_*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Ekstrak info konfigurasi dari nama file atau dari checkpoint config
        config = extract_config_from_filename(json_file.stem)
        
        metrics_list.append({
            "config": config,
            "accuracy": data["accuracy"],
            "f1_macro": data["f1_macro"],
            "precision": np.mean([v["precision"] for v in data["per_class"].values()]),
            "recall": np.mean([v["recall"] for v in data["per_class"].values()]),
            "confusion_matrix": np.array(data["confusion_matrix"]),
            "per_class": data["per_class"]
        })
    return metrics_list

def aggregate_by_config(metrics_list):
    """Kelompokkan metrik berdasarkan (model, fitur) dan hitung mean±std."""
    grouped = defaultdict(list)
    
    for m in metrics_list:
        key = (m["config"]["model"], m["config"]["features"])
        grouped[key].append(m)
    
    results = {}
    for (model, features), folds in grouped.items():
        if len(folds) != 5:
            print(f"⚠️  {model}/{features}: hanya {len(folds)} fold, butuh 5!")
            continue
            
        acc_vals = [f["accuracy"] for f in folds]
        f1_vals = [f["f1_macro"] for f in folds]
        
        # Agregasi confusion matrix (jumlahkan, lalu normalisasi jika perlu)
        cm_sum = np.sum([f["confusion_matrix"] for f in folds], axis=0)
        
        results[(model, features)] = {
            "accuracy_mean": np.mean(acc_vals),
            "accuracy_std": np.std(acc_vals),
            "f1_mean": np.mean(f1_vals),
            "f1_std": np.std(f1_vals),
            "cm_aggregated": cm_sum,
            "folds": folds  # Simpan untuk analisis per-kelas jika perlu
        }
    return results

def generate_latex_table(results, output_path: Path):
    """Hasilkan kode LaTeX tabel ringkasan hasil."""
    rows = []
    for (model, features), stats in sorted(results.items()):
        acc_str = f"{stats['accuracy_mean']*100:.2f} $\pm$ {stats['accuracy_std']*100:.2f}"
        f1_str = f"{stats['f1_mean']*100:.2f} $\pm$ {stats['f1_std']*100:.2f}"
        rows.append(f"{model} & {features} & {acc_str} & {f1_str} \\\\ \\hline")
    
    latex_code = r"""
\begin{table}[H]
  \centering
  \caption{Hasil evaluasi 5-fold cross-validation (Mean $\pm$ Std Dev)}
  \label{tab:hasil-kfold}
  \begin{tabular}{|l|l|c|c|}
  \hline
  \textbf{Model} & \textbf{Fitur} & \textbf{Akurasi (\%)} & \textbf{F1-Score (\%)} \\ \hline
""" + "\n".join(rows) + r"""
  \end{tabular}
\end{table}
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_code)
    print(f"✅ Tabel LaTeX tersimpan: {output_path}")

def plot_accuracy_barchart(results, output_path: Path):
    """Bar chart akurasi dengan error bars."""
    models = sorted(set(k[0] for k in results.keys()))
    x_pos = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot untuk xy dan xy_dxdy
    for i, feat in enumerate(["xy", "xy_dxdy"]):
        means = [results[(m, feat)]["accuracy_mean"]*100 for m in models if (m, feat) in results]
        stds = [results[(m, feat)]["accuracy_std"]*100 for m in models if (m, feat) in results]
        positions = x_pos + (i - 0.5) * width
        ax.bar(positions, means, yerr=stds, capsize=5, width=width, 
               label=feat, alpha=0.8)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.replace("resnet", "ResNet-").upper() if "resnet" in m else m.upper() for m in models])
    ax.set_ylabel("Akurasi (%)")
    ax.set_title("Perbandingan Akurasi Model (5-Fold CV)")
    ax.legend(title="Fitur")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Grafik bar chart tersimpan: {output_path}")
    plt.close()

def plot_boxplot_by_fold(metrics_list, output_path: Path):
def plot_boxplot_by_fold(metrics_list, output_path: Path):
    """Boxplot distribusi akurasi per konfigurasi."""
    try:
        import seaborn as sns
    except ImportError as e:
        raise ImportError("Seaborn belum terpasang. Install dengan: pip install seaborn") from e

    data_for_plot = []
    for m in metrics_list:
        label = f"{m['config']['model']}\n{m['config']['features']}"
        data_for_plot.append({
            "config": label,
            "fold": m["config"]["fold"],
            "accuracy": m["accuracy"] * 100
        })
    
    df = pd.DataFrame(data_for_plot)
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="config", y="accuracy", palette="Set2")
    plt.ylabel("Akurasi (%)")
    plt.title("Distribusi Akurasi per Lipatan (5-Fold)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
def plot_aggregated_cm(results, output_path: Path, config_key: tuple):
    """Plot confusion matrix teragregasi untuk konfigurasi tertentu."""
    try:
        import seaborn as sns
    except ImportError as e:
        raise ImportError("Seaborn belum terpasang. Install dengan: pip install seaborn") from e

    if config_key not in results:
        print(f"⚠️  Konfigurasi {config_key} tidak ditemukan")
        return
    
    cm = results[config_key]["cm_aggregated"]
    # Normalisasi per baris (true label)
    cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-10)
    
    labels = ["belum", "hati\_hati", "hobi", "izin", "maaf", 
              "sahabat", "teman", "terima\_kasih", "tidak\_punya", "ulang"]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=labels, yticklabels=labels)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(f"Confusion Matrix Teragregasi (5 Fold)\n{config_key[0]} | Fitur: {config_key[1]}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Aggregated CM tersimpan: {output_path}")
    plt.close()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Aggregated CM tersimpan: {output_path}")
    plt.close()

def main():
    args = parse_args()
    report_dir = Path(args.report_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Memuat metrik dari: {report_dir}")
    metrics_list = load_all_metrics(report_dir)
    print(f"✅ Ditemukan {len(metrics_list)} file metrik")
    
    print("🔄 Mengagregasi hasil per konfigurasi...")
    results = aggregate_by_config(metrics_list)
    print(f"✅ {len(results)} konfigurasi siap dianalisis")
    
    # 1. Generate LaTeX table
    generate_latex_table(results, output_dir / "table_kfold_results.tex")
    
    # 2. Plot bar chart with error bars
    plot_accuracy_barchart(results, output_dir / "bar_accuracy_kfold.png")
    
    # 3. Plot boxplot distribusi
    plot_boxplot_by_fold(metrics_list, output_dir / "boxplot_accuracy_by_fold.png")
    
    # 4. Plot aggregated CM untuk konfigurasi terbaik (contoh: resnet18 + xy)
    best_config = ("resnet18", "xy")
    if best_config in results:
        plot_aggregated_cm(results, output_dir / f"cm_aggregated_{'_'.join(best_config)}.png", best_config)
    
    # 5. Print ringkasan untuk copy-paste cepat
    print("\n📊 RINGKASAN HASIL (Mean ± Std):")
    print(f"{'Model':<12} {'Fitur':<10} {'Akurasi (%)':<15} {'F1-Score (%)':<15}")
    print("-" * 50)
    for (model, features), stats in sorted(results.items()):
        acc = f"{stats['accuracy_mean']*100:.2f} ± {stats['accuracy_std']*100:.2f}"
        f1 = f"{stats['f1_mean']*100:.2f} ± {stats['f1_std']*100:.2f}"
        print(f"{model:<12} {features:<10} {acc:<15} {f1:<15}")

if __name__ == "__main__":
    main()