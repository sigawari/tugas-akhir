#!/usr/bin/env python3
"""
18_detailed_recap.py
Menampilkan recap lengkap per-fold: accuracy, F1, precision, recall untuk semua konfigurasi.
Output: Terminal table + CSV + LaTeX table untuk Bab 4 skripsi.
"""
import json, re, csv
import numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict

# ==========================
# KONFIGURASI
# ==========================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
REPORT_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR = REPORT_DIR / "aggregated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================
# LOAD & PARSE METRICS
# ==========================
def load_all_metrics_detailed():
    """Load semua metrics dengan detail per-class & per-fold."""
    records = []
    
    for json_file in REPORT_DIR.rglob("metrics_*.json"):
        # Filter hanya file yang valid
        if not re.match(r'metrics_(cnn2d|resnet18|resnet34|resnet50)', json_file.name):
            continue
            
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Parse metadata dari filename
            model_match = re.search(r'(cnn2d|resnet18|resnet34|resnet50)', json_file.stem)
            fold_match = re.search(r'fold(\d+)', json_file.stem)
            
            if not model_match or not fold_match:
                continue
                
            model = model_match.group(1)
            fold = int(fold_match.group(1))
            features = "xy" if data.get("use_delta") is False else "xy_dxdy"
            
            # Extract per-class metrics
            per_class = data.get("per_class", {})
            
            # Hitung macro-average precision & recall dari per-class
            precisions = [v["precision"] for v in per_class.values()]
            recalls = [v["recall"] for v in per_class.values()]
            f1s = [v["f1"] for v in per_class.values()]
            
            record = {
                "model": model,
                "features": features,
                "fold": fold,
                "accuracy": data["accuracy"],
                "f1_macro": data["f1_macro"],
                "precision_macro": np.mean(precisions) if precisions else 0,
                "recall_macro": np.mean(recalls) if recalls else 0,
                "balanced_accuracy": data.get("balanced_accuracy", 0),
                "val_loss": data.get("val_loss", 0),
                "per_class": per_class,  # Simpan untuk analisis mendalam
                "confusion_matrix": np.array(data.get("confusion_matrix", []))
            }
            records.append(record)
            
        except Exception as e:
            print(f"⚠️  Gagal baca {json_file.name}: {e}")
            continue
    
    return records

# ==========================
# DISPLAY & EXPORT
# ==========================
def print_detailed_table(records):
    """Print tabel lengkap ke terminal."""
    # Sort: model → features → fold
    records_sorted = sorted(records, key=lambda r: (
        ["cnn2d", "resnet18", "resnet34", "resnet50"].index(r["model"]),
        r["features"], r["fold"]
    ))
    
    print("\n" + "="*130)
    print(f"{'Model':<12} {'Fitur':<10} {'Fold':<6} {'Acc':<8} {'F1':<8} {'Prec':<8} {'Rec':<8} {'BalAcc':<10} {'ValLoss':<10}")
    print("-"*130)
    
    for r in records_sorted:
        print(
            f"{r['model']:<12} "
            f"{r['features']:<10} "
            f"{r['fold']:<6} "
            f"{r['accuracy']*100:>7.2f}% "
            f"{r['f1_macro']*100:>7.2f}% "
            f"{r['precision_macro']*100:>7.2f}% "
            f"{r['recall_macro']*100:>7.2f}% "
            f"{r['balanced_accuracy']*100:>9.2f}% "
            f"{r['val_loss']:>9.4f}"
        )
    print("="*130 + "\n")

def export_to_csv(records, output_path):
    """Export ke CSV untuk analisis Excel/Google Sheets."""
    rows = []
    for r in records:
        row = {
            "model": r["model"],
            "features": r["features"],
            "fold": r["fold"],
            "accuracy": r["accuracy"],
            "f1_macro": r["f1_macro"],
            "precision_macro": r["precision_macro"],
            "recall_macro": r["recall_macro"],
            "balanced_accuracy": r["balanced_accuracy"],
            "val_loss": r["val_loss"]
        }
        # Tambah per-class metrics sebagai kolom terpisah
        for class_name, metrics in r["per_class"].items():
            row[f"{class_name}_f1"] = metrics["f1"]
            row[f"{class_name}_precision"] = metrics["precision"]
            row[f"{class_name}_recall"] = metrics["recall"]
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, float_format="%.4f")
    print(f"✅ CSV tersimpan: {output_path}")

def export_latex_summary(records, output_path):
    """Export tabel agregasi (mean±std) format LaTeX untuk Bab 4."""
    # Group by (model, features)
    grouped = defaultdict(list)
    for r in records:
        grouped[(r["model"], r["features"])].append(r)
    
    rows = []
    for (model, features), folds in sorted(grouped.items()):
        acc_mean = np.mean([f["accuracy"] for f in folds]) * 100
        acc_std = np.std([f["accuracy"] for f in folds]) * 100
        f1_mean = np.mean([f["f1_macro"] for f in folds]) * 100
        f1_std = np.std([f["f1_macro"] for f in folds]) * 100
        
        rows.append(
            f"{model} & {features} & "
            f"{acc_mean:.2f} $\\pm$ {acc_std:.2f} & "
            f"{f1_mean:.2f} $\\pm$ {f1_std:.2f} \\\\ \\hline"
        )
    
    latex = r"""\begin{table}[H]
  \centering
  \caption{Hasil Evaluasi K-Fold Cross-Validation (Mean $\pm$ Std Dev)}
  \label{tab:kfold-detailed}
  \begin{tabular}{|l|l|c|c|}
  \hline
  \textbf{Model} & \textbf{Fitur} & \textbf{Akurasi (\%)} & \textbf{F1-Score (\%)} \\ \hline
""" + "\n".join(rows) + r"""
  \end{tabular}
\end{table}"""
    
    output_path.write_text(latex, encoding="utf-8")
    print(f"✅ LaTeX table tersimpan: {output_path}")

def print_per_class_analysis(records, model, features):
    """Print analisis per-kelas untuk konfigurasi tertentu."""
    print(f"\n📊 Analisis Per-Kelas: {model} | {features}")
    print("-"*80)
    
    # Ambil semua fold untuk konfigurasi ini
    folds = [r for r in records if r["model"] == model and r["features"] == features]
    
    if not folds:
        print("⚠️  Tidak ada data")
        return
    
    # Agregasi per-class metrics across folds
    class_names = list(folds[0]["per_class"].keys())
    aggregated = {}
    
    for class_name in class_names:
        f1_vals = [f["per_class"][class_name]["f1"] for f in folds]
        prec_vals = [f["per_class"][class_name]["precision"] for f in folds]
        rec_vals = [f["per_class"][class_name]["recall"] for f in folds]
        
        aggregated[class_name] = {
            "f1_mean": np.mean(f1_vals),
            "f1_std": np.std(f1_vals),
            "prec_mean": np.mean(prec_vals),
            "rec_mean": np.mean(rec_vals)
        }
    
    print(f"{'Kelas':<15} {'F1 (mean±std)':<20} {'Prec (mean)':<15} {'Rec (mean)':<15}")
    print("-"*80)
    for class_name in sorted(class_names):
        stats = aggregated[class_name]
        print(
            f"{class_name:<15} "
            f"{stats['f1_mean']*100:>6.2f}% ± {stats['f1_std']*100:>5.2f}%   "
            f"{stats['prec_mean']*100:>10.2f}%      "
            f"{stats['rec_mean']*100:>10.2f}%"
        )
    print("-"*80)

# ==========================
# MAIN
# ==========================
def main():
    print("🔍 Memuat semua metrics untuk recap detail...\n")
    records = load_all_metrics_detailed()
    
    if not records:
        print("❌ Tidak ada file metrics yang ditemukan!")
        return
    
    print(f"✅ Ditemukan {len(records)} record metrik\n")
    
    # 1. Print tabel lengkap ke terminal
    print_detailed_table(records)
    
    # 2. Export ke CSV
    csv_path = OUTPUT_DIR / "detailed_metrics_all_folds.csv"
    export_to_csv(records, csv_path)
    
    # 3. Export LaTeX summary table
    latex_path = OUTPUT_DIR / "table_kfold_summary.tex"
    export_latex_summary(records, latex_path)
    
    # 4. (Opsional) Print per-class analysis untuk model terbaik
    print_per_class_analysis(records, "resnet18", "xy_dxdy")
    print_per_class_analysis(records, "cnn2d", "xy")  # Bandingkan dengan yang terendah
    
    print(f"\n📁 Semua output tersimpan di: {OUTPUT_DIR}")
    print("💡 Tips: Buka CSV di Excel untuk filter/sort, atau copy LaTeX table ke Bab 4.")

if __name__ == "__main__":
    main()