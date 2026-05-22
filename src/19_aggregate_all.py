"""
19_fold_per_model.py
Agregasi metrik per fold untuk tiap model (xy vs xy_dxdy berdampingan).
Output: Terminal table + CSV + LaTeX table siap Bab 4.
"""
import json, re
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
REPORT_DIR   = PROJECT_ROOT / "reports"
OUTPUT_DIR   = REPORT_DIR / "aggregated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["cnn2d", "resnet18", "resnet34", "resnet50"]

def load_metrics():
    records = []
    for json_file in REPORT_DIR.rglob("metrics_*.json"):
        if not re.match(r'metrics_(cnn2d|resnet18|resnet34|resnet50)', json_file.name):
            continue
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            model = re.search(r'(cnn2d|resnet18|resnet34|resnet50)', json_file.stem).group(1)
            fold  = int(re.search(r'fold(\d+)', json_file.stem).group(1))
            feat  = "xy" if data.get("use_delta") is False else "xy_dxdy"
            records.append({
                "model": model, "fold": fold, "feature": feat,
                "accuracy": data["accuracy"], "f1_macro": data["f1_macro"]
            })
        except Exception as e:
            print(f"⚠️ Skip {json_file.name}: {e}")
    return pd.DataFrame(records)

def aggregate_per_model_per_fold(df):
    """Group by model & fold, tampilkan xy & xy_dxdy berdampingan."""
    # Pivot: index=model+fold, columns=feature, values=[accuracy, f1]
    pivot = df.pivot_table(
        index=["model", "fold"], 
        columns="feature", 
        values=["accuracy", "f1_macro"],
        aggfunc="mean"
    )
    # Flatten column names
    pivot.columns = [f"{val}_{feat}" for val, feat in pivot.columns]
    pivot = pivot.reset_index()
    
    # Tambah kolom mean±std across features (opsional, untuk ringkasan)
    pivot["acc_mean"] = pivot[["accuracy_xy", "accuracy_xy_dxdy"]].mean(axis=1)
    pivot["acc_std"]  = pivot[["accuracy_xy", "accuracy_xy_dxdy"]].std(axis=1, ddof=1)
    pivot["f1_mean"]  = pivot[["f1_macro_xy", "f1_macro_xy_dxdy"]].mean(axis=1)
    pivot["f1_std"]   = pivot[["f1_macro_xy", "f1_macro_xy_dxdy"]].std(axis=1, ddof=1)
    
    # Urutkan sesuai MODEL_ORDER
    pivot["model"] = pd.Categorical(pivot["model"], categories=MODEL_ORDER, ordered=True)
    pivot = pivot.sort_values(["model", "fold"]).reset_index(drop=True)
    return pivot

def print_fold_table(pivot):
    print("\n📊 REKAP PER-FOLD TIAP MODEL (Accuracy % & F1-Score %)")
    print("="*110)
    header = f"{'Model':<12} {'Fold':<6} {'xy Acc':<8} {'xy F1':<8} {'xy_dxdy Acc':<12} {'xy_dxdy F1':<12} {'Mean Acc':<10} {'Std Acc':<8}"
    print(header)
    print("-"*110)
    
    for _, row in pivot.iterrows():
        print(
            f"{row['model']:<12} {row['fold']:<6} "
            f"{row['accuracy_xy']*100:>7.2f}%  {row['f1_macro_xy']*100:>7.2f}%  "
            f"{row['accuracy_xy_dxdy']*100:>10.2f}%  {row['f1_macro_xy_dxdy']*100:>10.2f}%  "
            f"{row['acc_mean']*100:>9.2f}%  {row['acc_std']*100:>7.2f}%"
        )
    print("="*110 + "\n")

def export_csv(pivot, path):
    pivot.to_csv(path, index=False, float_format="%.4f")
    print(f"✅ CSV tersimpan: {path}")

def export_latex(pivot, path):
    """LaTeX table: Model | Fold 1 | Fold 2 | ... | Fold 5 | Mean ± Std"""
    # Group by model, pivot fold ke kolom
    latex_rows = []
    for model in MODEL_ORDER:
        model_data = pivot[pivot["model"] == model]
        if model_data.empty: continue
        
        fold_accs = [f"{model_data[model_data['fold']==f]['acc_mean'].values[0]*100:.2f}" for f in range(1,6)]
        mean_acc  = model_data["acc_mean"].mean() * 100
        std_acc   = model_data["acc_mean"].std(ddof=1) * 100
        
        row = f"{model.replace('resnet','ResNet-').upper()} & " + " & ".join(fold_accs) + f" & {mean_acc:.2f} $\\pm$ {std_acc:.2f} \\\\ \\hline"
        latex_rows.append(row)
    
    latex = r"""\begin{table}[H]
  \centering
  \caption{Akurasi Rata-Rata per Lipatan Tiap Arsitektur (Mean dari xy \& xy\_dxdy)}
  \label{tab:fold-per-model}
  \begin{tabular}{|l|c|c|c|c|c|c|}
  \hline
  \textbf{Model} & \textbf{Fold 1} & \textbf{Fold 2} & \textbf{Fold 3} & \textbf{Fold 4} & \textbf{Fold 5} & \textbf{Mean $\pm$ Std (\%)} \\ \hline
""" + "\n".join(latex_rows) + r"""
  \end{tabular}
\end{table}"""
    
    path.write_text(latex, encoding="utf-8")
    print(f"✅ LaTeX tersimpan: {path}")

def main():
    print("🔍 Memuat metrics untuk agregasi per fold per model...")
    df = load_metrics()
    if df.empty:
        print("❌ Tidak ada data ditemukan.")
        return
    
    pivot = aggregate_per_model_per_fold(df)
    print_fold_table(pivot)
    
    csv_path  = OUTPUT_DIR / "fold_aggregation_per_model.csv"
    latex_path = OUTPUT_DIR / "table_fold_per_model.tex"
    
    export_csv(pivot, csv_path)
    export_latex(pivot, latex_path)
    
    print(f"\nOutput tersimpan di: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()