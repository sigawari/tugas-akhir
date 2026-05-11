import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

# 🔧 Path Resolution: Naik 1 level dari src/ ke project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATASET_INDEX = PROJECT_ROOT / "dataset" / "npy_dataset" / "dataset_index.json"
OUTPUT_DIR    = PROJECT_ROOT / "dataset" / "splits" / "kfold"

def main():
    if not DATASET_INDEX.is_file():
        raise FileNotFoundError(f"❌ dataset_index.json tidak ditemukan: {DATASET_INDEX}")

    print(f"📖 Membaca index: {DATASET_INDEX}")
    with open(DATASET_INDEX, "r", encoding="utf-8") as f:
        items = json.load(f)

    labels = [item["label"] for item in items]
    unique_labels = sorted(set(labels))
    print(f"✅ Dataset: {len(items)} sampel | {len(unique_labels)} kelas")

    # Build mapping identik dengan dataset.py
    label2idx = {item["label"]: int(item["label_id"]) for item in items}
    idx2label = {str(v): k for k, v in sorted(label2idx.items(), key=lambda x: x[1])}

    # Stratified 5-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(items, labels), start=1):
        train_items = [items[i] for i in train_idx]
        val_items   = [items[i] for i in val_idx]

        # 📦 Struktur JSON SAMA PERSIS dengan output build_split_from_dataset_index()
        split_data = {
            "meta": {
                "fold": fold_idx,
                "type": "stratified_5fold",
                "seed": 42,
                "num_samples": len(items),
                "num_classes": len(unique_labels),
                "split_type": "file_level_stratified"
            },
            "label2idx": label2idx,
            "idx2label": idx2label,
            "splits": {
                "train": train_items,
                "val": val_items
            }
        }

        out_path = OUTPUT_DIR / f"split_fold_{fold_idx}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
            
        print(f"✅ Fold {fold_idx} | Train: {len(train_items)} | Val: {len(val_items)} -> {out_path}")

    print(f"\n🎉 Selesai! 5 split JSON tersimpan di: {OUTPUT_DIR}")
    print("💡 Cara pakai: tambahkan --split_path <path_ke_file_json> di train.py / 12_eval.py")

if __name__ == "__main__":
    main()