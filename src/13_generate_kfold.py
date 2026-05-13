# 13_generate_kfold.py
import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.resolve()

DATASET_INDEX = PROJECT_ROOT / "dataset" / "npy_dataset" / "dataset_index.json"
OUTPUT_DIR    = PROJECT_ROOT / "dataset" / "splits" / "kfold"

def clean_path(raw_path: str) -> str:
    # Ganti backslash ke forward slash agar aman di Python
    p = raw_path.replace("\\", "/")
    
    # Cari anchor "dataset/" dan ambil dari posisi itu
    idx = p.find("dataset/")
    if idx != -1:
        return p[idx:]
        
    # Fallback: hapus drive letter Windows (C:/, D:/, dll)
    if len(p) >= 2 and p[1] == ":":
        p = p[2:].lstrip("/")
    return p

def main():
    if not DATASET_INDEX.is_file():
        raise FileNotFoundError(f"dataset_index.json tidak ditemukan: {DATASET_INDEX}")

    print(f"📖 Membaca index: {DATASET_INDEX}")
    with open(DATASET_INDEX, "r", encoding="utf-8") as f:
        items = json.load(f)

    # 🔥 Bersihkan semua path
    cleaned_count = 0
    for item in items:
        for key in ["json_file", "npy_file"]:
            if key in item:
                old = item[key]
                item[key] = clean_path(old)
                if old != item[key]:
                    cleaned_count += 1

    print(f"🧹 Dibersihkan: {cleaned_count} path -> format relatif (dataset/...)")
    if items:
        print(f"📌 Contoh path setelah bersih: {items[0]['npy_file']}")

    labels = [item["label"] for item in items]
    unique_labels = sorted(set(labels))
    print(f"✅ Dataset: {len(items)} sampel | {len(unique_labels)} kelas")

    label2idx = {item["label"]: int(item["label_id"]) for item in items}
    idx2label = {str(v): k for k, v in sorted(label2idx.items(), key=lambda x: x[1])}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(items, labels), start=1):
        train_items = [items[i] for i in train_idx]
        val_items   = [items[i] for i in val_idx]

        split_data = {
            "meta": {
                "fold": fold_idx, "type": "stratified_5fold", "seed": 42,
                "num_samples": len(items), "num_classes": len(unique_labels),
                "split_type": "file_level_stratified"
            },
            "label2idx": label2idx, "idx2label": idx2label,
            "splits": {"train": train_items, "val": val_items}
        }

        out_path = OUTPUT_DIR / f"split_fold_{fold_idx}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Fold {fold_idx} | Train: {len(train_items)} | Val: {len(val_items)}")

    print(f"\n🎉 Selesai! JSON split tersimpan di: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()