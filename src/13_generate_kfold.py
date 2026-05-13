import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

# 🔧 Path Resolution
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.resolve()  # resolve() penting untuk relative_to()

DATASET_INDEX = PROJECT_ROOT / "dataset" / "npy_dataset" / "dataset_index.json"
OUTPUT_DIR    = PROJECT_ROOT / "dataset" / "splits" / "kfold"

def normalize_path(file_path: str, base_dir: Path) -> str:
    """Mengubah path absolute (termasuk Windows) menjadi relatif terhadap PROJECT_ROOT."""
    # 1. Ganti backslash ke forward slash
    cleaned = file_path.replace("\\", "/")
    # 2. Hapus drive letter Windows jika ada (C:/, D:/, dll)
    if len(cleaned) >= 2 and cleaned[1] == ":":
        cleaned = cleaned[2:].lstrip("/")
    
    p = Path(cleaned)
    try:
        # Gabungkan dengan base_dir, resolve() untuk bersihkan "..", lalu jadikan relatif
        full_path = (base_dir / p).resolve()
        rel_path = full_path.relative_to(base_dir.resolve())
        return rel_path.as_posix()  # Pastikan pakai "/"
    except ValueError:
        # Fallback: kembalikan path bersih jika di luar struktur project
        print(f"⚠️ Path di luar PROJECT_ROOT, disimpan relatif: {cleaned}")
        return cleaned

def main():
    if not DATASET_INDEX.is_file():
        raise FileNotFoundError(f"❌ dataset_index.json tidak ditemukan: {DATASET_INDEX}")

    print(f"📖 Membaca index: {DATASET_INDEX}")
    with open(DATASET_INDEX, "r", encoding="utf-8") as f:
        items = json.load(f)

    # 🔥 NORMALISASI PATH DI SINI
    for item in items:
        for key in ["json_file", "npy_file"]:
            if key in item:
                item[key] = normalize_path(item[key], PROJECT_ROOT)

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

if __name__ == "__main__":
    main()