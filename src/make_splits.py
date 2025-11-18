# make_splits.py
# Create train/val/test split once from raw JSON.
# Output: split_default.json in data/processed/splits/.
import os, json, glob
from sklearn.model_selection import StratifiedKFold

DATA_DIR = "data_json"
LABELS = ["halo", "terima_kasih"]  # fokus 2 kata dulu
SPLIT_DIR = "splits"
os.makedirs(SPLIT_DIR, exist_ok=True)

def collect_files():
    files, y = [], []
    for lab in LABELS:
        paths = sorted(glob.glob(os.path.join(DATA_DIR, lab, "*.json")))
        files.extend(paths)
        y.extend([lab] * len(paths))
    if len(files) == 0:
        raise RuntimeError(f"Tidak ada file json di {DATA_DIR}. Cek struktur foldernya.")
    return files, y

def main(n_splits=5, random_state=42):
    files, y = collect_files()
    print(f"Total files: {len(files)} | labels: {set(y)}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(files, y), start=1):
        split_data = {
            "labels": LABELS,  # simpan daftar label yang dipakai
            "train_files": [files[i] for i in tr_idx],
            "val_files":   [files[i] for i in va_idx]
        }
        out = os.path.join(SPLIT_DIR, f"5fold_split_fold{fold_id}.json")
        with open(out, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"Wrote {out}: train={len(tr_idx)} val={len(va_idx)}")

if __name__ == "__main__":
    main()
