# fix_split_paths.py
import json
from pathlib import Path
import re

SPLIT_DIR = Path("dataset/splits/kfold")

# Pola untuk mendeteksi path Windows (C:\, D:\, dll)
WIN_PATTERN = re.compile(r'^[A-Za-z]:\\')

def fix_path(p_str: str) -> str:
    p = Path(p_str)
    # Jika terdeteksi path Windows, ambil bagian dari 'npy_dataset' atau 'dataset'
    if WIN_PATTERN.match(p_str) or (len(p_str) >= 2 and p_str[1] == ':'):
        parts = p.parts
        for keyword in ["npy_dataset", "dataset"]:
            if keyword in parts:
                idx = parts.index(keyword)
                return str(Path(*parts[idx:]))
        # Fallback: ambil 3 komponen terakhir (kelas/nama_file.npy)
        return str(Path(*parts[-3:]))
    return p_str

def recursive_fix(obj):
    if isinstance(obj, str):
        return fix_path(obj)
    elif isinstance(obj, list):
        return [recursive_fix(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: recursive_fix(v) for k, v in obj.items()}
    return obj

print("🔧 Memulai perbaikan path split JSON...")
for json_file in SPLIT_DIR.glob("split_fold_*.json"):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    fixed_data = recursive_fix(data)
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(fixed_data, f, indent=2)
    print(f"✅ {json_file.name} -> path telah dikonversi ke relatif")

print("🎉 Selesai! Semua split JSON siap dijalankan di Linux.")