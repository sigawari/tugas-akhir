# cek_leakage.py
import json
from pathlib import Path

# 🔧 FIX 1: Deteksi project root otomatis (2 level ke atas dari script ini)
# Struktur: tugas-akhir/src/cek_leakage.py -> parent.parent = tugas-akhir/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = PROJECT_ROOT / "dataset" / "splits" / "kfold"

print(f"🔍 Mengecek leakage di: {SPLIT_DIR}\n")

all_clean = True

for fold_file in sorted(SPLIT_DIR.glob("split_fold_*.json")):
    print(f"📄 Memeriksa {fold_file.name}...")
    
    with open(fold_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Ekstrak video_id dari train dan val
    # Kita pakai set agar operasi intersection cepat
    train_videos = {item.get("video_id") for item in data["splits"]["train"] if "video_id" in item}
    val_videos = {item.get("video_id") for item in data["splits"]["val"] if "video_id" in item}
    
    # 🔧 FIX 2: Cek Leakage HANYA di dalam fold yang sama (Intra-fold)
    # Jika ada video yang muncul di train DAN val fold ini -> LEAKAGE!
    overlap = train_videos & val_videos
    
    if overlap:
        print(f"   ❌ CRITICAL LEAKAGE: {len(overlap)} video muncul di Train & Val sekaligus!")
        print(f"      Contoh video bocor: {list(overlap)[:5]}")
        all_clean = False
    else:
        print(f"   ✅ Aman: Tidak ada tumpang tindih Train/Val di fold ini.")
    
    # Info tambahan (opsional)
    # print(f"   ℹ️  Total Train: {len(train_videos)}, Val: {len(val_videos)}")
    print("-" * 50)

if all_clean:
    print("\n🎉 SELAMAT! Semua fold bersih dari data leakage (intra-fold).")
    print("   Catatan: Video yang sama boleh muncul di train set fold yang berbeda. Itu wajar.")
else:
    print("\n⚠️  Ditemukan leakage! Periksa ulang script generate split (13_generate_kfold.py).")
    print("   Pastikan splitting dilakukan per FILE/VIDEO, bukan per frame.")