"""
download_dataset.py
-------------------
Auto-download dataset BISINDO dari Kaggle ke struktur folder yang benar.

Cara pakai:
    python scripts/download_dataset.py

Prasyarat:
    pip install kaggle
    Letakkan kaggle.json di ~/.kaggle/kaggle.json (chmod 600 di Linux/Mac)
    Lihat README.md bagian "Setup Dataset" untuk panduan lengkap.
"""

import sys
import os
import zipfile
import shutil
from pathlib import Path

# ── Root project ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Konfigurasi dataset ───────────────────────────────────────────────────────
KAGGLE_DATASET = "sikahnubuahtulilmi/slr-bisindo-landmarks"
DOWNLOAD_DIR   = ROOT_DIR / "data" / "_kaggle_tmp"

# Mapping: nama folder di dalam zip → path tujuan di project
# Struktur zip harus sesuai dengan ini (lihat KAGGLE_UPLOAD_GUIDE.md)
FOLDER_MAP = {
    "landmarks_extracted"    : ROOT_DIR / "data" / "interim" / "landmarks_extracted",
    "landmarks_selected"     : ROOT_DIR / "data" / "interim" / "landmarks_selected",
    "landmarks_interpolated" : ROOT_DIR / "data" / "interim" / "landmarks_interpolated",
    "landmarks_normalization": ROOT_DIR / "data" / "interim" / "landmarks_normalization",
    "npy"                    : ROOT_DIR / "data" / "processed" / "npy",
}

# Cek sentinel: kalau X.npy sudah ada → dataset dianggap lengkap
SENTINEL_FILE = ROOT_DIR / "data" / "processed" / "npy" / "X.npy"


def _is_dataset_ready() -> bool:
    return SENTINEL_FILE.exists()


def _check_kaggle_credentials():
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("\n[ERROR] kaggle.json tidak ditemukan di ~/.kaggle/kaggle.json")
        print("        Ikuti langkah berikut:")
        print("        1. Buka https://www.kaggle.com/settings")
        print("        2. Scroll ke bagian 'API' → klik 'Create New Token'")
        print("        3. Simpan kaggle.json ke:")
        print("             Linux/Mac : ~/.kaggle/kaggle.json")
        print("             Windows   : C:\\Users\\<username>\\.kaggle\\kaggle.json")
        print("        4. Linux/Mac: chmod 600 ~/.kaggle/kaggle.json")
        print("        5. Jalankan ulang script ini.\n")
        sys.exit(1)


def _download_from_kaggle():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApiExtended
    except ImportError:
        print("[ERROR] Package 'kaggle' belum terinstall.")
        print("        Jalankan: pip install kaggle")
        sys.exit(1)

    api = KaggleApiExtended()
    api.authenticate()

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Dataset  : {KAGGLE_DATASET}")
    print(f"[INFO] Temp dir : {DOWNLOAD_DIR}")
    print("[INFO] Mendownload... (±2 GB, mungkin beberapa menit)")

    api.dataset_download_files(
        KAGGLE_DATASET,
        path=str(DOWNLOAD_DIR),
        unzip=False,
        quiet=False,
    )

    zip_files = list(DOWNLOAD_DIR.glob("*.zip"))
    if not zip_files:
        print("[ERROR] File zip tidak ditemukan setelah download.")
        sys.exit(1)

    zip_path = zip_files[0]
    print(f"[INFO] Mengekstrak {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DOWNLOAD_DIR)
    zip_path.unlink()
    print("[INFO] Ekstraksi selesai.")


def _move_to_project():
    print("[INFO] Memindahkan file ke struktur project...")
    any_moved = False

    for src_name, dst_path in FOLDER_MAP.items():
        src_path = DOWNLOAD_DIR / src_name

        if not src_path.exists():
            print(f"  [SKIP] '{src_name}' tidak ada di zip — lewati.")
            continue

        dst_path.mkdir(parents=True, exist_ok=True)

        for item in src_path.iterdir():
            target = dst_path / item.name
            if target.exists():
                shutil.rmtree(target) if target.is_dir() else target.unlink()
            shutil.move(str(item), str(target))

        rel = dst_path.relative_to(ROOT_DIR)
        print(f"  [OK]   {src_name} → {rel}")
        any_moved = True

    if DOWNLOAD_DIR.exists():
        shutil.rmtree(DOWNLOAD_DIR)
        print("[INFO] Folder sementara dibersihkan.")

    if not any_moved:
        print("[ERROR] Tidak ada folder yang dipindahkan. Periksa struktur zip di Kaggle.")
        sys.exit(1)


def main():
    print("=" * 60)
    print("  BISINDO SLR — Dataset Downloader")
    print("=" * 60)

    if _is_dataset_ready():
        print("[OK] Dataset sudah lengkap, tidak perlu download ulang.")
        print(f"     Ditemukan: {SENTINEL_FILE.relative_to(ROOT_DIR)}")
        print("\nLanjutkan dengan:")
        print("  python scripts/run_preprocess.py --stage convert")
        print("  python scripts/run_train.py")
        return

    print("[INFO] Dataset belum ditemukan. Memulai download dari Kaggle...")
    _check_kaggle_credentials()
    _download_from_kaggle()
    _move_to_project()

    print("\n[DONE] Dataset berhasil didownload.")
    print("\nLanjutkan dengan:")
    print("  python scripts/run_preprocess.py --stage convert")
    print("  python scripts/run_train.py")


if __name__ == "__main__":
    main()