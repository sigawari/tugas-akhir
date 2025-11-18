import numpy as np
from pathlib import Path

DIR_PATH = "../data/processed/halo/full"

def print_preview(name, arr):
    """Print informasi dan preview isi array numpy."""
    print(f"\n--- {name} ---")
    print(f"Shape : {arr.shape}")
    print(f"DType : {arr.dtype}")

    try:
        flat = arr.flatten()
        n_show = min(50, flat.shape[0])
        print(f"Preview ({n_show} elemen pertama):")
        print(flat[:n_show])
    except Exception as e:
        print(f"(Tidak bisa preview isi: {e})")


def inspect_all_npy(dir_path):
    dir_path = Path(dir_path)

    if not dir_path.exists():
        print(f"❌ Folder tidak ditemukan: {dir_path}")
        return

    print(f"\n=== INSPECT NPY FOLDER ===")
    print(f"📁 Folder: {dir_path}")

    npy_files = list(dir_path.glob("*.npy"))
    if not npy_files:
        print("❌ Tidak ada file .npy di folder ini.")
        return

    print(f"📦 Ditemukan {len(npy_files)} file NPY:")

    for npy_file in npy_files:
        print(f"\n➡️ File: {npy_file.name}")
        try:
            arr = np.load(npy_file)
            print_preview(npy_file.name, arr)
        except Exception as e:
            print(f"❌ Gagal load {npy_file.name}: {e}")

    print("\n=== SELESAI LIST FILE ===\n")


def show_first_landmark(dir_path):
    """Tampilkan x, y, dx, dy, mask untuk landmark pertama (sample 0, frame 0)."""
    dir_path = Path(dir_path)
    x_path = dir_path / "X.npy"

    if not x_path.exists():
        print(f"❌ X.npy tidak ditemukan di {x_path}")
        return

    X = np.load(x_path)

    if X.ndim != 3:
        print(f"❌ Bentuk X tidak sesuai (bukan (N, T, D)): {X.shape}")
        return

    print("=== CONTOH LANDMARK PERTAMA ===")
    print("Sample index : 0")
    print("Frame index  : 0")

    # 5 fitur pertama = x, y, dx, dy, mask
    x, y, dx, dy, mask = X[0, 0, 0:5]

    print("x   :", x)
    print("y   :", y)
    print("dx  :", dx)
    print("dy  :", dy)
    print("mask:", mask)

    # Lihat 3 landmark pertama (15 angka pertama)
    first_15 = X[0, 0, :15]
    print("\n15 angka pertama (3 landmark × 5 fitur):")
    print(first_15)
    print("Interpretasi per 5 angka: [x, y, dx, dy, mask]")


if __name__ == "__main__":
    inspect_all_npy(DIR_PATH)
    show_first_landmark(DIR_PATH)
