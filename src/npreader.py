import argparse
import numpy as np
from pathlib import Path
from typing import Optional, List

# Default folder relatif terhadap lokasi file ini
DEFAULT_DIR = str((Path(__file__).resolve().parent.parent / "data" / "processed" / "halo" / "pose").resolve())


def print_preview(name: str, arr: np.ndarray) -> None:
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


def detect_channels_from_D(D: int) -> int:
    """
    Deteksi jumlah fitur per landmark (C) dari D.
    - Umumnya: D = L * C
    - Kandidat lazim: C=4 ([x,y,dx,dy]) atau C=5 ([x,y,dx,dy,mask])

    Return: C
    """
    c4 = (D % 4 == 0)
    c5 = (D % 5 == 0)

    if c4 and not c5:
        return 4
    if c5 and not c4:
        return 5
    if c4 and c5:
        # Ambigu (mis. D=300 habis dibagi 4 dan 5).
        # Default aman untuk proyek kamu sekarang: 4.
        return 4
    # Kalau tidak bisa 4/5, fallback 4 tapi beri warning di caller.
    return 4


def inspect_all_npy(dir_path: Path, mmap: bool = False) -> None:
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
            arr = np.load(npy_file, mmap_mode="r" if mmap else None)
            print_preview(npy_file.name, arr)
        except Exception as e:
            print(f"❌ Gagal load {npy_file.name}: {e}")

    print("\n=== SELESAI LIST FILE ===\n")


def show_first_landmark(dir_path: Path, channels: Optional[int] = None, mmap: bool = False) -> None:
    """
    Tampilkan fitur untuk landmark pertama (sample 0, frame 0).
    - Jika channels=None, auto-detect dari D.
    - Jika channels=4 → [x,y,dx,dy]
    - Jika channels=5 → [x,y,dx,dy,mask]
    """
    x_path = dir_path / "X.npy"
    if not x_path.exists():
        print(f"❌ X.npy tidak ditemukan di {x_path}")
        return

    X = np.load(x_path, mmap_mode="r" if mmap else None)

    if X.ndim != 3:
        print(f"❌ Bentuk X tidak sesuai (bukan (N, T, D)): {X.shape}")
        return

    N, T, D = X.shape

    # Tentukan jumlah channel
    if channels is None:
        C = detect_channels_from_D(D)
        amb_c4 = (D % 4 == 0)
        amb_c5 = (D % 5 == 0)
        if amb_c4 and amb_c5:
            print(f"⚠️ D={D} habis dibagi 4 dan 5, asumsi default C=4. "
                  f"Pakai --channels 5 jika memang ada mask.")
        elif not amb_c4 and not amb_c5:
            print(f"⚠️ D={D} tidak habis dibagi 4/5, fallback C=4. "
                  f"Pertimbangkan cek format fitur dataset.")
    else:
        if channels not in (4, 5):
            raise ValueError("--channels harus 4 atau 5")
        C = channels
        if D % C != 0:
            print(f"⚠️ D={D} tidak habis dibagi C={C}. Interpretasi bisa salah.")

    L = D // C if D % C == 0 else None

    print("=== CONTOH LANDMARK PERTAMA ===")
    print(f"Sample index : 0 / N={N}")
    print(f"Frame index  : 0 / T={T}")
    print(f"Detected C   : {C} fitur/landmark")
    if L is not None:
        print(f"Computed L   : {L} landmark (D={D} = L*C)")
    else:
        print(f"Computed L   : (tidak bisa dihitung karena D={D} tidak cocok dengan C={C})")

    vec = X[0, 0, :]

    if C == 4:
        x, y, dx, dy = vec[0:4]
        print("\nFitur landmark #0:")
        print("x  :", x)
        print("y  :", y)
        print("dx :", dx)
        print("dy :", dy)

        n_landmarks_show = 3
        first = vec[: n_landmarks_show * 4]
        print(f"\n{n_landmarks_show*4} angka pertama ({n_landmarks_show} landmark × 4 fitur):")
        print(first)
        print("Interpretasi per 4 angka: [x, y, dx, dy]")

        # cek frame ke-5
        t = 25
        j = 0  # landmark 0

        idx = j * 4
        vec = X[0, t, idx:idx+4]

        print(f"\nFrame {t}, landmark {j}:")
        print("x :", vec[0])
        print("y :", vec[1])
        print("dx:", vec[2])
        print("dy:", vec[3])

    else:  # C == 5
        x, y, dx, dy, mask = vec[0:5]
        print("\nFitur landmark #0:")
        print("x   :", x)
        print("y   :", y)
        print("dx  :", dx)
        print("dy  :", dy)
        print("mask:", mask)

        n_landmarks_show = 3
        first = vec[: n_landmarks_show * 5]
        print(f"\n{n_landmarks_show*5} angka pertama ({n_landmarks_show} landmark × 5 fitur):")
        print(first)
        print("Interpretasi per 5 angka: [x, y, dx, dy, mask]")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect .npy dataset (X.npy, y.npy).")
    p.add_argument("--dir", type=str, default=DEFAULT_DIR, help="Folder yang berisi X.npy & y.npy")
    p.add_argument("--mmap", action="store_true", help="Load numpy dengan mmap_mode='r' (hemat RAM)")
    p.add_argument("--channels", type=int, default=None, choices=[4, 5],
                   help="Paksa interpretasi fitur per landmark (4 atau 5). Jika tidak diisi, auto-detect.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dir_path = Path(args.dir)

    inspect_all_npy(dir_path, mmap=args.mmap)
    show_first_landmark(dir_path, channels=args.channels, mmap=args.mmap)

"""
Cara pakai

Default (auto, biasanya akan pilih 4):

py npreader.py --dir ../data/processed/halo/full


Paksa 5 kalau suatu saat kamu punya dataset yang memang ada mask:

py npreader.py --dir ../data/processed/halo/full --channels 5
"""