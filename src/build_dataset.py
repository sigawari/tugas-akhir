# build_dataset.py
# Convert raw JSON → NPY arrays (X, y) untuk ablation study.
#
# Fitur per landmark: x, y, dx, dy
#  - x, y   : posisi normal Mediapipe (tanpa z)
#  - dx, dy : delta (per frame) = x_t - x_{t-1}, y_t - y_{t-1}
#             frame 0 di-set 0
#
# Ablasi channel:
# (A) full   : Pose + Hands + Face subset (FACE_LANDMARK_MAP)
# (B) noface : Pose + Hands
# (C) hands  : Hands only
# (D) pose   : Pose only
#
# Input  (per kata):
#   data/raw/<kata>/data_json/sequence_*.json
#
# Output (per kata, per variant):
#   data/processed/<kata>/<variant>/X.npy    # shape (N, T, D)
#   data/processed/<kata>/<variant>/y.npy    # shape (N,) label (di sini semua 0)

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

# === Konfigurasi dasar (harus konsisten dengan data_collect.py) ===
SEQUENCE_LENGTH = 30  # jumlah frame per sequence

# Folder root project
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"

# Nama landmark pose (33 titik)
POSE_LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
    "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

# Nama landmark tangan (21 titik)
HAND_LANDMARK_NAMES = [
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
]

# Subset landmark wajah yang dipakai (index MediaPipe → nama semantis)
FACE_LANDMARK_MAP: Dict[int, str] = {
    234: "pipi_kiri",
    454: "pipi_kanan",
    10:  "jidat_tengah",
    297: "jidat_kiri",
    338: "jidat_kanan",
    152: "dagu",
    13:  "bibir_atas_tengah",
    14:  "bibir_bawah_tengah",
    61:  "bibir_kiri",
    291: "bibir_kanan",
    33:  "mata_kiri_luar",
    133: "mata_kiri_dalam",
    362: "mata_kanan_dalam",
    263: "mata_kanan_luar",
}

# Mendefinisikan kelas
WORD_LABEL_MAP = {
    "halo": 0,
    "maaf": 1,
    "permisi": 2,
    "terima_kasih": 3,
    "tolong": 4,
}

# Helper: baca JSON sequence
def load_sequence_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# Helper: definisikan urutan landmark untuk variant
def get_landmarks_order(variant: str) -> List[Tuple[str, str]]:
    """
    Return list of (group, key) in fixed order.
    group: "pose" | "left_hand" | "right_hand" | "face"
    key  : nama landmark (pose/hand) atau string index untuk face
    """
    order: List[Tuple[str, str]] = []

    include_pose = variant in ("full", "noface", "pose")
    include_face = variant in ("full",)
    include_hands = variant in ("full", "noface", "hands")

    if include_pose:
        for name in POSE_LANDMARK_NAMES:
            order.append(("pose", name))

    if include_hands:
        for name in HAND_LANDMARK_NAMES:
            order.append(("left_hand", name))
        for name in HAND_LANDMARK_NAMES:
            order.append(("right_hand", name))

    if include_face:
        # pakai urutan dari FACE_LANDMARK_MAP (insertion order)
        for idx in FACE_LANDMARK_MAP.keys():
            order.append(("face", str(idx)))

    return order


# Helper: sequence (list frame) → matrix (T, D)
def sequence_to_matrix(frames: List[dict], variant: str) -> np.ndarray:
    """
    frames : list data["frames"] dari JSON
    variant: full | noface | hands | pose
    Return: np.ndarray shape (T, D) dengan fitur [x, y, dx, dy] per landmark (flatten).
    """
    T = len(frames)
    if T == 0:
        raise ValueError("frames kosong")

    lm_order = get_landmarks_order(variant)
    L = len(lm_order)  # jumlah landmark total untuk variant ini

    # Siapkan array x, y, 
    x = np.zeros((T, L), dtype=np.float32)
    y = np.zeros((T, L), dtype=np.float32)

    for t, frame in enumerate(frames):
        lm_all = frame.get("landmarks", {})
        pose = lm_all.get("pose", {})
        left_hand = lm_all.get("left_hand", {})
        right_hand = lm_all.get("right_hand", {})
        face = lm_all.get("face", {})

        for j, (group, key) in enumerate(lm_order):
            if group == "pose":
                part = pose
                lk = key  # nama landmark pose
            elif group == "left_hand":
                part = left_hand
                lk = key
            elif group == "right_hand":
                part = right_hand
                lk = key
            elif group == "face":
                part = face
                lk = key  # string index, misal "234"
            else:
                continue

            if lk in part:
                v = part[lk]
                x[t, j] = float(v["x"])
                y[t, j] = float(v["y"])
            else:
                # sudah default 0 untuk x,y,
                # x[t,j] = 0; y[t,j] = 0; [t,j] = 0
                continue

    # Hitung delta sepanjang waktu
    dx = np.zeros_like(x)
    dy = np.zeros_like(y)
    if T > 1:
        dx[1:, :] = x[1:, :] - x[:-1, :]
        dy[1:, :] = y[1:, :] - y[:-1, :]

    # Gabungkan jadi (T, L, 4) → flatten ke (T, D)
    # urutan fitur per landmark: [x, y, dx, dy]
    features = np.stack([x, y, dx, dy], axis=-1)  # (T, L, 4)
    T_, L_, C_ = features.shape
    assert T_ == T and L_ == L and C_ == 4

    feat_flat = features.reshape(T, L * 4)  # (T, D)
    return feat_flat


def compute_feature_dim(variant: str) -> int:
    """Helper kecil buat ngecek dimensi fitur per frame untuk variant tertentu."""
    dummy_frame = {
        "landmarks": {
            "pose": {},          # tidak masalah kosong, cuma butuh shape
            "left_hand": {},
            "right_hand": {},
            "face": {},
        }
    }
    mat = sequence_to_matrix([dummy_frame], variant)
    return mat.shape[1]


# Build dataset UNTUK SATU KATA & SATU VARIANT
def build_class(word: str, variant: str):
    """
    word    : nama kata (folder di data/raw/<word>/data_json)
    variant : full | noface | hands | pose
    """
    word_raw_dir = RAW_ROOT / word / "data_json"
    if not word_raw_dir.exists():
        print(f"⚠️ Lewat '{word}': {word_raw_dir} tidak ada.")
        return

    seq_files = sorted(word_raw_dir.glob("sequence_*.json"))
    if not seq_files:
        print(f"⚠️ Lewat '{word}': tidak ada sequence_*.json.")
        return

    print(f"\n=== Kata '{word}' | Variant '{variant}' ===")
    print(f"   RAW dir  : {word_raw_dir}")
    print(f"   #files   : {len(seq_files)}")
    print(f"   feat dim : {compute_feature_dim(variant)} per frame")

    all_sequences = []

    for jp in seq_files:
        data = load_sequence_json(jp)
        frames = data.get("frames", [])
        if len(frames) != SEQUENCE_LENGTH:
            print(f"   ⚠️ Skip {jp.name}: frame len={len(frames)} (harus {SEQUENCE_LENGTH})")
            continue

        seq_mat = sequence_to_matrix(frames, variant)  # (T, D)
        all_sequences.append(seq_mat)

    if not all_sequences:
        print(f"❌ Tidak ada sequence valid untuk '{word}' variant '{variant}'.")
        return

    X = np.stack(all_sequences, axis=0)  # (N, T, D)

    # label berdasarkan kata
    if word not in WORD_LABEL_MAP:
        raise ValueError(f"Word '{word}' belum ada di WORD_LABEL_MAP")
    
    label = WORD_LABEL_MAP[word]
    y = np.full((X.shape[0],), label, dtype=np.int64)

    # Output: data/processed/<word>/<variant>/
    out_dir = PROCESSED_ROOT / word / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    x_path = out_dir / "X.npy"
    y_path = out_dir / "y.npy"

    np.save(x_path, X)
    np.save(y_path, y)

    # ===== PREVIEW isi NPY di sini =====
    print(f"   ✅ DONE '{word}' '{variant}'")
    print(f"      X path  : {x_path}")
    print(f"      y path  : {y_path}")
    print(f"      X shape : {X.shape}  (N, T, D)")
    print(f"      y shape : {y.shape}  (N,)")

    # Contoh isi dikit:
    N, T, D = X.shape
    print("      Preview:")
    print(f"        N Sequence     : {N}")
    print(f"        T Frame : {T}")
    print(f"        D Fitur/Frame  : {D}")
    print(f"        X[0, 0, :10]   : {X[0, 0, :10]}")


# Build dataset: loop kata & variant
def build_dataset(variants: List[str], words: List[str] | None = None):
    # kalau words tidak diberikan → ambil semua folder di data/raw/<kata>/data_json
    if not words:
        words = []
        for d in RAW_ROOT.iterdir():
            if not d.is_dir():
                continue
            if (d / "data_json").exists():
                words.append(d.name)

    words = sorted(words)
    if not words:
        print("❌ Tidak ada folder kata di data/raw. Rekam dulu pakai data_collect.py.")
        return

    print(f"📂 RAW_ROOT      : {RAW_ROOT}")
    print(f"💾 PROCESSED_ROOT: {PROCESSED_ROOT}")
    print(f"📝 Kata          : {words}")
    print(f"🔧 Variants      : {variants}")

    for word in words:
        for v in variants:
            build_class(word, v)


# CLI
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        type=str,
        nargs="*",
        default=["full", "noface", "hands", "pose"],
        choices=["full", "noface", "hands", "pose"],
        help="Daftar variant yang mau dibuat (default: semua)."
    )
    parser.add_argument(
        "--words",
        type=str,
        nargs="*",
        help="Daftar kata (folder di data/raw) yang mau diproses. Default: semua folder yang punya data_json."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(args.variants, args.words)
