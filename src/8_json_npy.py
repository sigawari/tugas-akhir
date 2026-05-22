import os
import json
import numpy as np
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_PATH = os.path.join(BASE_DIR, "..", "dataset", "normalized_json")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "dataset", "npy_dataset")

os.makedirs(OUTPUT_PATH, exist_ok=True)

TARGET_FRAMES = 90

# =========================
# LANDMARK ORDER
# =========================

POSE_KEYS = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb"
]

HAND_KEYS = [str(i) for i in range(21)]
FACE_KEYS = [str(i) for i in range(68)]


def build_landmark_index_map():
    """
    Build mapping L-index -> landmark name
    Final order:
      0-11   : pose
      12-32  : left_hand
      33-53  : right_hand
      54-121 : face
    """
    index_map = {}
    idx = 0

    for key in POSE_KEYS:
        index_map[idx] = {
            "component": "pose",
            "name": key
        }
        idx += 1

    for key in HAND_KEYS:
        index_map[idx] = {
            "component": "left_hand",
            "name": key
        }
        idx += 1

    for key in HAND_KEYS:
        index_map[idx] = {
            "component": "right_hand",
            "name": key
        }
        idx += 1

    for key in FACE_KEYS:
        index_map[idx] = {
            "component": "face",
            "name": key
        }
        idx += 1

    return index_map


INDEX_MAP = build_landmark_index_map()


# =========================
# JSON -> ARRAY
# =========================

def frame_to_tensor(frame):
    """
    Convert 1 frame JSON -> np.array [122, 2]
    """
    lm = frame["landmarks"]

    arr = []

    # pose
    for key in POSE_KEYS:
        arr.append([
            float(lm["pose"][key]["x"]),
            float(lm["pose"][key]["y"])
        ])

    # left hand
    for key in HAND_KEYS:
        arr.append([
            float(lm["left_hand"][key]["x"]),
            float(lm["left_hand"][key]["y"])
        ])

    # right hand
    for key in HAND_KEYS:
        arr.append([
            float(lm["right_hand"][key]["x"]),
            float(lm["right_hand"][key]["y"])
        ])

    # face
    for key in FACE_KEYS:
        arr.append([
            float(lm["face"][key]["x"]),
            float(lm["face"][key]["y"])
        ])

    arr = np.array(arr, dtype=np.float32)  # [122, 2]
    return arr


def json_to_npy_array(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    frames = data["frames"]

    if len(frames) != TARGET_FRAMES:
        raise ValueError(
            f"{os.path.basename(json_path)} punya {len(frames)} frame, expected {TARGET_FRAMES}"
        )

    arr = np.stack([frame_to_tensor(frame) for frame in frames], axis=0)  # [T, L, C]

    if arr.shape != (90, 122, 2):
        raise ValueError(
            f"Shape salah untuk {os.path.basename(json_path)}: {arr.shape}, expected (90, 122, 2)"
        )

    return data, arr


# =========================
# SAVE DATASET
# =========================

def save_index_map():
    output_file = os.path.join(OUTPUT_PATH, "index_map.json")

    readable_map = {
        str(k): v for k, v in INDEX_MAP.items()
    }

    with open(output_file, "w") as f:
        json.dump(readable_map, f, indent=2)

    print(f"Saved index map to: {output_file}")


def build_label_map(labels):
    labels = sorted(labels)
    return {label: idx for idx, label in enumerate(labels)}


def process_all():
    labels = [d for d in os.listdir(INPUT_PATH) if os.path.isdir(os.path.join(INPUT_PATH, d))]
    label_map = build_label_map(labels)

    dataset_index = []

    for label in labels:
        input_label_dir = os.path.join(INPUT_PATH, label)
        output_label_dir = os.path.join(OUTPUT_PATH, label)
        os.makedirs(output_label_dir, exist_ok=True)

        json_files = sorted([f for f in os.listdir(input_label_dir) if f.endswith(".json")])

        print(f"\nConverting label: {label} ({len(json_files)} files)")

        for file_name in tqdm(json_files):
            input_file = os.path.join(input_label_dir, file_name)

            data, arr = json_to_npy_array(input_file)

            video_id = data["metadata"].get("video_id", file_name.replace(".json", ""))
            npy_name = file_name.replace(".json", ".npy")
            output_file = os.path.join(output_label_dir, npy_name)

            np.save(output_file, arr)

            dataset_index.append({
                "video_id": video_id,
                "label": label,
                "label_id": label_map[label],
                "json_file": input_file,
                "npy_file": output_file,
                "shape": list(arr.shape)
            })

    # save label map
    label_map_file = os.path.join(OUTPUT_PATH, "label_map.json")
    with open(label_map_file, "w") as f:
        json.dump(label_map, f, indent=2)

    # save dataset index
    dataset_index_file = os.path.join(OUTPUT_PATH, "dataset_index.json")
    with open(dataset_index_file, "w") as f:
        json.dump(dataset_index, f, indent=2)

    print(f"\nSaved label map to: {label_map_file}")
    print(f"Saved dataset index to: {dataset_index_file}")


# =========================
# DEMO READ NPY
# =========================

def explain_index(L_idx):
    info = INDEX_MAP[L_idx]
    return f"L={L_idx} -> component={info['component']}, name={info['name']}"


def demo_read_one_npy():
    """
    Baca satu file npy pertama dan print contoh indexing.
    """
    first_npy = None
    for root, _, files in os.walk(OUTPUT_PATH):
        npy_files = sorted([f for f in files if f.endswith(".npy")])
        if npy_files:
            first_npy = os.path.join(root, npy_files[0])
            break

    if first_npy is None:
        print("Tidak ada file .npy untuk demo.")
        return

    arr = np.load(first_npy)

    print("\n=== DEMO READ NPY ===")
    print(f"File: {first_npy}")
    print(f"Shape: {arr.shape}")   # (90, 122, 2)

    print("\nInterpretasi shape:")
    print("  axis 0 = T (frame/time)")
    print("  axis 1 = L (landmark index)")
    print("  axis 2 = C (channel: 0=x, 1=y)")

    print("\nContoh indexing:")

    # Example 1
    t, l = 0, 0
    print(f"\narr[{t}, {l}] = {arr[t, l]}")
    print(f"Artinya: frame {t}, {explain_index(l)}")
    print(f"  x = arr[{t}, {l}, 0] = {arr[t, l, 0]}")
    print(f"  y = arr[{t}, {l}, 1] = {arr[t, l, 1]}")

    # Example 2
    t, l = 10, 5
    print(f"\narr[{t}, {l}] = {arr[t, l]}")
    print(f"Artinya: frame {t}, {explain_index(l)}")
    print(f"  x = arr[{t}, {l}, 0] = {arr[t, l, 0]}")
    print(f"  y = arr[{t}, {l}, 1] = {arr[t, l, 1]}")

    # Example 3
    t, l = 20, 12
    print(f"\narr[{t}, {l}] = {arr[t, l]}")
    print(f"Artinya: frame {t}, {explain_index(l)}")
    print(f"  x = arr[{t}, {l}, 0] = {arr[t, l, 0]}")
    print(f"  y = arr[{t}, {l}, 1] = {arr[t, l, 1]}")

    # Example 4
    t, l = 20, 33
    print(f"\narr[{t}, {l}] = {arr[t, l]}")
    print(f"Artinya: frame {t}, {explain_index(l)}")
    print(f"  x = arr[{t}, {l}, 0] = {arr[t, l, 0]}")
    print(f"  y = arr[{t}, {l}, 1] = {arr[t, l, 1]}")

    # Example 5
    t, l = 30, 54
    print(f"\narr[{t}, {l}] = {arr[t, l]}")
    print(f"Artinya: frame {t}, {explain_index(l)}")
    print(f"  x = arr[{t}, {l}, 0] = {arr[t, l, 0]}")
    print(f"  y = arr[{t}, {l}, 1] = {arr[t, l, 1]}")

    print("\nSubset contoh:")
    print("  arr[0].shape       =", arr[0].shape)        # (122, 2)
    print("  arr[:, 0].shape    =", arr[:, 0].shape)     # (90, 2)
    print("  arr[:, :, 0].shape =", arr[:, :, 0].shape)  # (90, 122)

    print("\nMakna subset:")
    print("  arr[0]        = semua landmark di frame pertama")
    print("  arr[:, 0]     = trajectory landmark L=0 sepanjang waktu")
    print("  arr[:, :, 0]  = semua nilai x untuk semua frame dan landmark")


def main():
    save_index_map()
    process_all()
    demo_read_one_npy()


if __name__ == "__main__":
    main()