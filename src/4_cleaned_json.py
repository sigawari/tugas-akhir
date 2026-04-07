import os
import json
import numpy as np
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_PATH = os.path.join(BASE_DIR, "..", "dataset", "selected_json")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "dataset", "cleaned_json")

os.makedirs(OUTPUT_PATH, exist_ok=True)

TARGET_FRAMES = 90

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


def is_missing_xy(point_dict):
    return float(point_dict.get("x", 0.0)) == 0.0 and float(point_dict.get("y", 0.0)) == 0.0


def crop_frames(frames, target_frames=90):
    """
    Potong frame jika lebih panjang dari target.
    Jika kurang/equal, dibiarkan apa adanya.
    """
    if len(frames) > target_frames:
        return frames[:target_frames]
    return frames


def interpolate_series(values):
    """
    Interpolasi linear untuk 1 deret waktu.
    Missing ditandai oleh nilai 0.0 pada pasangan (x,y), tapi fungsi ini
    dipanggil setelah mask missing dibuat di level titik.
    values: np.array shape [T]
    valid_mask: ditangani dari luar
    """
    raise NotImplementedError("Gunakan interpolate_xy_track(), bukan fungsi ini langsung.")


def interpolate_xy_track(track_xy):
    """
    track_xy: np.array shape [T, 2] untuk 1 landmark sepanjang waktu.

    Aturan:
    - kalau semua nol -> biarkan
    - kalau ada valid -> interpolasi linear per channel
    - missing di awal/akhir diisi nearest valid oleh np.interp
    """
    T = track_xy.shape[0]
    result = track_xy.copy()

    missing_mask = np.logical_and(result[:, 0] == 0.0, result[:, 1] == 0.0)
    valid_mask = ~missing_mask

    # Seluruh sequence kosong -> jangan diubah
    if valid_mask.sum() == 0:
        return result

    valid_idx = np.where(valid_mask)[0]

    for c in range(2):  # x, y
        series = result[:, c]
        valid_values = series[valid_mask]

        interpolated = np.interp(
            np.arange(T),
            valid_idx,
            valid_values
        )
        series[missing_mask] = interpolated[missing_mask]
        result[:, c] = series

    return result


def component_to_track_array(frames, component_name, keys):
    """
    Convert frames -> np.array [T, K, 2]
    """
    T = len(frames)
    K = len(keys)
    arr = np.zeros((T, K, 2), dtype=np.float32)

    for t, frame in enumerate(frames):
        comp = frame["landmarks"][component_name]
        for k_idx, key in enumerate(keys):
            arr[t, k_idx, 0] = float(comp[key]["x"])
            arr[t, k_idx, 1] = float(comp[key]["y"])

    return arr


def write_component_back(frames, component_name, keys, arr):
    """
    Write np.array [T, K, 2] back to frames JSON structure.
    source_indices untuk face dipertahankan kalau ada.
    """
    T = len(frames)

    for t in range(T):
        for k_idx, key in enumerate(keys):
            frames[t]["landmarks"][component_name][key]["x"] = float(arr[t, k_idx, 0])
            frames[t]["landmarks"][component_name][key]["y"] = float(arr[t, k_idx, 1])


def interpolate_hand_component(frames, component_name):
    """
    Interpolasi hanya untuk hand.
    """
    hand_arr = component_to_track_array(frames, component_name, HAND_KEYS)  # [T, 21, 2]

    for k in range(hand_arr.shape[1]):
        hand_arr[:, k, :] = interpolate_xy_track(hand_arr[:, k, :])

    write_component_back(frames, component_name, HAND_KEYS, hand_arr)
    return frames


def update_metadata(metadata, original_num_frames, cleaned_num_frames):
    metadata = dict(metadata)
    metadata["original_total_frames_after_selection"] = original_num_frames
    metadata["total_frames"] = cleaned_num_frames
    metadata["duration_sec"] = cleaned_num_frames / metadata["fps"] if metadata.get("fps", 0) else 0
    metadata["preprocessing"] = {
        "target_frames": TARGET_FRAMES,
        "cropping_applied": original_num_frames > TARGET_FRAMES,
        "missing_handling": {
            "pose": "none",
            "face": "none",
            "left_hand": "linear_interpolation_if_partial_missing_keep_zero_if_all_missing",
            "right_hand": "linear_interpolation_if_partial_missing_keep_zero_if_all_missing"
        }
    }
    return metadata


def process_file(input_file):
    with open(input_file, "r") as f:
        data = json.load(f)

    frames = data["frames"]
    original_num_frames = len(frames)

    # 1. Crop ke 90 kalau lebih panjang
    frames = crop_frames(frames, TARGET_FRAMES)

    # 2. Interpolasi hanya untuk hand
    frames = interpolate_hand_component(frames, "left_hand")
    frames = interpolate_hand_component(frames, "right_hand")

    # 3. Re-index frame_index dan timestamp_ms agar konsisten setelah crop
    fps = data["metadata"].get("fps", 0)
    for i, frame in enumerate(frames):
        frame["frame_index"] = i
        frame["timestamp_ms"] = int(i * (1000 / fps)) if fps > 0 else 0

    output_data = {
        "metadata": update_metadata(
            data["metadata"],
            original_num_frames=original_num_frames,
            cleaned_num_frames=len(frames)
        ),
        "frames": frames
    }

    return output_data


def main():
    for label in os.listdir(INPUT_PATH):
        label_input_dir = os.path.join(INPUT_PATH, label)
        if not os.path.isdir(label_input_dir):
            continue

        label_output_dir = os.path.join(OUTPUT_PATH, label)
        os.makedirs(label_output_dir, exist_ok=True)

        json_files = [f for f in os.listdir(label_input_dir) if f.endswith(".json")]

        print(f"\nProcessing label: {label} ({len(json_files)} files)")

        for file_name in tqdm(json_files):
            input_file = os.path.join(label_input_dir, file_name)
            output_file = os.path.join(label_output_dir, file_name)

            cleaned_data = process_file(input_file)

            with open(output_file, "w") as f:
                json.dump(cleaned_data, f, indent=2)


if __name__ == "__main__":
    main()