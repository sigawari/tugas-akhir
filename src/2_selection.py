import os
import json
import numpy as np
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_PATH = os.path.join(BASE_DIR, "..", "dataset", "extracted")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "dataset", "selected")

os.makedirs(OUTPUT_PATH, exist_ok=True)

# =========================
# SELEKSI LANDMARK
# =========================

# Pose: 11–22 (upper body)
POSE_SELECTED = list(range(11, 23))

# Hands: full
HAND_SELECTED = list(range(21))

# ✅ Face 68 mapping (MediaPipe → dlib-style 68)
FACE_SELECTED = [
    162, 234, 93, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379,
    365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338,
    10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58,
    172, 136, 150, 149, 176, 148, 152,
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246,
    263, 249, 390, 373, 374, 380, 381, 382,
    362, 398, 384, 385, 386, 387, 388, 466,
    78, 95, 88, 178, 87, 14, 317, 402,
    318, 324, 308, 191, 80, 81, 82, 13,
    312, 311, 310, 415
]

# =========================
# HELPER
# =========================

def get_xy(lm):
    return [lm["x"], lm["y"]]


def process_file(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    frames = data["frames"]
    selected_sequence = []

    for frame in frames:
        lm = frame["landmarks"]
        frame_vec = []

        # ===== POSE =====
        pose = lm["pose"]
        pose_keys = list(pose.keys())

        for idx in POSE_SELECTED:
            frame_vec.extend(get_xy(pose[pose_keys[idx]]))

        # ===== LEFT HAND =====
        left_hand = lm["left_hand"]
        for idx in HAND_SELECTED:
            frame_vec.extend(get_xy(left_hand[str(idx)]))

        # ===== RIGHT HAND =====
        right_hand = lm["right_hand"]
        for idx in HAND_SELECTED:
            frame_vec.extend(get_xy(right_hand[str(idx)]))

        # ===== FACE (68) =====
        face = lm["face"]
        for idx in FACE_SELECTED:
            frame_vec.extend(get_xy(face[str(idx)]))

        selected_sequence.append(frame_vec)

    return np.array(selected_sequence)


# =========================
# MAIN
# =========================

def main():
    for label in os.listdir(INPUT_PATH):
        label_path = os.path.join(INPUT_PATH, label)

        if not os.path.isdir(label_path):
            continue

        save_dir = os.path.join(OUTPUT_PATH, label)
        os.makedirs(save_dir, exist_ok=True)

        files = [f for f in os.listdir(label_path) if f.endswith(".json")]

        print(f"\nProcessing {label} ({len(files)} files)")

        for file in tqdm(files):
            file_path = os.path.join(label_path, file)

            arr = process_file(file_path)

            save_path = os.path.join(save_dir, file.replace(".json", ".npy"))
            np.save(save_path, arr)


if __name__ == "__main__":
    main()