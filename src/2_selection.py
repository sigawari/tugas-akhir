import os
import json
import numpy as np
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_PATH = os.path.join(BASE_DIR, "..", "dataset", "extracted")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "dataset", "selected_json")

os.makedirs(OUTPUT_PATH, exist_ok=True)

# =========================
# SELEKSI LANDMARK
# =========================

# Pose: 11–22 (upper body)
POSE_SELECTED = list(range(11, 23))

# Hands: full
HAND_SELECTED = list(range(21))

# Nama pose sesuai urutan MediaPipe
POSE_NAMES = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

# Face 468 -> dlib-inspired 68
mp2dlib_correspondence = [
    [127], [234], [93], [132, 58], [58, 172], [136], [150], [176], [152],
    [400], [379], [365], [397, 288], [361], [323], [454], [356],

    [70], [63], [105], [66], [107],

    [336], [296], [334], [293], [300],

    [168, 6], [197, 195], [5], [4], [75], [97], [2], [326], [305],

    [33], [160], [158], [133], [153], [144],

    [362], [385], [387], [263], [373], [380],

    [61], [39], [37], [0], [267], [269], [291],

    [321], [314], [17], [84], [91],

    [78], [82], [13], [312], [308],

    [317], [14], [87],
]

for i in range(len(mp2dlib_correspondence)):
    if len(mp2dlib_correspondence[i]) == 1:
        idx = mp2dlib_correspondence[i][0]
        mp2dlib_correspondence[i] = [idx, idx]

# =========================
# HELPER
# =========================

def get_xy(point_dict):
    return {
        "x": float(point_dict.get("x", 0.0)),
        "y": float(point_dict.get("y", 0.0))
    }

def mean_face_points(face_dict, source_indices):
    pts = []
    for idx in source_indices:
        p = face_dict[str(idx)]
        pts.append([float(p.get("x", 0.0)), float(p.get("y", 0.0))])

    pts = np.array(pts, dtype=np.float32)
    mean_xy = pts.mean(axis=0)

    return {
        "source_indices": source_indices,
        "x": float(mean_xy[0]),
        "y": float(mean_xy[1])
    }

def select_frame_landmarks(frame):
    lm = frame["landmarks"]

    # Pose selected
    pose_selected = {}
    for idx in POSE_SELECTED:
        pose_name = POSE_NAMES[idx]
        pose_selected[pose_name] = get_xy(lm["pose"][pose_name])

    # Left hand full
    left_hand_selected = {}
    for idx in HAND_SELECTED:
        left_hand_selected[str(idx)] = get_xy(lm["left_hand"][str(idx)])

    # Right hand full
    right_hand_selected = {}
    for idx in HAND_SELECTED:
        right_hand_selected[str(idx)] = get_xy(lm["right_hand"][str(idx)])

    # Face 68 selected
    face_selected = {}
    for dlib_idx, source_indices in enumerate(mp2dlib_correspondence):
        face_selected[str(dlib_idx)] = mean_face_points(lm["face"], source_indices)

    return {
        "frame_index": frame["frame_index"],
        "timestamp_ms": frame["timestamp_ms"],
        "landmarks": {
            "pose": pose_selected,
            "left_hand": left_hand_selected,
            "right_hand": right_hand_selected,
            "face": face_selected
        }
    }

def process_json_file(input_file):
    with open(input_file, "r") as f:
        data = json.load(f)

    selected_frames = []
    for frame in data["frames"]:
        selected_frames.append(select_frame_landmarks(frame))

    output_data = {
        "metadata": {
            **data["metadata"],
            "selection_info": {
                "pose_indices": POSE_SELECTED,
                "left_hand_indices": HAND_SELECTED,
                "right_hand_indices": HAND_SELECTED,
                "face_total_selected": 68,
                "coordinate_dim": ["x", "y"],
                "face_mapping": "mediapipe_468_to_dlib_inspired_68"
            }
        },
        "frames": selected_frames
    }

    return output_data

# =========================
# MAIN
# =========================

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

            selected_data = process_json_file(input_file)

            with open(output_file, "w") as f:
                json.dump(selected_data, f, indent=2)

if __name__ == "__main__":
    main()