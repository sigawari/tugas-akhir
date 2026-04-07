import os
import json
import numpy as np
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_PATH = os.path.join(BASE_DIR, "..", "dataset", "cleaned_json")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "dataset", "normalized_json")

os.makedirs(OUTPUT_PATH, exist_ok=True)

TARGET_FRAMES = 90
EPS = 1e-6

# =========================
# LANDMARK CONFIG
# =========================

POSE_KEYS = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb"
]

LEFT_ARM_KEYS = ["left_elbow", "left_wrist", "left_pinky", "left_index", "left_thumb"]
RIGHT_ARM_KEYS = ["right_elbow", "right_wrist", "right_pinky", "right_index", "right_thumb"]

HAND_KEYS = [str(i) for i in range(21)]
FACE_KEYS = [str(i) for i in range(68)]

# Face-68 nose anchor.
# Default: dlib point 30 => 0-based index 29
FACE_NOSE_INDEX = 29


# =========================
# HELPER
# =========================

def crop_frames(frames, target_frames=90):
    if len(frames) > target_frames:
        return frames[:target_frames]
    return frames


def to_arr(component_dict, keys):
    return np.array(
        [[float(component_dict[k]["x"]), float(component_dict[k]["y"])] for k in keys],
        dtype=np.float32
    )


def write_arr(component_dict, keys, arr):
    for i, k in enumerate(keys):
        component_dict[k]["x"] = float(arr[i, 0])
        component_dict[k]["y"] = float(arr[i, 1])


def get_absent_mask(hand_arr):
    return np.logical_and(hand_arr[:, 0] == 0.0, hand_arr[:, 1] == 0.0)


# =========================
# NORMALIZATION STEPS
# =========================

def full_body_normalize(pose_arr, face_arr, lh_arr, rh_arr, lh_abs_mask, rh_abs_mask):
    """
    Full body normalization:
    - neck = avg(left_shoulder, right_shoulder)
    - scale = distance between shoulders
    """
    left_shoulder = pose_arr[POSE_KEYS.index("left_shoulder")]
    right_shoulder = pose_arr[POSE_KEYS.index("right_shoulder")]

    neck = (left_shoulder + right_shoulder) / 2.0

    shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
    shoulder_dist = max(float(shoulder_dist), EPS)

    pose_norm = (pose_arr - neck) / shoulder_dist
    face_norm = (face_arr - neck) / shoulder_dist
    lh_norm = (lh_arr - neck) / shoulder_dist
    rh_norm = (rh_arr - neck) / shoulder_dist

    # restore absent hand points
    lh_norm[lh_abs_mask] = 0.0
    rh_norm[rh_abs_mask] = 0.0

    return pose_norm, face_norm, lh_norm, rh_norm


def face_normalize(face_arr):
    """
    Face normalization relative to nose anchor
    """
    nose = face_arr[FACE_NOSE_INDEX].copy()
    return face_arr - nose


def arm_normalize(pose_arr):
    """
    Arm normalization using shoulder-elbow segment length
    Applied separately for left and right arm
    """
    pose_norm = pose_arr.copy()

    # Left arm
    ls = pose_norm[POSE_KEYS.index("left_shoulder")]
    le = pose_norm[POSE_KEYS.index("left_elbow")]
    left_scale = np.linalg.norm(ls - le)
    left_scale = max(float(left_scale), EPS)

    for key in LEFT_ARM_KEYS:
        idx = POSE_KEYS.index(key)
        pose_norm[idx] = pose_norm[idx] / left_scale

    # Right arm
    rs = pose_norm[POSE_KEYS.index("right_shoulder")]
    re = pose_norm[POSE_KEYS.index("right_elbow")]
    right_scale = np.linalg.norm(rs - re)
    right_scale = max(float(right_scale), EPS)

    for key in RIGHT_ARM_KEYS:
        idx = POSE_KEYS.index(key)
        pose_norm[idx] = pose_norm[idx] / right_scale

    return pose_norm


def hand_bbox_normalize(hand_arr, absent_mask):
    """
    Hand normalization using bounding box:
    - compute bbox only from valid points
    - absent points stay (0,0)
    """
    out = hand_arr.copy()

    # entire hand absent
    if absent_mask.all():
        out[:] = 0.0
        return out

    valid_mask = ~absent_mask
    valid_points = out[valid_mask]

    x_min = float(valid_points[:, 0].min())
    x_max = float(valid_points[:, 0].max())
    y_min = float(valid_points[:, 1].min())
    y_max = float(valid_points[:, 1].max())

    width = max(x_max - x_min, EPS)
    height = max(y_max - y_min, EPS)

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0

    out[:, 0] = (out[:, 0] - x_center) / width
    out[:, 1] = (out[:, 1] - y_center) / height

    # restore absent points
    out[absent_mask] = 0.0

    return out


# =========================
# MAIN PROCESS
# =========================

def process_file(input_file):
    with open(input_file, "r") as f:
        data = json.load(f)

    frames = crop_frames(data["frames"], TARGET_FRAMES)

    for t, frame in enumerate(frames):
        landmarks = frame["landmarks"]

        pose_dict = landmarks["pose"]
        face_dict = landmarks["face"]
        left_hand_dict = landmarks["left_hand"]
        right_hand_dict = landmarks["right_hand"]

        pose_arr = to_arr(pose_dict, POSE_KEYS)
        face_arr = to_arr(face_dict, FACE_KEYS)
        left_hand_arr = to_arr(left_hand_dict, HAND_KEYS)
        right_hand_arr = to_arr(right_hand_dict, HAND_KEYS)

        # detect absent hand points BEFORE normalization
        left_abs_mask = get_absent_mask(left_hand_arr)
        right_abs_mask = get_absent_mask(right_hand_arr)

        # 1. full body normalization
        pose_arr, face_arr, left_hand_arr, right_hand_arr = full_body_normalize(
            pose_arr,
            face_arr,
            left_hand_arr,
            right_hand_arr,
            left_abs_mask,
            right_abs_mask
        )

        # 2. face normalization
        face_arr = face_normalize(face_arr)

        # 3. arm normalization
        pose_arr = arm_normalize(pose_arr)

        # 4. hand normalization
        left_hand_arr = hand_bbox_normalize(left_hand_arr, left_abs_mask)
        right_hand_arr = hand_bbox_normalize(right_hand_arr, right_abs_mask)

        # write back
        write_arr(pose_dict, POSE_KEYS, pose_arr)
        write_arr(face_dict, FACE_KEYS, face_arr)
        write_arr(left_hand_dict, HAND_KEYS, left_hand_arr)
        write_arr(right_hand_dict, HAND_KEYS, right_hand_arr)

        # re-index frame info biar tetap rapi
        frame["frame_index"] = t
        fps = data["metadata"].get("fps", 0)
        frame["timestamp_ms"] = int(t * (1000 / fps)) if fps > 0 else 0

    metadata = dict(data.get("metadata", {}))
    metadata["total_frames"] = len(frames)
    metadata["duration_sec"] = len(frames) / metadata["fps"] if metadata.get("fps", 0) else 0.0
    metadata["normalization"] = {
        "method": "reference_based_normalization",
        "full_body_anchor": "neck_from_shoulders",
        "full_body_scale": "shoulder_distance",
        "face_anchor_index": FACE_NOSE_INDEX,
        "face_anchor_note": "0-based index on selected 68 face points",
        "arm_scale": "shoulder_elbow_distance_per_side",
        "hand_method": "bounding_box_per_frame",
        "absent_hand_rule": "keep_zero_and_exclude_from_bbox"
    }

    return {
        "metadata": metadata,
        "frames": frames
    }


def main():
    for label in os.listdir(INPUT_PATH):
        label_input_dir = os.path.join(INPUT_PATH, label)
        if not os.path.isdir(label_input_dir):
            continue

        label_output_dir = os.path.join(OUTPUT_PATH, label)
        os.makedirs(label_output_dir, exist_ok=True)

        json_files = [f for f in os.listdir(label_input_dir) if f.endswith(".json")]

        print(f"\nNormalizing label: {label} ({len(json_files)} files)")

        for file_name in tqdm(json_files):
            input_file = os.path.join(label_input_dir, file_name)
            output_file = os.path.join(label_output_dir, file_name)

            normalized_data = process_file(input_file)

            with open(output_file, "w") as f:
                json.dump(normalized_data, f, indent=2)

    print("\nNormalization complete.")
    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()