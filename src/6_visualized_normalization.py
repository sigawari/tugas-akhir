import os
import json
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BEFORE_PATH = os.path.join(BASE_DIR, "..", "dataset", "cleaned_json")
AFTER_PATH = os.path.join(BASE_DIR, "..", "dataset", "normalized_json")

EPS = 1e-6

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

# =========================
# SKELETON CONNECTIONS
# =========================

POSE_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_wrist", "left_thumb"),
    ("left_wrist", "left_index"),
    ("left_wrist", "left_pinky"),
    ("right_wrist", "right_thumb"),
    ("right_wrist", "right_index"),
    ("right_wrist", "right_pinky"),
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17)
]

# dlib-68 style face connections
FACE_CONNECTIONS = []
# jawline 0-16
FACE_CONNECTIONS += [(i, i + 1) for i in range(0, 16)]
# right eyebrow 17-21
FACE_CONNECTIONS += [(i, i + 1) for i in range(17, 21)]
# left eyebrow 22-26
FACE_CONNECTIONS += [(i, i + 1) for i in range(22, 26)]
# nose bridge 27-30
FACE_CONNECTIONS += [(i, i + 1) for i in range(27, 30)]
# lower nose 31-35
FACE_CONNECTIONS += [(31, 32), (32, 33), (33, 34), (34, 35)]
# right eye 36-41 closed loop
FACE_CONNECTIONS += [(36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36)]
# left eye 42-47 closed loop
FACE_CONNECTIONS += [(42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42)]
# outer lip 48-59 closed loop
FACE_CONNECTIONS += [(48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54),
                     (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48)]
# inner lip 60-67 closed loop
FACE_CONNECTIONS += [(60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66),
                     (66, 67), (67, 60)]


# =========================
# BASIC IO
# =========================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def find_first_json(root_dir, preferred_label=None):
    if preferred_label is not None:
        label_dir = os.path.join(root_dir, preferred_label)
        if os.path.isdir(label_dir):
            files = sorted([f for f in os.listdir(label_dir) if f.endswith(".json")])
            if files:
                return os.path.join(label_dir, files[0])

    for root, _, files in os.walk(root_dir):
        json_files = sorted([f for f in files if f.endswith(".json")])
        if json_files:
            return os.path.join(root, json_files[0])

    return None


def pair_before_after(preferred_label=None):
    before_file = find_first_json(BEFORE_PATH, preferred_label=preferred_label)
    if before_file is None:
        raise FileNotFoundError("Tidak ada file JSON di cleaned_json")

    rel_path = os.path.relpath(before_file, BEFORE_PATH)
    after_file = os.path.join(AFTER_PATH, rel_path)

    if not os.path.exists(after_file):
        raise FileNotFoundError(f"File pasangan di normalized_json tidak ditemukan:\n{after_file}")

    return before_file, after_file


# =========================
# ARRAY HELPERS
# =========================

def frame_to_arrays(frame):
    lm = frame["landmarks"]

    pose = np.array([[lm["pose"][k]["x"], lm["pose"][k]["y"]] for k in POSE_KEYS], dtype=np.float32)
    left_hand = np.array([[lm["left_hand"][k]["x"], lm["left_hand"][k]["y"]] for k in HAND_KEYS], dtype=np.float32)
    right_hand = np.array([[lm["right_hand"][k]["x"], lm["right_hand"][k]["y"]] for k in HAND_KEYS], dtype=np.float32)
    face = np.array([[lm["face"][k]["x"], lm["face"][k]["y"]] for k in FACE_KEYS], dtype=np.float32)

    return pose, left_hand, right_hand, face


def collect_all_coords(data):
    coords = []
    for frame in data["frames"]:
        pose, lh, rh, face = frame_to_arrays(frame)
        coords.extend([pose, lh, rh, face])
    return np.concatenate(coords, axis=0)


def collect_global_normalized_coords(data):
    coords = []
    for frame in data["frames"]:
        pose, lh, rh, face = frame_to_arrays(frame)
        pose, lh, rh, face = full_body_normalize_frame(pose, lh, rh, face)
        coords.extend([pose, lh, rh, face])
    return np.concatenate(coords, axis=0)


def print_coordinate_range(title, coords):
    print(f"\n=== {title} ===")
    print(f"x_min = {coords[:,0].min():.6f}, x_max = {coords[:,0].max():.6f}")
    print(f"y_min = {coords[:,1].min():.6f}, y_max = {coords[:,1].max():.6f}")


# =========================
# DRAW HELPERS
# =========================

def draw_pose(ax, pose_arr, alpha=0.8):
    key_to_idx = {k: i for i, k in enumerate(POSE_KEYS)}
    for a, b in POSE_CONNECTIONS:
        ia, ib = key_to_idx[a], key_to_idx[b]
        x = [pose_arr[ia, 0], pose_arr[ib, 0]]
        y = [pose_arr[ia, 1], pose_arr[ib, 1]]
        ax.plot(x, y, alpha=alpha)


def draw_hand(ax, hand_arr, alpha=0.8):
    if np.all(hand_arr == 0):
        return
    for i, j in HAND_CONNECTIONS:
        x = [hand_arr[i, 0], hand_arr[j, 0]]
        y = [hand_arr[i, 1], hand_arr[j, 1]]
        ax.plot(x, y, alpha=alpha)


def draw_face(ax, face_arr, alpha=0.35):
    for i, j in FACE_CONNECTIONS:
        x = [face_arr[i, 0], face_arr[j, 0]]
        y = [face_arr[i, 1], face_arr[j, 1]]
        ax.plot(x, y, alpha=alpha, linewidth=0.8)


def scatter_with_lines(ax, pose, lh, rh, face, title):
    ax.scatter(face[:, 0], face[:, 1], s=10, alpha=0.5, label="Face")
    ax.scatter(pose[:, 0], pose[:, 1], s=30, label="Pose")
    ax.scatter(lh[:, 0], lh[:, 1], s=20, label="Left hand")
    ax.scatter(rh[:, 0], rh[:, 1], s=20, label="Right hand")

    draw_face(ax, face)
    draw_pose(ax, pose)
    draw_hand(ax, lh)
    draw_hand(ax, rh)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.axis("equal")
    ax.invert_yaxis()
    ax.legend()


# =========================
# GLOBAL FULL-BODY ONLY NORMALIZATION
# =========================

def full_body_normalize_frame(pose, left_hand, right_hand, face):
    """
    Hanya untuk visualisasi global utuh:
    - neck = midpoint shoulder
    - scale = shoulder distance
    - hand kosong tetap 0
    """
    pose_out = pose.copy()
    lh_out = left_hand.copy()
    rh_out = right_hand.copy()
    face_out = face.copy()

    left_shoulder = pose_out[POSE_KEYS.index("left_shoulder")]
    right_shoulder = pose_out[POSE_KEYS.index("right_shoulder")]

    neck = (left_shoulder + right_shoulder) / 2.0
    shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
    shoulder_dist = max(float(shoulder_dist), EPS)

    lh_abs_mask = np.logical_and(lh_out[:, 0] == 0.0, lh_out[:, 1] == 0.0)
    rh_abs_mask = np.logical_and(rh_out[:, 0] == 0.0, rh_out[:, 1] == 0.0)

    pose_out = (pose_out - neck) / shoulder_dist
    face_out = (face_out - neck) / shoulder_dist
    lh_out = (lh_out - neck) / shoulder_dist
    rh_out = (rh_out - neck) / shoulder_dist

    lh_out[lh_abs_mask] = 0.0
    rh_out[rh_abs_mask] = 0.0

    return pose_out, lh_out, rh_out, face_out


# =========================
# TRACKS
# =========================

def get_landmark_track(data, component, key):
    xs, ys = [], []
    for frame in data["frames"]:
        p = frame["landmarks"][component][key]
        xs.append(float(p["x"]))
        ys.append(float(p["y"]))
    return np.array(xs), np.array(ys)


# =========================
# PLOTTING
# =========================

def plot_global_before_after(before_data, frame_idx, title_prefix=""):
    before_frame = before_data["frames"][frame_idx]
    b_pose, b_lh, b_rh, b_face = frame_to_arrays(before_frame)
    a_pose, a_lh, a_rh, a_face = full_body_normalize_frame(b_pose, b_lh, b_rh, b_face)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    scatter_with_lines(
        axes[0], b_pose, b_lh, b_rh, b_face,
        f"{title_prefix}\nBEFORE (cleaned) - frame {frame_idx}"
    )
    scatter_with_lines(
        axes[1], a_pose, a_lh, a_rh, a_face,
        f"{title_prefix}\nAFTER GLOBAL BODY NORMALIZATION - frame {frame_idx}"
    )

    plt.tight_layout()
    plt.show()


def plot_component_before_after(before_data, after_data, frame_idx, title_prefix=""):
    before_frame = before_data["frames"][frame_idx]
    after_frame = after_data["frames"][frame_idx]

    b_pose, b_lh, b_rh, b_face = frame_to_arrays(before_frame)
    a_pose, a_lh, a_rh, a_face = frame_to_arrays(after_frame)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    # before
    axes[0, 0].scatter(b_pose[:, 0], b_pose[:, 1], s=20)
    draw_pose(axes[0, 0], b_pose)
    axes[0, 0].set_title("Pose BEFORE")

    axes[0, 1].scatter(b_face[:, 0], b_face[:, 1], s=12)
    draw_face(axes[0, 1], b_face)
    axes[0, 1].set_title("Face BEFORE")

    axes[0, 2].scatter(b_lh[:, 0], b_lh[:, 1], s=20)
    draw_hand(axes[0, 2], b_lh)
    axes[0, 2].set_title("Left hand BEFORE")

    axes[0, 3].scatter(b_rh[:, 0], b_rh[:, 1], s=20)
    draw_hand(axes[0, 3], b_rh)
    axes[0, 3].set_title("Right hand BEFORE")

    # after
    axes[1, 0].scatter(a_pose[:, 0], a_pose[:, 1], s=20)
    draw_pose(axes[1, 0], a_pose)
    axes[1, 0].set_title("Pose AFTER FINAL NORMALIZATION")

    axes[1, 1].scatter(a_face[:, 0], a_face[:, 1], s=12)
    draw_face(axes[1, 1], a_face)
    axes[1, 1].set_title("Face AFTER FINAL NORMALIZATION")

    axes[1, 2].scatter(a_lh[:, 0], a_lh[:, 1], s=20)
    draw_hand(axes[1, 2], a_lh)
    axes[1, 2].set_title("Left hand AFTER FINAL NORMALIZATION")

    axes[1, 3].scatter(a_rh[:, 0], a_rh[:, 1], s=20)
    draw_hand(axes[1, 3], a_rh)
    axes[1, 3].set_title("Right hand AFTER FINAL NORMALIZATION")

    for row in axes:
        for ax in row:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(True)
            ax.axis("equal")
            ax.invert_yaxis()

    fig.suptitle(f"{title_prefix} - frame {frame_idx}", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_trajectory_comparison(before_data, after_data, component, key, title):
    bx, by = get_landmark_track(before_data, component, key)
    ax_, ay_ = get_landmark_track(after_data, component, key)

    plt.figure(figsize=(7, 7))
    plt.plot(bx, by, marker="o", markersize=2, linewidth=1, label="Before")
    plt.plot(ax_, ay_, marker="o", markersize=2, linewidth=1, label="After")
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


# =========================
# MAIN
# =========================

def main():
    preferred_label = None
    mode = "global"   # "global" atau "component"

    before_file, after_file = pair_before_after(preferred_label=preferred_label)

    print("\nUsing files:")
    print("BEFORE :", before_file)
    print("AFTER  :", after_file)

    before_data = load_json(before_file)
    after_data = load_json(after_file)

    frame_idx = len(before_data["frames"]) // 2
    title_prefix = os.path.basename(before_file)

    before_coords = collect_all_coords(before_data)
    print_coordinate_range("BEFORE CLEANED", before_coords)

    if mode == "global":
        global_after_coords = collect_global_normalized_coords(before_data)
        print_coordinate_range("AFTER GLOBAL BODY NORMALIZATION", global_after_coords)
        plot_global_before_after(before_data, frame_idx=frame_idx, title_prefix=title_prefix)

    elif mode == "component":
        final_after_coords = collect_all_coords(after_data)
        print_coordinate_range("AFTER FINAL NORMALIZATION", final_after_coords)
        plot_component_before_after(before_data, after_data, frame_idx=frame_idx, title_prefix=title_prefix)

        plot_trajectory_comparison(
            before_data, after_data,
            component="pose",
            key="right_wrist",
            title="Trajectory comparison - right_wrist"
        )
        plot_trajectory_comparison(
            before_data, after_data,
            component="pose",
            key="left_wrist",
            title="Trajectory comparison - left_wrist"
        )

    else:
        raise ValueError("mode harus 'global' atau 'component'")


if __name__ == "__main__":
    main()