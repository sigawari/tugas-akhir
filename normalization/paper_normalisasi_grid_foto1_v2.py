#!/usr/bin/env python3
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-8
CANON_XLIM = (-1.5, 1.5)
CANON_YLIM = (1.5, -1.5)

FACE_POSE_IDXS = list(range(11))
POSE_IDXS = list(range(23))
ARM_SET_A = [14, 16, 18, 20, 22]
ARM_SET_B = [13, 15, 17, 19, 21]
ARM_SET_A_ANCHOR = [12] + ARM_SET_A
ARM_SET_B_ANCHOR = [11] + ARM_SET_B

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (0, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
]
FACE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (0, 8),
    (9, 10),
]
ARM_A_CONNECTIONS = [(12, 14), (14, 16), (16, 18), (16, 20), (16, 22)]
ARM_B_CONNECTIONS = [(11, 13), (13, 15), (15, 17), (15, 19), (15, 21)]
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def point_valid(p):
    return not (np.isclose(p[0], 0.0, atol=EPS) and np.isclose(p[1], 0.0, atol=EPS))


def valid_mask(points):
    points = np.asarray(points)
    return ~(
        np.isclose(points[:, 0], 0.0, atol=EPS) &
        np.isclose(points[:, 1], 0.0, atol=EPS)
    )


def safe_distance(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def load_image_with_os(filename="foto1.png"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(base_dir, filename)
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"File '{filename}' tidak ditemukan. Taruh file itu di folder yang sama dengan script.\n{image_path}"
        )

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Gagal membaca gambar: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return image_path, rgb


def extract_landmarks(rgb):
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise ImportError(
            "mediapipe belum terpasang. Install dulu:\n"
            "pip install mediapipe opencv-python matplotlib numpy"
        ) from exc

    pose = np.zeros((23, 2), dtype=np.float32)
    left_hand = np.zeros((21, 2), dtype=np.float32)
    right_hand = np.zeros((21, 2), dtype=np.float32)

    with mp.solutions.holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        refine_face_landmarks=False,
        enable_segmentation=False,
    ) as holistic:
        results = holistic.process(rgb)

    if results.pose_landmarks is not None:
        for i in range(23):
            lm = results.pose_landmarks.landmark[i]
            pose[i] = [lm.x, lm.y]

    if results.left_hand_landmarks is not None:
        for i, lm in enumerate(results.left_hand_landmarks.landmark[:21]):
            left_hand[i] = [lm.x, lm.y]

    if results.right_hand_landmarks is not None:
        for i, lm in enumerate(results.right_hand_landmarks.landmark[:21]):
            right_hand[i] = [lm.x, lm.y]

    return pose, left_hand, right_hand


def full_body_normalize(pose):
    pose = np.asarray(pose, dtype=np.float32)
    out = np.zeros_like(pose)

    if not (point_valid(pose[11]) and point_valid(pose[12])):
        return pose.copy()

    neck = (pose[11] + pose[12]) / 2.0
    shoulder_dist = safe_distance(pose[11], pose[12])
    if shoulder_dist < EPS:
        return pose.copy()

    mask = valid_mask(pose)
    out[mask] = (pose[mask] - neck) / shoulder_dist
    return out


def face_normalize(pose_after_full):
    out = np.asarray(pose_after_full, dtype=np.float32).copy()
    if not point_valid(out[0]):
        return out
    nose = out[0].copy()
    for idx in FACE_POSE_IDXS:
        if point_valid(out[idx]):
            out[idx] = out[idx] - nose
    return out


def scale_pose_subset(pose, subset, ref_a, ref_b):
    out = np.asarray(pose, dtype=np.float32).copy()
    if not (point_valid(out[ref_a]) and point_valid(out[ref_b])):
        return out
    scale = safe_distance(out[ref_a], out[ref_b])
    if scale < EPS:
        return out
    for idx in subset:
        if point_valid(out[idx]):
            out[idx] = out[idx] / scale
    return out


def arm_normalize(pose_after_face):
    out = np.asarray(pose_after_face, dtype=np.float32).copy()
    out = scale_pose_subset(out, ARM_SET_A, 12, 14)
    out = scale_pose_subset(out, ARM_SET_B, 11, 13)
    return out


def hand_bbox_normalize(hand):
    hand = np.asarray(hand, dtype=np.float32)
    out = np.zeros_like(hand)
    mask = valid_mask(hand)
    valid = hand[mask]
    if len(valid) == 0:
        return out

    x_min, x_max = float(valid[:, 0].min()), float(valid[:, 0].max())
    y_min, y_max = float(valid[:, 1].min()), float(valid[:, 1].max())
    x_center, y_center = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    width = max(x_max - x_min, EPS)
    height = max(y_max - y_min, EPS)

    out[mask, 0] = (hand[mask, 0] - x_center) / width
    out[mask, 1] = (hand[mask, 1] - y_center) / height
    return out


def preprocess_by_paper(pose, left_hand, right_hand):
    pose_full = full_body_normalize(pose)
    pose_face = face_normalize(pose_full)
    pose_arm = arm_normalize(pose_face)
    left_norm = hand_bbox_normalize(left_hand)
    right_norm = hand_bbox_normalize(right_hand)
    return {
        "pose_raw": pose,
        "pose_full": pose_full,
        "pose_face": pose_face,
        "pose_arm": pose_arm,
        "left_raw": left_hand,
        "left_norm": left_norm,
        "right_raw": right_hand,
        "right_norm": right_norm,
    }


def plot_points_lines(ax, points, connections, color, label=None, allowed_idx=None, image_shape=None):
    points = np.asarray(points, dtype=np.float32)
    mask = valid_mask(points)

    if allowed_idx is None:
        allowed_idx = list(range(len(points)))
    allowed_set = set(allowed_idx)

    if image_shape is not None:
        h, w = image_shape[:2]
        xs = points[:, 0] * w
        ys = points[:, 1] * h
    else:
        xs = points[:, 0]
        ys = points[:, 1]

    idxs = [i for i in allowed_idx if mask[i]]
    if idxs:
        ax.scatter(xs[idxs], ys[idxs], c=color, s=18, label=label, zorder=3)

    for a, b in connections:
        if a in allowed_set and b in allowed_set and mask[a] and mask[b]:
            ax.plot([xs[a], xs[b]], [ys[a], ys[b]], color=color, linewidth=2, alpha=0.95, zorder=2)


def set_fixed_limits(ax):
    ax.set_xlim(*CANON_XLIM)
    ax.set_ylim(*CANON_YLIM)


def style_coord_ax(ax, title):
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axhline(0, color="gray", linewidth=0.7, alpha=0.6)
    ax.axvline(0, color="gray", linewidth=0.7, alpha=0.6)


def annotate_no_data(ax, points, allowed_idx=None):
    mask = valid_mask(points)
    if allowed_idx is None:
        ok = bool(mask.any())
    else:
        idx_arr = np.array(list(allowed_idx), dtype=int) if len(allowed_idx) > 0 else np.array([], dtype=int)
        ok = bool(mask[idx_arr].any()) if len(idx_arr) > 0 else False
    if not ok:
        ax.text(0.5, 0.5, "Landmark tidak terdeteksi", ha="center", va="center", transform=ax.transAxes)


def overlay_before_after(ax, before, after, connections, title, allowed_idx=None):
    plot_points_lines(ax, before, connections, color="green", label="sebelum", allowed_idx=allowed_idx)
    plot_points_lines(ax, after, connections, color="purple", label="sesudah", allowed_idx=allowed_idx)
    set_fixed_limits(ax)
    style_coord_ax(ax, title)
    annotate_no_data(ax, before, allowed_idx=allowed_idx)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8, loc="best")


def normalized_to_canvas(points, out_shape, xlim=CANON_XLIM, ylim=CANON_YLIM):
    points = np.asarray(points, dtype=np.float32)
    h, w = out_shape[:2]
    xs = (points[:, 0] - xlim[0]) / (xlim[1] - xlim[0]) * (w - 1)
    ys = (points[:, 1] - ylim[0]) / (ylim[1] - ylim[0]) * (h - 1)
    return np.column_stack([xs, ys]).astype(np.float32)


def warp_rgb_to_canonical(rgb, pose_raw, out_size=720, xlim=CANON_XLIM, ylim=CANON_YLIM):
    pose_raw = np.asarray(pose_raw, dtype=np.float32)
    if not (point_valid(pose_raw[11]) and point_valid(pose_raw[12])):
        return rgb.copy(), {"ok": False, "reason": "invalid_shoulders"}

    neck = (pose_raw[11] + pose_raw[12]) / 2.0
    shoulder_dist = safe_distance(pose_raw[11], pose_raw[12])
    if shoulder_dist < EPS:
        return rgb.copy(), {"ok": False, "reason": "shoulder_dist_too_small"}

    in_h, in_w = rgb.shape[:2]
    out_h = out_size
    out_w = out_size

    xv = np.linspace(xlim[0], xlim[1], out_w, dtype=np.float32)
    yv = np.linspace(ylim[0], ylim[1], out_h, dtype=np.float32)
    xx, yy = np.meshgrid(xv, yv)

    src_x_norm = xx * shoulder_dist + neck[0]
    src_y_norm = yy * shoulder_dist + neck[1]

    map_x = (src_x_norm * in_w).astype(np.float32)
    map_y = (src_y_norm * in_h).astype(np.float32)

    rgb_can = cv2.remap(
        rgb,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    return rgb_can, {
        "ok": True,
        "neck": neck.tolist(),
        "shoulder_dist": float(shoulder_dist),
        "out_size": int(out_size),
    }


def draw_canonical_rgb_with_skeleton(ax, rgb_can, pose_norm, left_norm=None, right_norm=None, title="RGB + skeleton sesudah"):
    ax.imshow(rgb_can)
    pose_canvas = normalized_to_canvas(pose_norm, rgb_can.shape)
    plot_points_lines(ax, pose_canvas, POSE_CONNECTIONS, color="magenta", image_shape=None)

    if left_norm is not None and valid_mask(left_norm).any():
        left_canvas = normalized_to_canvas(left_norm, rgb_can.shape)
        plot_points_lines(ax, left_canvas, HAND_CONNECTIONS, color="cyan", image_shape=None)
    if right_norm is not None and valid_mask(right_norm).any():
        right_canvas = normalized_to_canvas(right_norm, rgb_can.shape)
        plot_points_lines(ax, right_canvas, HAND_CONNECTIONS, color="yellow", image_shape=None)

    ax.set_title(title)
    ax.axis("off")


def make_grid(rgb, data):
    rgb_can, _ = warp_rgb_to_canonical(rgb, data["pose_raw"], out_size=720)

    fig, axes = plt.subplots(3, 4, figsize=(22, 14))
    fig.suptitle(
        "Normalisasi landmark berdasarkan paper\nHijau = sebelum, Ungu = sesudah",
        fontsize=14,
        y=0.98,
    )

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("RGB asli")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(rgb)
    plot_points_lines(axes[0, 1], data["pose_raw"], POSE_CONNECTIONS, color="limegreen", image_shape=rgb.shape)
    plot_points_lines(axes[0, 1], data["left_raw"], HAND_CONNECTIONS, color="deepskyblue", image_shape=rgb.shape)
    plot_points_lines(axes[0, 1], data["right_raw"], HAND_CONNECTIONS, color="orange", image_shape=rgb.shape)
    axes[0, 1].set_title("RGB + skeleton sebelum")
    axes[0, 1].axis("off")

    draw_canonical_rgb_with_skeleton(
        axes[0, 2],
        rgb_can,
        data["pose_full"],
        title="RGB + skeleton sesudah (full-body)",
    )

    overlay_before_after(
        axes[0, 3],
        data["pose_raw"],
        data["pose_full"],
        POSE_CONNECTIONS,
        "Full-body normalization",
        allowed_idx=POSE_IDXS,
    )

    overlay_before_after(
        axes[1, 0],
        data["pose_full"],
        data["pose_face"],
        FACE_CONNECTIONS,
        "Face normalization",
        allowed_idx=FACE_POSE_IDXS,
    )

    overlay_before_after(
        axes[1, 1],
        data["pose_full"],
        data["pose_arm"],
        ARM_A_CONNECTIONS,
        "Arm normalization set A",
        allowed_idx=ARM_SET_A_ANCHOR,
    )

    overlay_before_after(
        axes[1, 2],
        data["pose_full"],
        data["pose_arm"],
        ARM_B_CONNECTIONS,
        "Arm normalization set B",
        allowed_idx=ARM_SET_B_ANCHOR,
    )

    axes[1, 3].axis("off")
    axes[1, 3].text(
        0.02,
        0.98,
        "Catatan:\n"
        "- panel RGB sesudah memakai canonical body view\n"
        "- neck dipusatkan ke tengah kanvas\n"
        "- scale memakai shoulder distance\n"
        "- ini visualisasi ilustratif untuk dosen\n"
        "- normalisasi utama tetap terjadi pada landmark",
        va="top",
        ha="left",
        fontsize=11,
    )

    overlay_before_after(
        axes[2, 0],
        data["left_raw"],
        data["left_norm"],
        HAND_CONNECTIONS,
        "Left hand normalization",
    )

    overlay_before_after(
        axes[2, 1],
        data["right_raw"],
        data["right_norm"],
        HAND_CONNECTIONS,
        "Right hand normalization",
    )

    axes[2, 2].axis("off")
    axes[2, 3].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def main():
    image_path, rgb = load_image_with_os("foto1.png")
    pose, left_hand, right_hand = extract_landmarks(rgb)
    data = preprocess_by_paper(pose, left_hand, right_hand)
    print(f"Loaded: {image_path}")
    make_grid(rgb, data)


if __name__ == "__main__":
    main()
