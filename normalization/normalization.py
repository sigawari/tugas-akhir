#!/usr/bin/env python3
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-8

FACE_POSE_IDXS = list(range(11))
POSE_IDXS = list(range(23))
ARM_SET_A = [14, 16, 18, 20, 22]       # sesuai rumus paper: scale pakai dist(12,14)
ARM_SET_B = [13, 15, 17, 19, 21]       # sesuai rumus paper: scale pakai dist(11,13)
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


def set_limits(ax, groups):
    valid_groups = []
    for pts in groups:
        pts = np.asarray(pts, dtype=np.float32)
        mask = valid_mask(pts)
        if mask.any():
            valid_groups.append(pts[mask])

    if not valid_groups:
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(1.5, -1.5)
        return

    all_pts = np.concatenate(valid_groups, axis=0)
    x_min, x_max = float(all_pts[:, 0].min()), float(all_pts[:, 0].max())
    y_min, y_max = float(all_pts[:, 1].min()), float(all_pts[:, 1].max())
    dx = max(x_max - x_min, 1e-3)
    dy = max(y_max - y_min, 1e-3)
    ax.set_xlim(x_min - 0.12 * dx, x_max + 0.12 * dx)
    ax.set_ylim(y_max + 0.12 * dy, y_min - 0.12 * dy)


def style_coord_ax(ax, title):
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def annotate_no_data(ax, points, allowed_idx=None):
    mask = valid_mask(points)
    if allowed_idx is None:
        ok = bool(mask.any())
    else:
        ok = bool(mask[np.array(list(allowed_idx), dtype=int)].any()) if len(allowed_idx) > 0 else False
    if not ok:
        ax.text(0.5, 0.5, "Landmark tidak terdeteksi", ha="center", va="center", transform=ax.transAxes)

def set_fixed_limits(ax):
    """
    Gunakan koordinat tetap supaya pose sebelum dan sesudah
    bisa dibandingkan secara konsisten.
    """
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(1.5, -1.5)

def overlay_before_after(ax, before, after, connections, title, allowed_idx=None):
    plot_points_lines(ax, before, connections, color="green", label="sebelum", allowed_idx=allowed_idx)
    plot_points_lines(ax, after, connections, color="purple", label="sesudah", allowed_idx=allowed_idx)
    set_fixed_limits(ax)
    style_coord_ax(ax, title)
    annotate_no_data(ax, before, allowed_idx=allowed_idx)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8, loc="best")

def normalize_image_full_body(rgb, pose):
    pose = np.asarray(pose, dtype=np.float32)

    if not (point_valid(pose[11]) and point_valid(pose[12])):
        return rgb.copy(), {"ok": False, "reason": "invalid_shoulders"}

    h, w = rgb.shape[:2]

    # ubah landmark normalized -> pixel
    pose_px = pose.copy()
    pose_px[:, 0] *= w
    pose_px[:, 1] *= h

    neck_px = (pose_px[11] + pose_px[12]) / 2.0
    shoulder_dist_px = safe_distance(pose_px[11], pose_px[12])

    if shoulder_dist_px < EPS:
        return rgb.copy(), {"ok": False, "reason": "shoulder_dist_too_small"}

    center_out = np.array([w / 2.0, h / 2.0], dtype=np.float32)

    # target shoulder distance pada output
    target_shoulder = shoulder_dist_px

    # skala isotropik
    scale = target_shoulder / shoulder_dist_px

    # x_out = scale * (x_in - neck_px) + center_out
    # inverse untuk warpAffine:
    # x_in = (1/scale) * x_out + (neck_px - center_out/scale)
    inv_scale = 1.0 / scale

    A = np.array([
        [inv_scale, 0.0, neck_px[0] - center_out[0] * inv_scale],
        [0.0, inv_scale, neck_px[1] - center_out[1] * inv_scale]
    ], dtype=np.float32)

    rgb_norm = cv2.warpAffine(
        rgb,
        A,
        dsize=(w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    return rgb_norm, {
        "ok": True,
        "neck_px": neck_px.tolist(),
        "shoulder_dist_px": float(shoulder_dist_px),
        "scale": float(scale),
    }

def make_grid(rgb, data):
    # ubah dari 3x3 -> 3x4 untuk nambah 1 panel "gambar setelah normalisasi"
    fig, axes = plt.subplots(3, 4, figsize=(21, 14))
    fig.suptitle(
        "Normalisasi landmark berdasarkan paper\nHijau = sebelum, Ungu = sesudah",
        fontsize=14,
        y=0.98,
    )

    # 1. raw image
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("RGB asli")
    axes[0, 0].axis("off")

    # 2. raw image + raw skeleton
    axes[0, 1].imshow(rgb)
    plot_points_lines(axes[0, 1], data["pose_raw"], POSE_CONNECTIONS, color="limegreen", image_shape=rgb.shape)
    plot_points_lines(axes[0, 1], data["left_raw"], HAND_CONNECTIONS, color="deepskyblue", image_shape=rgb.shape)
    plot_points_lines(axes[0, 1], data["right_raw"], HAND_CONNECTIONS, color="orange", image_shape=rgb.shape)
    axes[0, 1].set_title("RGB + skeleton asli")
    axes[0, 1].axis("off")

    # 3. image after full-body normalization (tanpa skeleton)
    rgb_norm, meta = normalize_image_full_body(rgb, data["pose_raw"])
    axes[0, 2].imshow(rgb_norm)
    ttl = "RGB setelah full-body normalization"
    if not meta.get("ok", False):
        ttl += f"\n(fallback: {meta.get('reason', 'unknown')})"
    axes[0, 2].set_title(ttl)
    axes[0, 2].axis("off")

    # 4. full body (overlay before/after)
    overlay_before_after(
        axes[0, 3],
        data["pose_raw"],
        data["pose_full"],
        POSE_CONNECTIONS,
        "Full-body normalization",
        allowed_idx=POSE_IDXS,
    )

    # 5. face
    overlay_before_after(
        axes[1, 0],
        data["pose_full"],
        data["pose_face"],
        FACE_CONNECTIONS,
        "Face normalization",
        allowed_idx=FACE_POSE_IDXS,
    )

    # 6. arm set A
    overlay_before_after(
        axes[1, 1],
        data["pose_full"],
        data["pose_arm"],
        ARM_A_CONNECTIONS,
        "Arm normalization set A",
        allowed_idx=ARM_SET_A_ANCHOR,
    )

    # 7. arm set B
    overlay_before_after(
        axes[1, 2],
        data["pose_full"],
        data["pose_arm"],
        ARM_B_CONNECTIONS,
        "Arm normalization set B",
        allowed_idx=ARM_SET_B_ANCHOR,
    )

    # slot axes[1,3] kosong (opsional) — matikan biar rapi
    axes[1, 3].axis("off")

    # 8. left hand
    overlay_before_after(
        axes[2, 0],
        data["left_raw"],
        data["left_norm"],
        HAND_CONNECTIONS,
        "Left hand normalization",
    )

    # 9. right hand
    overlay_before_after(
        axes[2, 1],
        data["right_raw"],
        data["right_norm"],
        HAND_CONNECTIONS,
        "Right hand normalization",
    )

    # slot axes[2,2] & axes[2,3] kosong — matikan biar rapi
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
