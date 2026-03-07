#!/usr/bin/env python3
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-8

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (0, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
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
            f"File '{filename}' tidak ditemukan.\nTaruh file di folder yang sama dengan script.\n{image_path}"
        )
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Gagal membaca gambar: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return image_path, rgb


def extract_pose_only(rgb):
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise ImportError(
            "mediapipe belum terpasang. Install dulu:\n"
            "pip install mediapipe opencv-python matplotlib numpy"
        ) from exc

    pose = np.zeros((23, 2), dtype=np.float32)

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

    return pose


def full_body_normalize(pose):
    pose = np.asarray(pose, dtype=np.float32)
    out = np.zeros_like(pose)

    if not (point_valid(pose[11]) and point_valid(pose[12])):
        return pose.copy(), {"ok": False, "reason": "invalid_shoulders"}

    neck = (pose[11] + pose[12]) / 2.0
    shoulder_dist = safe_distance(pose[11], pose[12])
    if shoulder_dist < EPS:
        return pose.copy(), {"ok": False, "reason": "small_shoulder_distance"}

    mask = valid_mask(pose)
    out[mask] = (pose[mask] - neck) / shoulder_dist
    return out, {
        "ok": True,
        "neck": neck,
        "shoulder_dist": shoulder_dist,
    }


def normalized_pose_to_raw_image_overlay(pose_norm, pose_raw, mode="centered_hint", anchor_y_ratio=0.42):
    """
    Overlay ilustratif TANPA mentransform RGB.

    mode='identity_back':
        p_overlay = p_norm * shoulder_dist + neck
        -> hasil hampir sama dengan pose raw.

    mode='centered_hint':
        p_overlay = p_norm * shoulder_dist + neck_target
        -> skeleton terlihat 'dipusatkan' di RGB asli agar dosen melihat efek center.
    """
    pose_norm = np.asarray(pose_norm, dtype=np.float32)
    pose_raw = np.asarray(pose_raw, dtype=np.float32)
    out = np.zeros_like(pose_norm)

    if not (point_valid(pose_raw[11]) and point_valid(pose_raw[12])):
        return out

    mask = valid_mask(pose_norm)
    if not mask.any():
        return out

    shoulder_dist = safe_distance(pose_raw[11], pose_raw[12])
    if shoulder_dist < EPS:
        return out

    if mode == "identity_back":
        neck_target = (pose_raw[11] + pose_raw[12]) / 2.0
    else:
        # hanya geser origin ke tengah-ish frame, TANPA ubah RGB
        neck_target = np.array([0.5, anchor_y_ratio], dtype=np.float32)

    out[mask] = pose_norm[mask] * shoulder_dist + neck_target
    return out


def plot_pose_on_image(ax, pose, color, image_shape, s=18, lw=2):
    pose = np.asarray(pose, dtype=np.float32)
    mask = valid_mask(pose)
    h, w = image_shape[:2]
    xs = pose[:, 0] * w
    ys = pose[:, 1] * h

    idxs = np.where(mask)[0].tolist()
    if idxs:
        ax.scatter(xs[idxs], ys[idxs], c=color, s=s, zorder=3)

    for a, b in POSE_CONNECTIONS:
        if mask[a] and mask[b]:
            ax.plot([xs[a], xs[b]], [ys[a], ys[b]], color=color, linewidth=lw, zorder=2)


def plot_pose_on_coord(ax, pose, color, s=18, lw=2):
    pose = np.asarray(pose, dtype=np.float32)
    mask = valid_mask(pose)

    idxs = np.where(mask)[0].tolist()
    if idxs:
        ax.scatter(pose[idxs, 0], pose[idxs, 1], c=color, s=s, zorder=3)

    for a, b in POSE_CONNECTIONS:
        if mask[a] and mask[b]:
            ax.plot([pose[a, 0], pose[b, 0]], [pose[a, 1], pose[b, 1]], color=color, linewidth=lw, zorder=2)


def compute_pose_limits(poses, pad=0.15, default=(-1.5, 1.5, 1.5, -1.5)):
    valids = []
    for pose in poses:
        pose = np.asarray(pose, dtype=np.float32)
        mask = valid_mask(pose)
        if mask.any():
            valids.append(pose[mask])

    if not valids:
        return default[0:2], default[2:4]

    all_pts = np.concatenate(valids, axis=0)
    x_min, x_max = float(all_pts[:, 0].min()), float(all_pts[:, 0].max())
    y_min, y_max = float(all_pts[:, 1].min()), float(all_pts[:, 1].max())
    dx = max(x_max - x_min, 1e-3)
    dy = max(y_max - y_min, 1e-3)
    return (x_min - pad * dx, x_max + pad * dx), (y_max + pad * dy, y_min - pad * dy)


def style_coord_ax(ax, title, xlim, ylim):
    ax.set_title(title, fontsize=11)
    ax.set_facecolor("white")
    ax.grid(alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axhline(0, color="gray", linewidth=0.8, alpha=0.6)
    ax.axvline(0, color="gray", linewidth=0.8, alpha=0.6)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def style_image_ax(ax, title, rgb):
    ax.imshow(rgb)
    ax.set_title(title)
    # Matplotlib imshow uses pixel coordinates; force 1:1 so geometry matches.
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def make_visual_grid(rgb, pose_raw):
    pose_norm, meta = full_body_normalize(pose_raw)

    # Overlay ilustratif di RGB asli, tanpa transform RGB
    pose_norm_overlay = normalized_pose_to_raw_image_overlay(
        pose_norm, pose_raw, mode="identity_back", anchor_y_ratio=0.42
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Full-body normalization tanpa transform RGB", fontsize=16, y=0.98)

    # Panel [0,0] raw image + raw pose (ensure aspect 1:1)
    style_image_ax(axes[0, 0], "Raw image + raw pose skeleton", rgb)
    plot_pose_on_image(axes[0, 0], pose_raw, color="limegreen", image_shape=rgb.shape)

    # Panel [0,1] raw pose only
    xlim_raw, ylim_raw = compute_pose_limits([pose_raw], pad=0.10, default=(-0.1, 1.1, 1.1, -0.1))
    plot_pose_on_coord(axes[0, 1], pose_raw, color="limegreen")
    style_coord_ax(axes[0, 1], "Raw pose skeleton only", xlim_raw, ylim_raw)

    # Panel [1,0] raw image + normalized pose overlay (ensure aspect 1:1)
    style_image_ax(axes[1, 0], "Raw image + normalized pose skeleton", rgb)
    plot_pose_on_image(axes[1, 0], pose_norm_overlay, color="magenta", image_shape=rgb.shape)

    # Panel [1,1] normalized pose only
    xlim_norm, ylim_norm = compute_pose_limits([pose_norm], pad=0.15, default=(-1.5, 1.5, 1.5, -1.5))
    plot_pose_on_coord(axes[1, 1], pose_norm, color="magenta")
    style_coord_ax(axes[1, 1], "Normalized pose skeleton only", xlim_norm, ylim_norm)

    # Also enforce equal aspect for all axes explicitly (defensive)
    for ax in axes.ravel():
        try:
            ax.set_aspect("equal", adjustable="box")
        except Exception:
            pass

    note = (
        "Catatan: RGB tidak ditransform. Panel kiri bawah hanya overlay ilustratif agar terlihat bahwa "
        "normalisasi full-body memusatkan pose ke acuan neck/shoulder. "
        "Panel kanan bawah adalah koordinat normalisasi apa adanya sesuai rumus paper."
    )
    if not meta.get("ok", False):
        note += f" Fallback: {meta.get('reason', 'unknown')}."
    fig.text(0.5, 0.02, note, ha="center", fontsize=9)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()


def main():
    image_path, rgb = load_image_with_os("foto1.png")
    pose_raw = extract_pose_only(rgb)
    print(f"Loaded: {image_path}")
    make_visual_grid(rgb, pose_raw)


if __name__ == "__main__":
    main()
