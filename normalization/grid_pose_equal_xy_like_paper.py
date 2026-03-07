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


def raw_pose_to_centered_pose(pose):
    """
    Biar raw pose bisa dibandingkan visualnya dengan normalized pose:
    - translasi ke neck
    - TANPA dibagi shoulder distance
    Jadi proporsi asli tubuh masih kelihatan, tapi pusatnya sama.
    """
    pose = np.asarray(pose, dtype=np.float32)
    out = np.zeros_like(pose)

    if not (point_valid(pose[11]) and point_valid(pose[12])):
        return pose.copy(), {"ok": False, "reason": "invalid_shoulders"}

    neck = (pose[11] + pose[12]) / 2.0
    mask = valid_mask(pose)
    out[mask] = pose[mask] - neck
    return out, {"ok": True, "neck": neck}


def normalized_pose_to_raw_image_overlay(pose_norm, pose_raw):
    """
    Overlay balik ke gambar asli TANPA menggeser ke tengah frame.
    Jadi tidak bikin RGB terasa 'membesar' atau 'berubah'.
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

    neck_target = (pose_raw[11] + pose_raw[12]) / 2.0
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


def plot_pose_on_coord(ax, pose, color, label=None, s=18, lw=2, alpha=1.0):
    pose = np.asarray(pose, dtype=np.float32)
    mask = valid_mask(pose)

    idxs = np.where(mask)[0].tolist()
    if idxs:
        ax.scatter(pose[idxs, 0], pose[idxs, 1], c=color, s=s, zorder=3, label=label, alpha=alpha)

    first = True
    for a, b in POSE_CONNECTIONS:
        if mask[a] and mask[b]:
            ax.plot(
                [pose[a, 0], pose[b, 0]],
                [pose[a, 1], pose[b, 1]],
                color=color,
                linewidth=lw,
                zorder=2,
                alpha=alpha,
                label=label if first and label is not None else None,
            )
            first = False


def equal_limits_from_groups(groups, pad=0.08, default_half=1.0):
    """
    Buat xlim dan ylim dengan skala x/y sama.
    Mirip figure paper: objek tidak gepeng dan proporsi tetap.
    """
    valids = []
    for pts in groups:
        pts = np.asarray(pts, dtype=np.float32)
        mask = valid_mask(pts)
        if mask.any():
            valids.append(pts[mask])

    if not valids:
        return (-default_half, default_half), (default_half, -default_half)

    all_pts = np.concatenate(valids, axis=0)
    x_min, x_max = float(all_pts[:, 0].min()), float(all_pts[:, 0].max())
    y_min, y_max = float(all_pts[:, 1].min()), float(all_pts[:, 1].max())

    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    span_x = max(x_max - x_min, 1e-3)
    span_y = max(y_max - y_min, 1e-3)
    half = 0.5 * max(span_x, span_y)
    half *= (1.0 + pad)

    xlim = (cx - half, cx + half)
    ylim = (cy + half, cy - half)  # y dibalik supaya mirip koordinat gambar
    return xlim, ylim


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
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def summarize_to_terminal(pose_raw, pose_centered, pose_norm, meta_norm):
    print("\n=== RINGKASAN VISUALISASI ===")
    if meta_norm.get("ok", False):
        print(f"neck raw              : ({meta_norm['neck'][0]:.4f}, {meta_norm['neck'][1]:.4f})")
        print(f"shoulder distance raw : {meta_norm['shoulder_dist']:.6f}")

    if point_valid(pose_norm[11]) and point_valid(pose_norm[12]):
        norm_neck = (pose_norm[11] + pose_norm[12]) / 2.0
        norm_shoulder = safe_distance(pose_norm[11], pose_norm[12])
        print(f"neck normalized       : ({norm_neck[0]:.6f}, {norm_neck[1]:.6f})")
        print(f"shoulder dist norm    : {norm_shoulder:.6f}")

    print("\nInterpretasi:")
    print("- Raw-centered = raw pose yang hanya digeser ke neck sebagai pusat.")
    print("- Normalized   = raw-centered lalu dibagi shoulder distance.")
    print("- Jadi perbedaan proporsi paling enak dilihat di panel koordinat, bukan di overlay RGB.")


def make_visual_grid(rgb, pose_raw):
    pose_centered, meta_centered = raw_pose_to_centered_pose(pose_raw)
    pose_norm, meta_norm = full_body_normalize(pose_raw)
    pose_norm_overlay = normalized_pose_to_raw_image_overlay(pose_norm, pose_raw)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Full-body normalization dengan skala x/y seimbang", fontsize=16, y=0.98)

    # kiri atas: raw image + raw pose
    style_image_ax(axes[0, 0], "Raw image + raw pose skeleton", rgb)
    plot_pose_on_image(axes[0, 0], pose_raw, color="limegreen", image_shape=rgb.shape)

    # kanan atas: raw-centered vs normalized dalam satu ruang koordinat
    xlim_cmp, ylim_cmp = equal_limits_from_groups([pose_centered, pose_norm], pad=0.10, default_half=1.0)
    plot_pose_on_coord(axes[0, 1], pose_centered, color="green", label="Before normalization")
    plot_pose_on_coord(axes[0, 1], pose_norm, color="purple", label="After normalization", alpha=0.9)
    style_coord_ax(axes[0, 1], "Before vs after (equal x/y scale)", xlim_cmp, ylim_cmp)
    axes[0, 1].legend(fontsize=8, loc="best")

    # kiri bawah: raw image + normalized overlay (apa adanya)
    style_image_ax(axes[1, 0], "Raw image + normalized pose skeleton", rgb)
    plot_pose_on_image(axes[1, 0], pose_norm_overlay, color="magenta", image_shape=rgb.shape)

    # kanan bawah: normalized only dengan skala x/y seimbang
    xlim_norm, ylim_norm = equal_limits_from_groups([pose_norm], pad=0.10, default_half=1.0)
    plot_pose_on_coord(axes[1, 1], pose_norm, color="magenta")
    style_coord_ax(axes[1, 1], "Normalized pose skeleton only", xlim_norm, ylim_norm)

    note = (
        "Catatan: panel kanan atas memakai skala x/y yang sama agar proporsi tubuh tidak terlihat gepeng. "
        "Raw pose di panel itu dipusatkan ke neck dulu, lalu dibandingkan dengan hasil normalisasi."
    )
    fig.text(0.5, 0.02, note, ha="center", fontsize=9)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

    summarize_to_terminal(pose_raw, pose_centered, pose_norm, meta_norm)


def main():
    image_path, rgb = load_image_with_os("foto1.png")
    pose_raw = extract_pose_only(rgb)
    print(f"Loaded: {image_path}")
    make_visual_grid(rgb, pose_raw)


if __name__ == "__main__":
    main()
