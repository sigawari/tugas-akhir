import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception as e:  # pragma: no cover
    mp = None


def _clamp_point(x: float, y: float, w: int, h: int) -> Tuple[int, int]:
    """Clamp float pixel coords into image bounds and return int tuple."""
    xi = int(round(x))
    yi = int(round(y))
    xi = 0 if xi < 0 else (w - 1 if xi >= w else xi)
    yi = 0 if yi < 0 else (h - 1 if yi >= h else yi)
    return xi, yi


def draw_face_landmarks(
    img: np.ndarray,
    face_landmarks,
    color: Tuple[int, int, int] = (0, 0, 255),
    radius: int = 1,
    draw_index: bool = True,
) -> None:
    """Face: titik saja (tanpa garis) sesuai style request."""
    if face_landmarks is None:
        return
    h, w = img.shape[:2]
    for idx, lm in enumerate(face_landmarks.landmark):
        x, y = _clamp_point(lm.x * w, lm.y * h, w, h)
        cv2.circle(img, (x, y), radius, color, thickness=-1)
        if draw_index:
            cv2.putText(
                img,
                str(idx),
                (x + 2, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )


def draw_pose_landmarks(
    img: np.ndarray,
    pose_landmarks,
    green: Tuple[int, int, int] = (0, 255, 0),
    white: Tuple[int, int, int] = (255, 255, 255),
    r_green: int = 3,
    r_white: int = 1,
    draw_index: bool = True,
) -> None:
    """Pose: titik hijau + titik putih di tengah (+ index opsional)."""
    if pose_landmarks is None:
        return
    h, w = img.shape[:2]
    for idx, lm in enumerate(pose_landmarks.landmark):
        x, y = _clamp_point(lm.x * w, lm.y * h, w, h)
        cv2.circle(img, (x, y), r_green, green, thickness=-1)
        cv2.circle(img, (x, y), r_white, white, thickness=-1)
        if draw_index:
            cv2.putText(
                img,
                str(idx),
                (x + 2, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                green,
                1,
                cv2.LINE_AA,
            )


def draw_hand_landmarks(
    img: np.ndarray,
    hand_landmarks,
    blue: Tuple[int, int, int] = (255, 0, 0),
    red: Tuple[int, int, int] = (0, 0, 255),
    r_blue: int = 3,
    r_red: int = 1,
    draw_index: bool = True,
) -> None:
    """Hands: titik biru + titik merah di tengah (+ index opsional)."""
    if hand_landmarks is None:
        return
    h, w = img.shape[:2]
    for idx, lm in enumerate(hand_landmarks.landmark):
        x, y = _clamp_point(lm.x * w, lm.y * h, w, h)
        cv2.circle(img, (x, y), r_blue, blue, thickness=-1)
        cv2.circle(img, (x, y), r_red, red, thickness=-1)
        if draw_index:
            cv2.putText(
                img,
                str(idx),
                (x + 2, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                blue,
                1,
                cv2.LINE_AA,
            )


def _draw_connections(img: np.ndarray, landmark_list, connections, color: Tuple[int, int, int], thickness: int = 2) -> None:
    if landmark_list is None:
        return
    h, w = img.shape[:2]
    for a, b in connections:
        la = landmark_list.landmark[a]
        lb = landmark_list.landmark[b]
        x1, y1 = _clamp_point(la.x * w, la.y * h, w, h)
        x2, y2 = _clamp_point(lb.x * w, lb.y * h, w, h)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def draw_all_landmarks(frame_bgr: np.ndarray, results_holistic, draw_index: bool = True) -> np.ndarray:
    """Menggambar face/pose/hands sesuai style + garis pose & tangan."""
    if mp is None:
        raise ImportError("mediapipe is required for drawing connections")

    out = frame_bgr.copy()

    # Pose connections (garis hijau)
    if results_holistic.pose_landmarks is not None:
        _draw_connections(
            out,
            results_holistic.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            color=(0, 255, 0),
            thickness=2,
        )

    # Hands connections (garis biru)
    if results_holistic.left_hand_landmarks is not None:
        _draw_connections(
            out,
            results_holistic.left_hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
            color=(255, 0, 0),
            thickness=2,
        )
    if results_holistic.right_hand_landmarks is not None:
        _draw_connections(
            out,
            results_holistic.right_hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
            color=(255, 0, 0),
            thickness=2,
        )

    # Face (merah) - titik saja (tanpa garis)
    if results_holistic.face_landmarks is not None:
        draw_face_landmarks(out, results_holistic.face_landmarks, color=(0, 0, 255), radius=1, draw_index=draw_index)

    # Pose (hijau + putih)
    if results_holistic.pose_landmarks is not None:
        draw_pose_landmarks(out, results_holistic.pose_landmarks, draw_index=draw_index)

    # Hands (biru + merah)
    if results_holistic.left_hand_landmarks is not None:
        draw_hand_landmarks(out, results_holistic.left_hand_landmarks, draw_index=draw_index)
    if results_holistic.right_hand_landmarks is not None:
        draw_hand_landmarks(out, results_holistic.right_hand_landmarks, draw_index=draw_index)

    return out


def _landmark_list_to_jsonable(landmark_list) -> Optional[Dict[str, Any]]:
    if landmark_list is None:
        return None
    out: Dict[str, Any] = {"landmark": []}
    for lm in landmark_list.landmark:
        item = {
            "x": float(lm.x),
            "y": float(lm.y),
            "z": float(getattr(lm, "z", 0.0)),
        }
        if hasattr(lm, "visibility"):
            item["visibility"] = float(lm.visibility)
        if hasattr(lm, "presence"):
            item["presence"] = float(lm.presence)
        out["landmark"].append(item)
    return out


def _extract_holistic_from_image(image_bgr: np.ndarray):
    if mp is None:
        raise ImportError("mediapipe is required to extract holistic landmarks")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with mp.solutions.holistic.Holistic(
        static_image_mode=True,
        model_complexity=2,
        refine_face_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        return holistic.process(image_rgb)


@dataclass
class ComparisonStats:
    n: int
    mean_l2: float
    max_l2: float
    mean_abs_z: float
    mean_abs_visibility: Optional[float]


def _compare_landmark_lists(a, b) -> Optional[ComparisonStats]:
    """Compare two landmark lists (normalized coords) using per-point L2 on (x,y)."""
    if a is None or b is None:
        return None
    if len(a.landmark) != len(b.landmark):
        return None

    l2s: List[float] = []
    abs_z: List[float] = []
    abs_vis: List[float] = []
    has_vis = hasattr(a.landmark[0], "visibility") and hasattr(b.landmark[0], "visibility")

    for la, lb in zip(a.landmark, b.landmark):
        dx = float(la.x) - float(lb.x)
        dy = float(la.y) - float(lb.y)
        l2s.append(float(np.hypot(dx, dy)))
        abs_z.append(abs(float(getattr(la, "z", 0.0)) - float(getattr(lb, "z", 0.0))))
        if has_vis:
            abs_vis.append(abs(float(la.visibility) - float(lb.visibility)))

    return ComparisonStats(
        n=len(l2s),
        mean_l2=float(np.mean(l2s)) if l2s else 0.0,
        max_l2=float(np.max(l2s)) if l2s else 0.0,
        mean_abs_z=float(np.mean(abs_z)) if abs_z else 0.0,
        mean_abs_visibility=(float(np.mean(abs_vis)) if abs_vis else None),
    )


def extract_and_compare_photos(
    photo_dir: str,
    base_name: str = "foto1",
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Ekstrak landmark raw dari foto (1080/720/480) dan keluarkan perbandingan.

    Ekspektasi file:
      - {base_name}.png (asli, mis. 1080p)
      - {base_name}_720p.png
      - {base_name}_480p.png

    Output:
      - menyimpan JSON raw tiap resolusi
      - return dict ringkas berisi perbandingan 1080 vs 720/480
    """
    if out_dir is None:
        out_dir = os.path.join(photo_dir, "landmark_json")
    os.makedirs(out_dir, exist_ok=True)

    paths = {
        "1080": os.path.join(photo_dir, f"{base_name}.png"),
        "720": os.path.join(photo_dir, f"{base_name}_720p.png"),
        "480": os.path.join(photo_dir, f"{base_name}_480p.png"),
    }

    images: Dict[str, np.ndarray] = {}
    for k, p in paths.items():
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(f"Image not found: {p}")
        images[k] = img

    results = {k: _extract_holistic_from_image(img) for k, img in images.items()}

    # save raw json
    raw_json: Dict[str, Any] = {}
    for k, res in results.items():
        h, w = images[k].shape[:2]
        raw_json[k] = {
            "image": {"path": paths[k], "width": int(w), "height": int(h)},
            "face_landmarks": _landmark_list_to_jsonable(getattr(res, "face_landmarks", None)),
            "pose_landmarks": _landmark_list_to_jsonable(getattr(res, "pose_landmarks", None)),
            "left_hand_landmarks": _landmark_list_to_jsonable(getattr(res, "left_hand_landmarks", None)),
            "right_hand_landmarks": _landmark_list_to_jsonable(getattr(res, "right_hand_landmarks", None)),
        }
        out_path = os.path.join(out_dir, f"{base_name}_{k}_raw.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(raw_json[k], f, ensure_ascii=False, indent=2)

    base = results["1080"]

    comparisons: Dict[str, Any] = {"base": "1080", "comparisons": {}}
    for k in ("720", "480"):
        tgt = results[k]
        comparisons["comparisons"][k] = {
            "face": _compare_landmark_lists(getattr(base, "face_landmarks", None), getattr(tgt, "face_landmarks", None)).__dict__
            if _compare_landmark_lists(getattr(base, "face_landmarks", None), getattr(tgt, "face_landmarks", None))
            else None,
            "pose": _compare_landmark_lists(getattr(base, "pose_landmarks", None), getattr(tgt, "pose_landmarks", None)).__dict__
            if _compare_landmark_lists(getattr(base, "pose_landmarks", None), getattr(tgt, "pose_landmarks", None))
            else None,
            "left_hand": _compare_landmark_lists(getattr(base, "left_hand_landmarks", None), getattr(tgt, "left_hand_landmarks", None)).__dict__
            if _compare_landmark_lists(getattr(base, "left_hand_landmarks", None), getattr(tgt, "left_hand_landmarks", None))
            else None,
            "right_hand": _compare_landmark_lists(getattr(base, "right_hand_landmarks", None), getattr(tgt, "right_hand_landmarks", None)).__dict__
            if _compare_landmark_lists(getattr(base, "right_hand_landmarks", None), getattr(tgt, "right_hand_landmarks", None))
            else None,
        }

    compare_path = os.path.join(out_dir, f"{base_name}_compare_1080_vs_720_480.json")
    with open(compare_path, "w", encoding="utf-8") as f:
        json.dump(comparisons, f, ensure_ascii=False, indent=2)

    return comparisons


def _default_photo_dir() -> str:
    # project_root/src/draw.py -> project_root/photo
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    return os.path.join(project_root, "photo")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Extract MediaPipe Holistic landmarks from photos and compare resolutions")
    parser.add_argument("--photo-dir", default=_default_photo_dir(), help="Folder berisi foto1.png, foto1_720p.png, foto1_480p.png")
    parser.add_argument("--base-name", default="foto1", help="Nama base file (default: foto1)")
    parser.add_argument("--out-dir", default=None, help="Output folder untuk JSON (default: <photo-dir>/landmark_json)")
    args = parser.parse_args()

    comparisons = extract_and_compare_photos(args.photo_dir, base_name=args.base_name, out_dir=args.out_dir)
    print(json.dumps(comparisons, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
