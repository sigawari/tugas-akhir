import os
import time
from pathlib import Path

import cv2
import numpy as np

import mediapipe as mp

# ---------- Output folders (di luar src/) ----------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # .../ta-code
PHOTO_ROOT = PROJECT_ROOT / "photo"
RAW_DIR = PHOTO_ROOT / "raw"
RAW_WITH_LM_DIR = PHOTO_ROOT / "raw_with_landmark"
LM_ONLY_DIR = PHOTO_ROOT / "landmark_only"


def ensure_dirs():
    for d in (RAW_DIR, RAW_WITH_LM_DIR, LM_ONLY_DIR):
        d.mkdir(parents=True, exist_ok=True)


def ts_name(prefix: str = "img", ext: str = ".png") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time()*1000)%1000:03d}{ext}"


# ---------- Drawing helpers ----------
def _clamp_point(x, y, w, h):
    return max(0, min(int(x), w - 1)), max(0, min(int(y), h - 1))


def draw_face_landmarks(img, face_landmarks, color=(0, 0, 255), radius=1, draw_index: bool = True):
    """Face: titik merah (+ index opsional)."""
    h, w = img.shape[:2]
    for idx, lm in enumerate(face_landmarks.landmark):
        x, y = _clamp_point(lm.x * w, lm.y * h, w, h)
        cv2.circle(img, (x, y), radius, color, thickness=-1)
        if draw_index:
            cv2.putText(img, str(idx), (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


def draw_pose_landmarks(
    img,
    pose_landmarks,
    green=(0, 255, 0),
    white=(255, 255, 255),
    r_green=3,
    r_white=1,
    draw_index: bool = True,
):
    """Pose: titik hijau + titik putih di tengah (+ index opsional)."""
    h, w = img.shape[:2]
    for idx, lm in enumerate(pose_landmarks.landmark):
        x, y = _clamp_point(lm.x * w, lm.y * h, w, h)
        cv2.circle(img, (x, y), r_green, green, thickness=-1)
        cv2.circle(img, (x, y), r_white, white, thickness=-1)
        if draw_index:
            cv2.putText(img, str(idx), (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, green, 1, cv2.LINE_AA)


def draw_hand_landmarks(
    img,
    hand_landmarks,
    blue=(255, 0, 0),
    red=(0, 0, 255),
    r_blue=3,
    r_red=1,
    draw_index: bool = True,
):
    """Hands: titik biru + titik merah di tengah (+ index opsional)."""
    h, w = img.shape[:2]
    for idx, lm in enumerate(hand_landmarks.landmark):
        x, y = _clamp_point(lm.x * w, lm.y * h, w, h)
        cv2.circle(img, (x, y), r_blue, blue, thickness=-1)
        cv2.circle(img, (x, y), r_red, red, thickness=-1)
        if draw_index:
            cv2.putText(img, str(idx), (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue, 1, cv2.LINE_AA)


def _draw_connections(img, landmark_list, connections, color, thickness=2):
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
    """Menggambar face/pose/hands sesuai style yang diminta + garis pose & tangan."""
    out = frame_bgr.copy()

    # Pose connections (garis hijau)
    if results_holistic.pose_landmarks is not None:
        _draw_connections(out, results_holistic.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS, color=(0, 255, 0), thickness=2)

    # Hands connections (garis biru)
    if results_holistic.left_hand_landmarks is not None:
        _draw_connections(out, results_holistic.left_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS, color=(255, 0, 0), thickness=2)
    if results_holistic.right_hand_landmarks is not None:
        _draw_connections(out, results_holistic.right_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS, color=(255, 0, 0), thickness=2)

    # Face (merah) - titik saja sesuai request (tanpa garis)
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


def make_landmark_only_canvas(shape_hw, results_holistic) -> np.ndarray:
    """Canvas hitam dengan landmark saja."""
    h, w = shape_hw
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    return draw_all_landmarks(canvas, results_holistic)


def put_hud(img, mode_name: str, show_index: bool):
    lines = [
        f"Mode: {mode_name}   (1=RAW, 2=RAW+LM)",
        f"Index: {'ON' if show_index else 'OFF'} (i=toggle)",
        "c=save RAW | l=save RAW+LM | k=save LM only (black bg)",
        "q/Esc=quit",
    ]
    y = 22
    for t in lines:
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        y += 22


def _ensure_mediapipe_solutions_available():
    """Return a holistic module-like object.

    Some broken installs / shadowing can remove mp.solutions; we try a fallback import.
    """
    # Quick shadowing detection in current working directory / script directory
    cwd = Path.cwd()
    script_dir = THIS_FILE.parent if "THIS_FILE" in globals() else Path(__file__).resolve().parent
    for base in (cwd, script_dir):
        if (base / "mediapipe.py").exists() or (base / "mediapipe").is_dir():
            # Don't hard-fail; just warn via exception message if fallback also fails.
            shadow_hint = (
                f"Kemungkinan ada file/folder 'mediapipe' yang menimpa package di: {base}"
            )
            break
    else:
        shadow_hint = None

    if hasattr(mp, "solutions") and hasattr(mp.solutions, "holistic"):
        return mp.solutions.holistic

    # Fallback: direct import path used by mediapipe
    try:
        from mediapipe.python.solutions import holistic as holistic_mod  # type: ignore
        return holistic_mod
    except Exception as e:  # pragma: no cover
        mp_file = getattr(mp, "__file__", "<unknown>")
        mp_ver = getattr(mp, "__version__", "<unknown>")
        extra = f"\n{shadow_hint}\n" if shadow_hint else "\n"
        raise RuntimeError(
            "Tidak menemukan 'mp.solutions.holistic'. Ini biasanya karena mediapipe tertimpa modul lokal atau install mediapipe rusak.\n"
            f"Imported mediapipe from: {mp_file}\n"
            f"mediapipe version: {mp_ver}"
            f"{extra}"
            "Perbaikan cepat:\n"
            "- Pastikan tidak ada mediapipe.py / folder mediapipe di project.\n"
            "- Reinstall: pip uninstall mediapipe -y && pip install mediapipe"
        ) from e


def main():
    ensure_dirs()

    mp_holistic = _ensure_mediapipe_solutions_available()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows-friendly; kalau bermasalah, hapus argumen kedua
    if not cap.isOpened():
        raise RuntimeError("Tidak bisa membuka kamera. Pastikan webcam tidak dipakai app lain.")

    # Set resolusi 1920x1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    mode = 2  # 1=raw, 2=raw+landmark
    show_index = True

    # Tuning confidences sesuai kebutuhan
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # mirror biar lebih natural
            h, w = frame.shape[:2]

            # MediaPipe proses RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = holistic.process(frame_rgb)
            frame_rgb.flags.writeable = True

            # Render sesuai mode
            if mode == 1:
                view = frame.copy()
                mode_name = "RAW"
            else:
                view = draw_all_landmarks(frame, results, draw_index=show_index)
                mode_name = "RAW + LANDMARK"

            put_hud(view, mode_name, show_index)
            cv2.imshow("Camera (MediaPipe)", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):  # Esc / q
                break

            if key == ord("1"):
                mode = 1
            elif key == ord("2"):
                mode = 2
            elif key == ord("i"):
                show_index = not show_index

            # Save actions
            elif key == ord("c"):
                # raw
                fname = ts_name("raw")
                out_path = RAW_DIR / fname
                cv2.imwrite(str(out_path), frame)
                print(f"[SAVE] RAW -> {out_path}")

            elif key == ord("l"):
                # raw + landmark
                fname = ts_name("raw_with_lm")
                out_path = RAW_WITH_LM_DIR / fname
                img = draw_all_landmarks(frame, results, draw_index=show_index)
                cv2.imwrite(str(out_path), img)
                print(f"[SAVE] RAW+LM -> {out_path}")

            elif key == ord("k"):
                # landmark only
                fname = ts_name("lm_only")
                out_path = LM_ONLY_DIR / fname
                canvas = make_landmark_only_canvas((h, w), results)
                # apply index on landmark-only too
                if show_index:
                    canvas = draw_all_landmarks(canvas, results, draw_index=True)
                cv2.imwrite(str(out_path), canvas)
                print(f"[SAVE] LM ONLY -> {out_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()