# Try extracting by mp4 video
import os
import json
import math
from datetime import datetime

import cv2
import mediapipe as mp
from tqdm import tqdm


# --- Landmark name maps (MediaPipe canonical ordering) ---
POSE_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

HAND_NAMES = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_finger_mcp",
    "index_finger_pip",
    "index_finger_dip",
    "index_finger_tip",
    "middle_finger_mcp",
    "middle_finger_pip",
    "middle_finger_dip",
    "middle_finger_tip",
    "ring_finger_mcp",
    "ring_finger_pip",
    "ring_finger_dip",
    "ring_finger_tip",
    "pinky_finger_mcp",
    "pinky_finger_pip",
    "pinky_finger_dip",
    "pinky_finger_tip",
]

# FaceMesh has 468 landmarks; name them as face_0..face_467 for a stable schema
FACE_NAMES = [f"face_{i}" for i in range(468)]


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0


def lms_to_named_dict(landmarks, names, with_visibility=False):
    """Convert MediaPipe landmarks into a fixed-schema dict keyed by landmark names.

    If landmarks is None -> fill zeros.
    Values are dicts:
      - without visibility: {x,y,z}
      - with visibility: {x,y,z,visibility}
    """
    out = {}
    if landmarks is None:
        for name in names:
            if with_visibility:
                out[name] = {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.0}
            else:
                out[name] = {"x": 0.0, "y": 0.0, "z": 0.0}
        return out

    for i, name in enumerate(names):
        if i < len(landmarks):
            lm = landmarks[i]
            x = safe_float(getattr(lm, "x", 0.0))
            y = safe_float(getattr(lm, "y", 0.0))
            z = safe_float(getattr(lm, "z", 0.0))
            if with_visibility:
                v = safe_float(getattr(lm, "visibility", 0.0))
                out[name] = {"x": x, "y": y, "z": z, "visibility": v}
            else:
                out[name] = {"x": x, "y": y, "z": z}
        else:
            if with_visibility:
                out[name] = {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.0}
            else:
                out[name] = {"x": 0.0, "y": 0.0, "z": 0.0}

    return out


def get_video_info(video_path: str):
    st = os.stat(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Gagal membuka video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

    duration_sec = (frame_count / fps) if fps and fps > 0 else None

    info = {
        "path": os.path.abspath(video_path),
        "file_name": os.path.basename(video_path),
        "file_size_bytes": st.st_size,
        "modified_time": datetime.fromtimestamp(st.st_mtime).isoformat(),
        "fps": float(fps) if fps else 0.0,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "fourcc": fourcc,
        "duration_sec": float(duration_sec) if duration_sec is not None else None,
    }

    cap.release()
    return info


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(base_dir, "try.MP4")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"File video tidak ditemukan: {video_path}")

    out_dir = os.path.join(base_dir, "try_result")
    os.makedirs(out_dir, exist_ok=True)

    video_info = get_video_info(video_path)

    # MediaPipe configs
    mp_holistic = mp.solutions.holistic

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Gagal membuka video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # Build requested JSON schema
    fps = video_info.get("fps", 0.0) or 0.0
    duration_sec = video_info.get("duration_sec", None)

    # Derive video_id (filename without extension)
    video_id = os.path.splitext(os.path.basename(video_path))[0]

    results_json = {
        "metadata": {
            "video_id": video_id,
            "fps": float(fps),
            "duration_sec": float(duration_sec) if duration_sec is not None else None,
            "total_frames": int(video_info.get("frame_count", total_frames) or total_frames),
            "model": "MediaPipe Holistic",
            "action": "unknown",
        },
        "frames": [],
    }

    # Base timestamp in ms (best-effort). If the file has no timing metadata accessible,
    # we use current time at extraction start as t0.
    t0_ms = int(datetime.now().timestamp() * 1000)

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        pbar = tqdm(total=total_frames if total_frames > 0 else None, desc="Extracting", unit="frame")
        frame_idx = 0

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = holistic.process(frame_rgb)

            pose_landmarks = res.pose_landmarks.landmark if res.pose_landmarks else None
            face_landmarks = res.face_landmarks.landmark if res.face_landmarks else None
            left_landmarks = res.left_hand_landmarks.landmark if res.left_hand_landmarks else None
            right_landmarks = res.right_hand_landmarks.landmark if res.right_hand_landmarks else None

            # Timestamp per frame in ms (best-effort)
            if fps and fps > 0:
                timestamp_ms = t0_ms + int((frame_idx / fps) * 1000)
            else:
                timestamp_ms = t0_ms

            results_json["frames"].append(
                {
                    "frame_index": frame_idx,
                    "timestamp_ms": int(timestamp_ms),
                    "landmarks": {
                        "pose": lms_to_named_dict(pose_landmarks, POSE_NAMES, with_visibility=False),
                        "face": lms_to_named_dict(face_landmarks, FACE_NAMES, with_visibility=False),
                        "left_hand": lms_to_named_dict(left_landmarks, HAND_NAMES, with_visibility=False),
                        "right_hand": lms_to_named_dict(right_landmarks, HAND_NAMES, with_visibility=False),
                    },
                }
            )

            frame_idx += 1
            pbar.update(1)

        pbar.close()

    cap.release()

    out_path = os.path.join(out_dir, "try_landmarks.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False)

    print(f"Done. JSON tersimpan di: {out_path}")


if __name__ == "__main__":
    main()