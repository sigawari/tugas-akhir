# filepath: c:\Users\hp\multimedia\ta-code\normalization\rgb.py

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

# MediaPipe imports
import mediapipe as mp


def _norm_to_pixel(
    x: float, y: float, w: int, h: int
) -> Tuple[Optional[float], Optional[float]]:
    if x is None or y is None:
        return None, None
    return float(x) * w, float(y) * h


def _landmarks_to_rows(
    *,
    kind: str,
    image_name: str,
    w: int,
    h: int,
    person_idx: int,
    hand_idx: Optional[int],
    handedness: Optional[str],
    landmarks: Any,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if landmarks is None:
        return rows

    for i, lm in enumerate(landmarks.landmark):
        px, py = _norm_to_pixel(getattr(lm, "x", None), getattr(lm, "y", None), w, h)
        row: Dict[str, Any] = {
            "image": image_name,
            "kind": kind,
            "person_index": person_idx,
            "hand_index": hand_idx,
            "handedness": handedness,
            "landmark_index": i,
            "x_norm": float(getattr(lm, "x", np.nan)),
            "y_norm": float(getattr(lm, "y", np.nan)),
            "z_norm": float(getattr(lm, "z", np.nan)),
            "x_px": px,
            "y_px": py,
            "visibility": float(getattr(lm, "visibility", np.nan))
            if hasattr(lm, "visibility")
            else np.nan,
            "presence": float(getattr(lm, "presence", np.nan))
            if hasattr(lm, "presence")
            else np.nan,
        }
        rows.append(row)

    return rows


def extract_landmarks_to_excel(
    image_path: str,
    output_excel_path: str,
    static_image_mode: bool = True,
    max_num_faces: int = 1,
    max_num_hands: int = 2,
    model_complexity: int = 2,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> None:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    image_name = os.path.basename(image_path)

    face_rows: List[Dict[str, Any]] = []
    hand_rows: List[Dict[str, Any]] = []
    pose_rows: List[Dict[str, Any]] = []

    with mp_face_mesh.FaceMesh(
        static_image_mode=static_image_mode,
        max_num_faces=max_num_faces,
        refine_landmarks=True,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as face_mesh, mp_hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=max_num_hands,
        model_complexity=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as hands, mp_pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        enable_segmentation=False,
        smooth_landmarks=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as pose:
        face_res = face_mesh.process(rgb)
        hands_res = hands.process(rgb)
        pose_res = pose.process(rgb)

        if face_res.multi_face_landmarks:
            for face_idx, face_lms in enumerate(face_res.multi_face_landmarks):
                face_rows.extend(
                    _landmarks_to_rows(
                        kind="face",
                        image_name=image_name,
                        w=w,
                        h=h,
                        person_idx=face_idx,
                        hand_idx=None,
                        handedness=None,
                        landmarks=face_lms,
                    )
                )

        if hands_res.multi_hand_landmarks:
            handedness_list = hands_res.multi_handedness or []
            for i, hand_lms in enumerate(hands_res.multi_hand_landmarks):
                label: Optional[str] = None
                if i < len(handedness_list) and handedness_list[i].classification:
                    label = handedness_list[i].classification[0].label
                hand_rows.extend(
                    _landmarks_to_rows(
                        kind="hand",
                        image_name=image_name,
                        w=w,
                        h=h,
                        person_idx=0,
                        hand_idx=i,
                        handedness=label,
                        landmarks=hand_lms,
                    )
                )

        if pose_res.pose_landmarks:
            pose_rows.extend(
                _landmarks_to_rows(
                    kind="pose",
                    image_name=image_name,
                    w=w,
                    h=h,
                    person_idx=0,
                    hand_idx=None,
                    handedness=None,
                    landmarks=pose_res.pose_landmarks,
                )
            )

    df_face = pd.DataFrame(face_rows)
    df_hand = pd.DataFrame(hand_rows)
    df_pose = pd.DataFrame(pose_rows)

    os.makedirs(os.path.dirname(output_excel_path) or ".", exist_ok=True)
    with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
        df_face.to_excel(writer, index=False, sheet_name="face")
        df_hand.to_excel(writer, index=False, sheet_name="hands")
        df_pose.to_excel(writer, index=False, sheet_name="pose")


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))

    image_path = os.path.join(script_dir, "foto1.png")
    output_excel_path = os.path.join(script_dir, "landmarks_foto1.xlsx")

    extract_landmarks_to_excel(
        image_path=image_path,
        output_excel_path=output_excel_path,
    )

    print(f"Saved Excel: {output_excel_path}")


if __name__ == "__main__":
    main()
