import os
import cv2
import json
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from statistics import mean


# =========================
# CONFIG DEFAULT
# =========================

DEFAULT_CONFIG = {
    "paths": {
        # Paths are relative to the PROJECT ROOT (one level above src/)
        # Project layout:
        #   project_root/
        #   ├── data/
        #   │   ├── raw/videos/<label>/*.mp4
        #   │   ├── interim/
        #   │   │   ├── landmarks_extracted/
        #   │   │   ├── landmarks_selected/
        #   │   │   ├── landmarks_interpolated/
        #   │   │   └── landmarks_normalization/
        #   │   └── processed/npy/
        #   ├── src/
        #   │   └── data_handler.py   ← this file
        #   ├── configs/
        #   └── scripts/
        "raw_video_dir":      "data/raw/raw_test",
        "extracted_dir":      "data/interim/landmarks_test_extracted",
        "selected_json_dir":  "data/interim/landmarks_test_selected",
        "cleaned_json_dir":   "data/interim/landmarks_test_interpolated",
        "normalized_json_dir":"data/interim/landmarks_test_normalization",
        "npy_dir":            "data/processed/testnpy",
    },
    "preprocessing": {
        "target_frames": 90,
        "model_complexity": 1,
    }
}


# =========================
# LANDMARK CONSTANTS
# =========================

mp_holistic = mp.solutions.holistic

POSE_NAMES = [lm.name.lower() for lm in mp_holistic.PoseLandmark]
HAND_NAMES = [str(i) for i in range(21)]
FACE_NAMES = [str(i) for i in range(468)]

POSE_KEYS = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
]
LEFT_ARM_KEYS  = ["left_elbow",  "left_wrist",  "left_pinky",  "left_index",  "left_thumb"]
RIGHT_ARM_KEYS = ["right_elbow", "right_wrist", "right_pinky", "right_index", "right_thumb"]
HAND_KEYS = [str(i) for i in range(21)]
FACE_KEYS = [str(i) for i in range(68)]

POSE_SELECTED = list(range(11, 23))

# MediaPipe-468 → dlib-68 face mapping
MP2DLIB = [
    [127], [234], [93], [132, 58], [58, 172], [136], [150], [176], [152],
    [400], [379], [365], [397, 288], [361], [323], [454], [356],
    [70], [63], [105], [66], [107],
    [336], [296], [334], [293], [300],
    [168, 6], [197, 195], [5], [4], [75], [97], [2], [326], [305],
    [33], [160], [158], [133], [153], [144],
    [362], [385], [387], [263], [373], [380],
    [61], [39], [37], [0], [267], [269], [291],
    [321], [314], [17], [84], [91],
    [78], [82], [13], [312], [308],
    [317], [14], [87],
]
# Pad single-element entries so every row has 2 indices
for i, row in enumerate(MP2DLIB):
    if len(row) == 1:
        MP2DLIB[i] = [row[0], row[0]]

FACE_NOSE_INDEX = 29  # dlib point 30 (0-based)
EPS = 1e-6


# =========================
# PREPROCESSOR CLASS
# =========================

class Preprocessor:
    """
    End-to-end sign-language landmark preprocessing pipeline.

    Steps (can be called individually or chained via run_all):
        1. extract_landmarks()   – video → raw JSON (MediaPipe Holistic)
        2. select_landmarks()    – raw JSON → selected/reduced JSON
        3. clean_data()          – selected JSON → cleaned JSON (crop + interpolate hands)
        4. normalize_data()      – cleaned JSON → normalized JSON
    """

    def __init__(self, config: dict = None):
        cfg = config or DEFAULT_CONFIG
        paths = cfg.get("paths", {})
        prep  = cfg.get("preprocessing", {})

        # ── base = project root (parent of the src/ directory this file lives in) ──
        # __file__ → .../project_root/src/data_handler.py
        # parent   → .../project_root/src/
        # parent   → .../project_root/          ← base
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # ── resolve paths ─────────────────────────────────────────────
        # Supports two config styles:
        #
        # Style A – preprocess.yaml (flat, uses interim_dir as base):
        #   paths:
        #     raw_data_dir: "data/raw/videos"
        #     interim_dir:  "data/interim"
        #     processed_dir: "data/processed"
        #
        # Style B – DEFAULT_CONFIG (explicit per-subfolder keys):
        #   paths:
        #     raw_video_dir:      "data/raw/videos"
        #     extracted_dir:      "data/interim/landmarks_extracted"
        #     ...
        #
        # Style A takes priority when interim_dir is present.

        interim   = paths.get("interim_dir",   "data/interim")
        processed = paths.get("processed_dir", "data/processed")

        self.raw_video_dir       = self._abs(base, paths.get("raw_data_dir",
                                             paths.get("raw_video_dir",       "data/raw/videos")))
        self.extracted_dir       = self._abs(base, paths.get("extracted_dir",
                                             f"{interim}/landmarks_extracted"))
        self.selected_json_dir   = self._abs(base, paths.get("selected_json_dir",
                                             f"{interim}/landmarks_selected"))
        self.cleaned_json_dir    = self._abs(base, paths.get("cleaned_json_dir",
                                             f"{interim}/landmarks_interpolated"))
        self.normalized_json_dir = self._abs(base, paths.get("normalized_json_dir",
                                             f"{interim}/landmarks_normalization"))
        self.npy_dir             = self._abs(base, paths.get("npy_dir",
                                             f"{processed}/npy"))

        # ── hyper-params ───────────────────────────────────────
        self.target_frames    = prep.get("frame_count",
                                prep.get("target_frames",    90))
        self.model_complexity = prep.get("model_complexity", 1)

        # create all output directories up front
        for d in [self.extracted_dir, self.selected_json_dir,
                  self.cleaned_json_dir, self.normalized_json_dir,
                  self.npy_dir]:
            os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------
    # PUBLIC: pipeline entry points
    # ------------------------------------------------------------------

    def run_all(self):
        """Run the full pipeline end-to-end."""
        self.extract_landmarks()
        self.select_landmarks()
        self.clean_data()
        self.normalize_data()
        self.convert_to_npy()

    def extract_landmarks(self):
        """Step 1 – extract MediaPipe Holistic landmarks from raw videos."""
        self._process_label_dirs(
            src_root=self.raw_video_dir,
            dst_root=self.extracted_dir,
            file_ext=".mp4",
            process_fn=self._extract_single_video,
            step_name="Extracting landmarks",
        )

    def select_landmarks(self):
        """Step 2 – reduce 468 face points → 68, keep upper-body pose only."""
        self._process_label_dirs(
            src_root=self.extracted_dir,
            dst_root=self.selected_json_dir,
            file_ext=".json",
            process_fn=self._select_single_file,
            step_name="Selecting landmarks",
        )

    def clean_data(self):
        """Step 3 – crop to target_frames and interpolate missing hand landmarks."""
        self._process_label_dirs(
            src_root=self.selected_json_dir,
            dst_root=self.cleaned_json_dir,
            file_ext=".json",
            process_fn=self._clean_single_file,
            step_name="Cleaning data",
        )

    def normalize_data(self):
        """Step 4 – full-body, face, arm, and hand-bbox normalization."""
        self._process_label_dirs(
            src_root=self.cleaned_json_dir,
            dst_root=self.normalized_json_dir,
            file_ext=".json",
            process_fn=self._normalize_single_file,
            step_name="Normalizing data",
        )

    def convert_to_npy(self):
        """Step 5 – Convert normalized JSON → .npy arrays ready for model training.

        Output layout
        -------------
        npy_dir/
            X.npy   – float32  [N, T, L, 2]
                        N = total samples across all labels
                        T = target_frames (e.g. 90)
                        L = total landmarks (12 pose + 21 lh + 21 rh + 68 face = 122)
                        2 = (x, y)
            y.npy   – int64    [N]  class indices
            labels.npy          sorted label strings (decode y with labels[y[i]])

        Landmark order inside the L axis (fixed, documented here):
            [0:12]   pose      (POSE_KEYS order)
            [12:33]  left_hand (0-20)
            [33:54]  right_hand(0-20)
            [54:122] face      (0-67)
        """
        print("\nConverting normalized JSON to .npy files...")

        # ── 1. collect all label names (sorted for reproducibility) ──
        label_names = sorted([
            d for d in os.listdir(self.normalized_json_dir)
            if os.path.isdir(os.path.join(self.normalized_json_dir, d))
        ])

        if not label_names:
            raise FileNotFoundError(
                f"No label subdirectories found in: {self.normalized_json_dir}"
            )

        label_to_idx = {name: idx for idx, name in enumerate(label_names)}

        # ── 2. stack all samples ──────────────────────────────────────
        X_list: list[np.ndarray] = []
        y_list: list[int]        = []

        for label in label_names:
            label_dir  = os.path.join(self.normalized_json_dir, label)
            json_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".json")])

            print(f"  [{label}]  {len(json_files)} samples")

            for fname in tqdm(json_files, desc=f"  stacking {label}"):
                fpath = os.path.join(label_dir, fname)
                arr   = self._json_to_array(fpath)   # [T, L, 2]
                X_list.append(arr)
                y_list.append(label_to_idx[label])

        X = np.stack(X_list, axis=0).astype(np.float32)   # [N, T, L, 2]
        y = np.array(y_list, dtype=np.int64)               # [N]

        # ── 3. save ───────────────────────────────────────────────────
        x_path      = os.path.join(self.npy_dir, "X.npy")
        y_path      = os.path.join(self.npy_dir, "y.npy")
        labels_path = os.path.join(self.npy_dir, "labels.npy")

        np.save(x_path,      X)
        np.save(y_path,      y)
        np.save(labels_path, np.array(label_names))

        print(f"\n  X shape : {X.shape}  (N, T, L, 2)")
        print(f"  y shape : {y.shape}  (N,)")
        print(f"  Classes : {label_names}")
        print(f"\n  Saved → {x_path}")
        print(f"  Saved → {y_path}")
        print(f"  Saved → {labels_path}")

    # ------------------------------------------------------------------
    # PRIVATE: generic label-directory walker
    # ------------------------------------------------------------------

    def _process_label_dirs(self, src_root, dst_root, file_ext, process_fn, step_name):
        for label in os.listdir(src_root):
            src_label_dir = os.path.join(src_root, label)
            if not os.path.isdir(src_label_dir):
                continue

            dst_label_dir = os.path.join(dst_root, label)
            os.makedirs(dst_label_dir, exist_ok=True)

            files = [f for f in os.listdir(src_label_dir) if f.endswith(file_ext)]
            print(f"\n{step_name} – label: {label} ({len(files)} files)")

            for fname in tqdm(files):
                src_path = os.path.join(src_label_dir, fname)
                out_stem = fname.replace(file_ext, ".json")
                dst_path = os.path.join(dst_label_dir, out_stem)
                result = process_fn(src_path, label)
                self._save_json(result, dst_path)

    # ------------------------------------------------------------------
    # PRIVATE: per-file processors
    # ------------------------------------------------------------------

    # ── Step 1 ──────────────────────────────────────────────────────────

    def _extract_single_video(self, video_path: str, label: str) -> dict:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        output = {
            "metadata": {
                "video_id": video_name,
                "fps": fps,
                "duration_sec": total_frames / fps if fps > 0 else 0,
                "total_frames": total_frames,
                "model": "MediaPipe Holistic",
                "action": label,
            },
            "frames": [],
        }

        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            enable_segmentation=False,
            refine_face_landmarks=False,
        ) as holistic:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)
                output["frames"].append({
                    "frame_index": frame_idx,
                    "timestamp_ms": int(frame_idx * (1000 / fps)) if fps > 0 else 0,
                    "landmarks": {
                        "pose":       self._lm_to_dict(results.pose_landmarks,       POSE_NAMES),
                        "left_hand":  self._lm_to_dict(results.left_hand_landmarks,  HAND_NAMES),
                        "right_hand": self._lm_to_dict(results.right_hand_landmarks, HAND_NAMES),
                        "face":       self._lm_to_dict(results.face_landmarks,       FACE_NAMES),
                    },
                })
                frame_idx += 1

        cap.release()
        return output

    # ── Step 2 ──────────────────────────────────────────────────────────

    def _select_single_file(self, input_file: str, label: str) -> dict:
        data = self._load_json(input_file)
        selected_frames = [self._select_frame_landmarks(f) for f in data["frames"]]
        return {
            "metadata": {
                **data["metadata"],
                "selection_info": {
                    "pose_indices": POSE_SELECTED,
                    "left_hand_indices": list(range(21)),
                    "right_hand_indices": list(range(21)),
                    "face_total_selected": 68,
                    "coordinate_dim": ["x", "y"],
                    "face_mapping": "mediapipe_468_to_dlib_inspired_68",
                },
            },
            "frames": selected_frames,
        }

    def _select_frame_landmarks(self, frame: dict) -> dict:
        lm = frame["landmarks"]

        pose_selected = {
            POSE_NAMES[i]: self._get_xy(lm["pose"][POSE_NAMES[i]])
            for i in POSE_SELECTED
        }
        lh_selected = {str(i): self._get_xy(lm["left_hand"][str(i)])  for i in range(21)}
        rh_selected = {str(i): self._get_xy(lm["right_hand"][str(i)]) for i in range(21)}
        face_selected = {
            str(dlib_idx): self._mean_face_points(lm["face"], src_indices)
            for dlib_idx, src_indices in enumerate(MP2DLIB)
        }

        return {
            "frame_index": frame["frame_index"],
            "timestamp_ms": frame["timestamp_ms"],
            "landmarks": {
                "pose": pose_selected,
                "left_hand": lh_selected,
                "right_hand": rh_selected,
                "face": face_selected,
            },
        }

    # ── Step 3 ──────────────────────────────────────────────────────────

    def _clean_single_file(self, input_file: str, label: str) -> dict:
        data = self._load_json(input_file)
        frames = data["frames"]
        original_count = len(frames)

        frames = self._crop_frames(frames)
        frames = self._interpolate_hand_component(frames, "left_hand")
        frames = self._interpolate_hand_component(frames, "right_hand")

        fps = data["metadata"].get("fps", 0)
        for i, frame in enumerate(frames):
            frame["frame_index"] = i
            frame["timestamp_ms"] = int(i * (1000 / fps)) if fps > 0 else 0

        metadata = dict(data["metadata"])
        metadata["original_total_frames_after_selection"] = original_count
        metadata["total_frames"] = len(frames)
        metadata["duration_sec"] = len(frames) / fps if fps > 0 else 0
        metadata["preprocessing"] = {
            "target_frames": self.target_frames,
            "cropping_applied": original_count > self.target_frames,
            "missing_handling": {
                "pose": "none",
                "face": "none",
                "left_hand":  "linear_interpolation_if_partial_missing_keep_zero_if_all_missing",
                "right_hand": "linear_interpolation_if_partial_missing_keep_zero_if_all_missing",
            },
        }
        return {"metadata": metadata, "frames": frames}

    def _interpolate_hand_component(self, frames: list, component: str) -> list:
        arr = self._component_to_array(frames, component, HAND_KEYS)   # [T, 21, 2]
        for k in range(arr.shape[1]):
            arr[:, k, :] = self._interpolate_xy_track(arr[:, k, :])
        self._write_array_back(frames, component, HAND_KEYS, arr)
        return frames

    # ── Step 4 ──────────────────────────────────────────────────────────

    def _normalize_single_file(self, input_file: str, label: str) -> dict:
        data = self._load_json(input_file)
        frames = self._crop_frames(data["frames"])

        for t, frame in enumerate(frames):
            lm = frame["landmarks"]

            pose_arr = self._to_arr(lm["pose"],       POSE_KEYS)
            face_arr = self._to_arr(lm["face"],       FACE_KEYS)
            lh_arr   = self._to_arr(lm["left_hand"],  HAND_KEYS)
            rh_arr   = self._to_arr(lm["right_hand"], HAND_KEYS)

            lh_absent = self._absent_mask(lh_arr)
            rh_absent = self._absent_mask(rh_arr)

            pose_arr, face_arr, lh_arr, rh_arr = self._full_body_normalize(
                pose_arr, face_arr, lh_arr, rh_arr, lh_absent, rh_absent
            )
            face_arr = self._face_normalize(face_arr)
            pose_arr = self._arm_normalize(pose_arr)
            lh_arr   = self._hand_bbox_normalize(lh_arr, lh_absent)
            rh_arr   = self._hand_bbox_normalize(rh_arr, rh_absent)

            self._write_arr(lm["pose"],       POSE_KEYS, pose_arr)
            self._write_arr(lm["face"],       FACE_KEYS, face_arr)
            self._write_arr(lm["left_hand"],  HAND_KEYS, lh_arr)
            self._write_arr(lm["right_hand"], HAND_KEYS, rh_arr)

            fps = data["metadata"].get("fps", 0)
            frame["frame_index"] = t
            frame["timestamp_ms"] = int(t * (1000 / fps)) if fps > 0 else 0

        metadata = dict(data.get("metadata", {}))
        fps = metadata.get("fps", 0)
        metadata["total_frames"] = len(frames)
        metadata["duration_sec"] = len(frames) / fps if fps > 0 else 0.0
        metadata["normalization"] = {
            "method": "reference_based_normalization",
            "full_body_anchor": "neck_from_shoulders",
            "full_body_scale": "shoulder_distance",
            "face_anchor_index": FACE_NOSE_INDEX,
            "face_anchor_note": "0-based index on selected 68 face points",
            "arm_scale": "shoulder_elbow_distance_per_side",
            "hand_method": "bounding_box_per_frame",
            "absent_hand_rule": "keep_zero_and_exclude_from_bbox",
        }
        return {"metadata": metadata, "frames": frames}

    # ------------------------------------------------------------------
    # PRIVATE: normalization helpers
    # ------------------------------------------------------------------

    def _full_body_normalize(self, pose, face, lh, rh, lh_abs, rh_abs):
        ls = pose[POSE_KEYS.index("left_shoulder")]
        rs = pose[POSE_KEYS.index("right_shoulder")]
        neck = (ls + rs) / 2.0
        scale = max(float(np.linalg.norm(ls - rs)), EPS)

        pose = (pose - neck) / scale
        face = (face - neck) / scale
        lh   = (lh   - neck) / scale
        rh   = (rh   - neck) / scale

        lh[lh_abs] = 0.0
        rh[rh_abs] = 0.0
        return pose, face, lh, rh

    def _face_normalize(self, face_arr):
        return face_arr - face_arr[FACE_NOSE_INDEX].copy()

    def _arm_normalize(self, pose_arr):
        out = pose_arr.copy()
        for shoulder_key, elbow_key, arm_keys in [
            ("left_shoulder",  "left_elbow",  LEFT_ARM_KEYS),
            ("right_shoulder", "right_elbow", RIGHT_ARM_KEYS),
        ]:
            s = out[POSE_KEYS.index(shoulder_key)]
            e = out[POSE_KEYS.index(elbow_key)]
            scale = max(float(np.linalg.norm(s - e)), EPS)
            for k in arm_keys:
                out[POSE_KEYS.index(k)] /= scale
        return out

    def _hand_bbox_normalize(self, hand_arr, absent_mask):
        out = hand_arr.copy()
        if absent_mask.all():
            return out * 0.0
        valid = out[~absent_mask]
        x_min, x_max = float(valid[:, 0].min()), float(valid[:, 0].max())
        y_min, y_max = float(valid[:, 1].min()), float(valid[:, 1].max())
        w = max(x_max - x_min, EPS)
        h = max(y_max - y_min, EPS)
        cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
        out[:, 0] = (out[:, 0] - cx) / w
        out[:, 1] = (out[:, 1] - cy) / h
        out[absent_mask] = 0.0
        return out

    # ------------------------------------------------------------------
    # PRIVATE: array / frame utilities
    # ------------------------------------------------------------------

    def _crop_frames(self, frames: list) -> list:
        return frames[:self.target_frames] if len(frames) > self.target_frames else frames

    @staticmethod
    def _interpolate_xy_track(track_xy: np.ndarray) -> np.ndarray:
        """Linear interpolation over time for one landmark (shape [T, 2])."""
        T = track_xy.shape[0]
        result = track_xy.copy()
        missing = np.logical_and(result[:, 0] == 0.0, result[:, 1] == 0.0)
        if missing.all() or not missing.any():
            return result
        valid_idx = np.where(~missing)[0]
        for c in range(2):
            result[missing, c] = np.interp(
                np.where(missing)[0], valid_idx, result[valid_idx, c]
            )
        return result

    @staticmethod
    def _component_to_array(frames: list, component: str, keys: list) -> np.ndarray:
        T, K = len(frames), len(keys)
        arr = np.zeros((T, K, 2), dtype=np.float32)
        for t, frame in enumerate(frames):
            comp = frame["landmarks"][component]
            for k_idx, key in enumerate(keys):
                arr[t, k_idx, 0] = float(comp[key]["x"])
                arr[t, k_idx, 1] = float(comp[key]["y"])
        return arr

    @staticmethod
    def _write_array_back(frames: list, component: str, keys: list, arr: np.ndarray):
        for t, frame in enumerate(frames):
            comp = frame["landmarks"][component]
            for k_idx, key in enumerate(keys):
                comp[key]["x"] = float(arr[t, k_idx, 0])
                comp[key]["y"] = float(arr[t, k_idx, 1])

    @staticmethod
    def _to_arr(component_dict: dict, keys: list) -> np.ndarray:
        return np.array(
            [[float(component_dict[k]["x"]), float(component_dict[k]["y"])] for k in keys],
            dtype=np.float32,
        )

    @staticmethod
    def _write_arr(component_dict: dict, keys: list, arr: np.ndarray):
        for i, k in enumerate(keys):
            component_dict[k]["x"] = float(arr[i, 0])
            component_dict[k]["y"] = float(arr[i, 1])

    @staticmethod
    def _absent_mask(hand_arr: np.ndarray) -> np.ndarray:
        return np.logical_and(hand_arr[:, 0] == 0.0, hand_arr[:, 1] == 0.0)

    # ------------------------------------------------------------------
    # PRIVATE: landmark extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _lm_to_dict(landmarks, name_list: list) -> dict:
        if landmarks is None:
            return {name: {"x": 0.0, "y": 0.0, "z": 0.0} for name in name_list}
        return {
            name_list[idx]: {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)}
            for idx, lm in enumerate(landmarks.landmark)
        }

    @staticmethod
    def _get_xy(point_dict: dict) -> dict:
        return {"x": float(point_dict.get("x", 0.0)), "y": float(point_dict.get("y", 0.0))}

    @staticmethod
    def _mean_face_points(face_dict: dict, source_indices: list) -> dict:
        pts = np.array(
            [[float(face_dict[str(i)].get("x", 0.0)), float(face_dict[str(i)].get("y", 0.0))]
             for i in source_indices],
            dtype=np.float32,
        )
        mean_xy = pts.mean(axis=0)
        return {"source_indices": source_indices, "x": float(mean_xy[0]), "y": float(mean_xy[1])}

    # ------------------------------------------------------------------
    # PRIVATE: JSON I/O
    # ------------------------------------------------------------------

    def _json_to_array(self, path: str) -> np.ndarray:
        """Read one normalized JSON and return a float32 array [target_frames, L, 2].

        Output shape is always [target_frames, L, 2]:
        - Sequences longer  than target_frames are cropped.
        - Sequences shorter than target_frames are zero-padded at the end.

        Landmark axis order:
            [0:12]   pose       (POSE_KEYS)
            [12:33]  left_hand  (HAND_KEYS 0-20)
            [33:54]  right_hand (HAND_KEYS 0-20)
            [54:122] face       (FACE_KEYS 0-67)
        Total L = 12 + 21 + 21 + 68 = 122
        """
        data     = self._load_json(path)
        frames   = self._crop_frames(data["frames"])
        T_actual = len(frames)
        T        = self.target_frames                  # always output this many frames
        L        = len(POSE_KEYS) + len(HAND_KEYS) + len(HAND_KEYS) + len(FACE_KEYS)  # 122

        arr = np.zeros((T, L, 2), dtype=np.float32)   # zero-pad short sequences by default

        for t, frame in enumerate(frames):   # t only goes up to T_actual; rest stays zero
            lm     = frame["landmarks"]
            offset = 0

            # pose  [12 points]
            for k_idx, key in enumerate(POSE_KEYS):
                arr[t, offset + k_idx, 0] = float(lm["pose"][key]["x"])
                arr[t, offset + k_idx, 1] = float(lm["pose"][key]["y"])
            offset += len(POSE_KEYS)

            # left_hand  [21 points]
            for k_idx, key in enumerate(HAND_KEYS):
                arr[t, offset + k_idx, 0] = float(lm["left_hand"][key]["x"])
                arr[t, offset + k_idx, 1] = float(lm["left_hand"][key]["y"])
            offset += len(HAND_KEYS)

            # right_hand  [21 points]
            for k_idx, key in enumerate(HAND_KEYS):
                arr[t, offset + k_idx, 0] = float(lm["right_hand"][key]["x"])
                arr[t, offset + k_idx, 1] = float(lm["right_hand"][key]["y"])
            offset += len(HAND_KEYS)

            # face  [68 points]
            for k_idx, key in enumerate(FACE_KEYS):
                arr[t, offset + k_idx, 0] = float(lm["face"][key]["x"])
                arr[t, offset + k_idx, 1] = float(lm["face"][key]["y"])

        return arr   # [T, 122, 2]

    @staticmethod
    def _load_json(path: str) -> dict:
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def _save_json(data: dict, path: str):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _abs(base: str, rel: str) -> str:
        return os.path.join(base, rel) if not os.path.isabs(rel) else rel


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    preprocessor = Preprocessor(DEFAULT_CONFIG)
    preprocessor.run_all()