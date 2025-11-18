import os, json, math, glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# -------------------------
# Konstanta MediaPipe
POSE_N = 33
HAND_N = 21
FACE_N = 468

# -------------------------
# Util normalisasi pose
def _shoulder_width(pose_xy):
    L, R = pose_xy[11], pose_xy[12]
    return np.linalg.norm(R - L) + 1e-6

def _mid_hip(pose_xyz):
    return (pose_xyz[23] + pose_xyz[24]) / 2.0

def _yaw_align_angle(pose_xy):
    L, R = pose_xy[11], pose_xy[12]
    v = R - L
    return math.atan2(v[1], v[0])

def _rotate_xyz(coords_xyz, angle_rad):
    c, s = math.cos(-angle_rad), math.sin(-angle_rad)
    rot = np.array([[c, -s, 0.0],
                    [s,  c, 0.0],
                    [0.0,0.0,1.0]], dtype=np.float32)
    return coords_xyz @ rot.T

# -------------------------
# Pilih & susun modalitas -> (K,4) per frame
def _select_modalities(frame, modal_cfg, face_stride):
    use_pose  = modal_cfg.get("use_pose", True)
    use_hands = modal_cfg.get("use_hands", True)
    use_face  = modal_cfg.get("use_face", True)

    K_pose = POSE_N if use_pose else 0
    K_lh   = HAND_N if use_hands else 0
    K_rh   = HAND_N if use_hands else 0
    K_face = len(np.arange(FACE_N)[::max(1, face_stride)]) if use_face else 0

    parts = []

    # Pose
    if use_pose:
        pose = np.array(frame.get("pose", []), dtype=np.float32)
        parts.append(pose if pose.shape == (POSE_N, 4) else np.zeros((POSE_N, 4), dtype=np.float32))

    # Hands
    if use_hands:
        lh = np.array(frame.get("left_hand", []), dtype=np.float32)
        rh = np.array(frame.get("right_hand", []), dtype=np.float32)
        parts.append(lh if lh.shape == (HAND_N, 4) else np.zeros((HAND_N, 4), dtype=np.float32))
        parts.append(rh if rh.shape == (HAND_N, 4) else np.zeros((HAND_N, 4), dtype=np.float32))

    # Face (subsample)
    if use_face:
        face = np.array(frame.get("face", []), dtype=np.float32)
        if face.shape == (FACE_N, 4):
            idx = np.arange(FACE_N, dtype=np.int32)[::max(1, face_stride)]
            parts.append(face[idx])
        else:
            parts.append(np.zeros((K_face, 4), dtype=np.float32))

    if len(parts) == 0:
        return np.zeros((0,4), dtype=np.float32), {"pose":0,"lh":0,"rh":0,"face":0}

    allj = np.vstack(parts)
    info = {"pose": K_pose, "lh": K_lh, "rh": K_rh, "face": K_face}
    return allj, info

def _coerce_frames_any_schema(raw):
    """
    Kembalikan list[dict_frame] dengan keys seperti 'pose', 'left_hand', 'right_hand', 'face'.
    Support skema:
      1) list of dicts
      2) dict dengan key 'frames' / 'sequence' / 'data' / 'items' → list of dicts
      3) dict of frames (keys '0','1',...) → ambil values terurut → list of dicts
      4) per-modal list: {'pose':[...], 'left_hand':[...], ...} → dirakit per time-step
      5) list of JSON-strings → parse tiap item
    """
    # 1) sudah list
    if isinstance(raw, list):
        if len(raw) == 0:
            return []
        # a) list of dict
        if isinstance(raw[0], dict):
            return raw
        # b) list of JSON-strings
        if isinstance(raw[0], str):
            parsed = [json.loads(s) for s in raw]
            if len(parsed) and isinstance(parsed[0], dict):
                return parsed
            raise ValueError("List berisi string, tapi bukan JSON frame yang valid.")
        raise ValueError("List ada, tapi elemennya bukan dict atau string.")

    # 2) dict dengan field pembungkus
    if isinstance(raw, dict):
        for key in ("frames", "sequence", "data", "items", "results"):
            if key in raw and isinstance(raw[key], list):
                lst = raw[key]
                if len(lst) == 0 or isinstance(lst[0], dict):
                    return lst

        # 3) dict of frames (keys -> values)
        # cek apakah semua values dict (frame)
        if all(isinstance(v, dict) for v in raw.values()):
            # urutkan berdasarkan key numerik jika bisa
            try:
                ordered = [raw[k] for k in sorted(raw.keys(), key=lambda x: int(x))]
            except:
                ordered = [raw[k] for k in sorted(raw.keys())]
            return ordered

        # 4) per-modal list
        # contoh: {'pose': [ (33,4), ...T ], 'left_hand': [...], 'right_hand': [...], 'face': [...] }
        modal_keys = ["pose", "left_hand", "right_hand", "face"]
        if any(k in raw for k in modal_keys):
            # tentukan T
            T_candidates = []
            for mk in modal_keys:
                if mk in raw and isinstance(raw[mk], list):
                    T_candidates.append(len(raw[mk]))
            if len(T_candidates) == 0:
                raise ValueError("Per-modal dict ada, tapi tidak ada list per waktu.")
            T = max(T_candidates)
            frames = []
            for t in range(T):
                fr = {}
                for mk in modal_keys:
                    if mk in raw and isinstance(raw[mk], list) and t < len(raw[mk]):
                        fr[mk] = raw[mk][t]
                frames.append(fr)
            return frames

    raise ValueError("Skema JSON tidak dikenali. Harusnya list-of-dicts atau varian yang didukung.")

# -------------------------
# Proses satu video JSON -> tensor (C_feat, K, T)
def json_video_to_tensor(json_path, modal_cfg, face_stride=3, vis_thresh=0.0, add_derivatives=True):
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    frames = _coerce_frames_any_schema(raw)
    T = len(frames)
    if T == 0:
        raise ValueError(f"Empty sequence: {json_path}")

    feats_list, masks_list = [], []
    K_ref = None

    for fr in frames:
        allj, _ = _select_modalities(fr, modal_cfg, face_stride)
        K = allj.shape[0]
        if K_ref is None:
            K_ref = K
        if K != K_ref:
            pad = np.zeros((K_ref - K, 4), dtype=np.float32)
            allj = np.vstack([allj, pad])
            K = K_ref

        # NORMALISASI
        pose_arr = np.array(fr.get("pose", []), dtype=np.float32) if isinstance(fr, dict) else None
        has_pose = (pose_arr is not None and pose_arr.shape == (POSE_N, 4))

        if has_pose:
            center = _mid_hip(pose_arr[:, :3])
            shoulder_w = _shoulder_width(pose_arr[:, :2])
            scale = shoulder_w if np.isfinite(shoulder_w) and shoulder_w > 1e-6 else 1.0
            ang = _yaw_align_angle(pose_arr[:, :2])
        else:
            vis_all = allj[:, 3:4]
            if np.isfinite(vis_all).all() and vis_all.sum() > 0:
                center = (allj[:, :3] * np.clip(vis_all,0,1)).sum(0) / (np.clip(vis_all,0,1).sum() + 1e-6)
            else:
                center = np.zeros(3, dtype=np.float32)
            std_xy = allj[:, :2].std() if np.isfinite(allj[:, :2]).any() else 0.0
            scale = std_xy if np.isfinite(std_xy) and std_xy > 1e-6 else 1.0
            ang = 0.0

        coords = allj[:, :3] - center
        coords = coords / scale
        coords = _rotate_xyz(coords, ang)
        vis = allj[:, 3:4]
        mask = (vis >= 0.0).astype(np.float32)

        feats = np.concatenate([coords, vis], axis=1)  # (K,4)

        # >>> Yang kemarin lupa:
        feats_list.append(feats)
        masks_list.append(mask)

    X = np.stack(feats_list, axis=0)           # (T,K,4)
    M = np.stack(masks_list, axis=0)           # (T,K,1)

    if add_derivatives:
        dX = np.diff(X[:, :, :3], axis=0, prepend=X[:1, :, :3])  # (T,K,3)
        speed = np.linalg.norm(dX, axis=2, keepdims=True)        # (T,K,1)
        F = np.concatenate([X, dX, speed, M], axis=2)            # (T,K,9)
    else:
        F = np.concatenate([X, M], axis=2)                       # (T,K,5)

    # ke format ResNet-2D: (C_feat, K, T)
    X2D = np.transpose(F, (2,1,0)).astype(np.float32)            # (C_feat, K, T)

    # >>> assert setelah X2D dibuat
    _, K_final, T_final = X2D.shape
    assert K_final > 0 and T_final > 0, f"K or T invalid: {K_final}, {T_final} for file {json_path}"

    return X2D

# -------------------------
# Dataset K-Fold untuk ResNet
class BISINDOResNetDatasetKFold(Dataset):
    def __init__(self,
                 fold_number=1,
                 is_train=True,
                 split_dir="splits",
                 modal_cfg=None,
                 face_stride=3,
                 jitter_std=0.0,          # augmentasi kecil di koordinat
                 time_mask_prob=0.0,      # hapus frame random (lalu interp)
                 print_path=False):
        """
        modal_cfg: dict {"use_pose":bool, "use_hands":bool, "use_face":bool}
        """
        self.print_path = print_path
        self.modal_cfg = modal_cfg or {"use_pose": True, "use_hands": True, "use_face": True}
        self.face_stride = face_stride
        self.jitter_std = float(jitter_std)
        self.time_mask_prob = float(time_mask_prob)

        split_file = os.path.join(split_dir, f"5fold_split_fold{fold_number}.json")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"{split_file} not found. Buat dulu dengan make_splits.py")

        with open(split_file, "r") as f:
            split = json.load(f)

        self.labels = [lab.lower() for lab in split.get("labels", [])]  # ex: ["halo","terima_kasih"]
        files = split["train_files"] if is_train else split["val_files"]
        self.file_paths = [os.path.join(os.getcwd(), p) if not os.path.isabs(p) else p for p in files]

        if self.labels:
            self.label_map = {lab:i for i,lab in enumerate(self.labels)}
        else:
            uniq = sorted({os.path.basename(os.path.dirname(p)).lower() for p in self.file_paths})
            self.label_map = {lab:i for i,lab in enumerate(uniq)}

        if print_path:
            split_name = "Train" if is_train else "Val"
            print(f"{split_name} fold {fold_number} | total files: {len(self.file_paths)}")
            print("label_map:", self.label_map)
            print("modal_cfg:", self.modal_cfg, "| face_stride:", self.face_stride)

    def __len__(self):
        return len(self.file_paths)

    def _label_from_parent(self, path):
        lab = os.path.basename(os.path.dirname(path)).lower()
        if lab not in self.label_map:
            raise KeyError(f"Unknown label '{lab}' in path {path}")
        return self.label_map[lab]

    def _augment(self, X2D):
        # X2D: (C_feat, K, T)
        C,K,T = X2D.shape
        out = X2D.copy()

        # jitter kecil (hanya kanal koordinat & delta: indeks [0:3] dan [4:7])
        if self.jitter_std > 0:
            # hati2 jika C_feat < 7
            coord_idx = [i for i in [0,1,2,4,5,6] if i < C]
            noise = np.random.normal(0, self.jitter_std, size=(len(coord_idx), K, T)).astype(np.float32)
            out[coord_idx, :, :] += noise

        # time masking sederhana: drop beberapa frame lalu linear interp
        if self.time_mask_prob > 0 and T > 3:
            mask = (np.random.rand(T) < self.time_mask_prob)
            if mask.any() and mask.sum() < T:
                keep_idx = np.where(~mask)[0]
                # linear interp sepanjang waktu untuk tiap channel & joint
                for c in range(C):
                    for k in range(K):
                        out[c, k, :] = np.interp(np.arange(T), keep_idx, out[c, k, keep_idx])
        return out

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        y = self._label_from_parent(path)
        X2D = json_video_to_tensor(
            json_path=path,
            modal_cfg=self.modal_cfg,
            face_stride=self.face_stride,
            vis_thresh=0.0,
            add_derivatives=True
        )  # (C_feat, K, T)

        X2D = self._augment(X2D)
        X = torch.from_numpy(X2D).float()
        y = torch.tensor(y, dtype=torch.long)
        return X, y

# -------------------------
# Collate & contoh pakai
def collate_fn(batch):
    xs, ys = zip(*batch)
    X = torch.stack(xs, dim=0)        # (B, C, K, T)
    y = torch.stack(ys, dim=0)        # (B,)
    return X, y

if __name__ == "__main__":
    # Contoh: FULL (pose+hands+face), wajah di-subsample tiap 3 titik
    modal_cfg = {"use_pose": True, "use_hands": True, "use_face": True}

    train_ds = BISINDOResNetDatasetKFold(
        fold_number=1, is_train=True,
        modal_cfg=modal_cfg, face_stride=3,
        jitter_std=0.01, time_mask_prob=0.0,
        print_path=True
    )
    val_ds = BISINDOResNetDatasetKFold(
        fold_number=1, is_train=False,
        modal_cfg=modal_cfg, face_stride=3,
        print_path=True
    )

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2, collate_fn=collate_fn)
    X, y = next(iter(train_loader))
    print("Batch X:", X.shape)  # (B, C_feat, K, T)
    print("Batch y:", y.shape, y.dtype)
