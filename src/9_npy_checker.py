import os
import json
import numpy as np
from tqdm import tqdm
from statistics import mean

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_PATH = os.path.join(BASE_DIR, "..", "dataset", "npy_dataset")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "dataset", "analysis_npy")

os.makedirs(OUTPUT_PATH, exist_ok=True)

TARGET_SHAPE = (90, 122, 2)
EPS = 1e-12

POSE_SLICE = slice(0, 12)
LEFT_HAND_SLICE = slice(12, 33)
RIGHT_HAND_SLICE = slice(33, 54)
FACE_SLICE = slice(54, 122)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def is_zero_pair(xy):
    return abs(float(xy[0])) <= EPS and abs(float(xy[1])) <= EPS


def count_zero_pairs(arr):
    """
    arr: (..., 2)
    return jumlah pasangan (x,y) yang keduanya nol
    """
    zero_mask = np.logical_and(np.abs(arr[..., 0]) <= EPS, np.abs(arr[..., 1]) <= EPS)
    return int(zero_mask.sum()), zero_mask


def analyze_single_npy(npy_path, dataset_index_lookup):
    arr = np.load(npy_path)

    rel_path = os.path.relpath(npy_path, INPUT_PATH).replace("\\", "/")
    meta = dataset_index_lookup.get(rel_path, {})

    structure_ok = tuple(arr.shape) == TARGET_SHAPE
    has_nan = bool(np.isnan(arr).any())
    has_inf = bool(np.isinf(arr).any())

    if arr.ndim == 3 and arr.shape[-1] == 2:
        pose = arr[:, POSE_SLICE, :]
        left_hand = arr[:, LEFT_HAND_SLICE, :]
        right_hand = arr[:, RIGHT_HAND_SLICE, :]
        face = arr[:, FACE_SLICE, :]

        pose_zero_count, pose_zero_mask = count_zero_pairs(pose)
        left_zero_count, left_zero_mask = count_zero_pairs(left_hand)
        right_zero_count, right_zero_mask = count_zero_pairs(right_hand)
        face_zero_count, face_zero_mask = count_zero_pairs(face)

        pose_total = pose.shape[0] * pose.shape[1]
        left_total = left_hand.shape[0] * left_hand.shape[1]
        right_total = right_hand.shape[0] * right_hand.shape[1]
        face_total = face.shape[0] * face.shape[1]

        # frame-level all-zero hand
        left_all_zero_frames = int(left_zero_mask.all(axis=1).sum())
        right_all_zero_frames = int(right_zero_mask.all(axis=1).sum())

        # landmark-level all-zero tracks sepanjang waktu
        left_all_zero_landmarks = int(left_zero_mask.all(axis=0).sum())
        right_all_zero_landmarks = int(right_zero_mask.all(axis=0).sum())

        result = {
            "file_name": os.path.basename(npy_path),
            "relative_path": rel_path,
            "video_id": meta.get("video_id"),
            "label": meta.get("label"),
            "label_id": meta.get("label_id"),
            "shape": list(arr.shape),
            "structure_ok": structure_ok,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "zero_statistics": {
                "pose": {
                    "zero_pairs": pose_zero_count,
                    "total_pairs": pose_total,
                    "zero_ratio": round(pose_zero_count / pose_total, 6) if pose_total > 0 else 0.0
                },
                "left_hand": {
                    "zero_pairs": left_zero_count,
                    "total_pairs": left_total,
                    "zero_ratio": round(left_zero_count / left_total, 6) if left_total > 0 else 0.0,
                    "all_zero_frames": left_all_zero_frames,
                    "all_zero_frame_ratio": round(left_all_zero_frames / arr.shape[0], 6),
                    "all_zero_landmarks": left_all_zero_landmarks
                },
                "right_hand": {
                    "zero_pairs": right_zero_count,
                    "total_pairs": right_total,
                    "zero_ratio": round(right_zero_count / right_total, 6) if right_total > 0 else 0.0,
                    "all_zero_frames": right_all_zero_frames,
                    "all_zero_frame_ratio": round(right_all_zero_frames / arr.shape[0], 6),
                    "all_zero_landmarks": right_all_zero_landmarks
                },
                "face": {
                    "zero_pairs": face_zero_count,
                    "total_pairs": face_total,
                    "zero_ratio": round(face_zero_count / face_total, 6) if face_total > 0 else 0.0
                }
            },
            "value_range": {
                "x_min": float(arr[:, :, 0].min()),
                "x_max": float(arr[:, :, 0].max()),
                "y_min": float(arr[:, :, 1].min()),
                "y_max": float(arr[:, :, 1].max())
            }
        }
    else:
        result = {
            "file_name": os.path.basename(npy_path),
            "relative_path": rel_path,
            "video_id": meta.get("video_id"),
            "label": meta.get("label"),
            "label_id": meta.get("label_id"),
            "shape": list(arr.shape),
            "structure_ok": structure_ok,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "error": "Unexpected ndim or last channel size"
        }

    return result


def aggregate_results(all_results):
    bad_shape_files = []
    nan_files = []
    inf_files = []

    per_label = {}
    overall = {
        "pose_zero": 0,
        "pose_total": 0,
        "left_zero": 0,
        "left_total": 0,
        "right_zero": 0,
        "right_total": 0,
        "face_zero": 0,
        "face_total": 0
    }

    all_zero_left_videos = []
    all_zero_right_videos = []

    for r in all_results:
        if not r.get("structure_ok", False):
            bad_shape_files.append({
                "file_name": r["file_name"],
                "relative_path": r["relative_path"],
                "shape": r["shape"]
            })

        if r.get("has_nan", False):
            nan_files.append({
                "file_name": r["file_name"],
                "relative_path": r["relative_path"]
            })

        if r.get("has_inf", False):
            inf_files.append({
                "file_name": r["file_name"],
                "relative_path": r["relative_path"]
            })

        label = r.get("label", "unknown")
        if label not in per_label:
            per_label[label] = {
                "num_files": 0,
                "pose_zero": 0,
                "pose_total": 0,
                "left_zero": 0,
                "left_total": 0,
                "right_zero": 0,
                "right_total": 0,
                "face_zero": 0,
                "face_total": 0,
                "all_zero_left_videos": 0,
                "all_zero_right_videos": 0
            }

        per_label[label]["num_files"] += 1

        zs = r["zero_statistics"]

        per_label[label]["pose_zero"] += zs["pose"]["zero_pairs"]
        per_label[label]["pose_total"] += zs["pose"]["total_pairs"]

        per_label[label]["left_zero"] += zs["left_hand"]["zero_pairs"]
        per_label[label]["left_total"] += zs["left_hand"]["total_pairs"]

        per_label[label]["right_zero"] += zs["right_hand"]["zero_pairs"]
        per_label[label]["right_total"] += zs["right_hand"]["total_pairs"]

        per_label[label]["face_zero"] += zs["face"]["zero_pairs"]
        per_label[label]["face_total"] += zs["face"]["total_pairs"]

        overall["pose_zero"] += zs["pose"]["zero_pairs"]
        overall["pose_total"] += zs["pose"]["total_pairs"]

        overall["left_zero"] += zs["left_hand"]["zero_pairs"]
        overall["left_total"] += zs["left_hand"]["total_pairs"]

        overall["right_zero"] += zs["right_hand"]["zero_pairs"]
        overall["right_total"] += zs["right_hand"]["total_pairs"]

        overall["face_zero"] += zs["face"]["zero_pairs"]
        overall["face_total"] += zs["face"]["total_pairs"]

        if zs["left_hand"]["all_zero_frame_ratio"] == 1.0:
            per_label[label]["all_zero_left_videos"] += 1
            all_zero_left_videos.append({
                "label": label,
                "video_id": r.get("video_id"),
                "file_name": r["file_name"]
            })

        if zs["right_hand"]["all_zero_frame_ratio"] == 1.0:
            per_label[label]["all_zero_right_videos"] += 1
            all_zero_right_videos.append({
                "label": label,
                "video_id": r.get("video_id"),
                "file_name": r["file_name"]
            })

    per_label_summary = {}
    for label, info in per_label.items():
        per_label_summary[label] = {
            "num_files": info["num_files"],
            "zero_ratio": {
                "pose": round(info["pose_zero"] / info["pose_total"], 6) if info["pose_total"] > 0 else 0.0,
                "left_hand": round(info["left_zero"] / info["left_total"], 6) if info["left_total"] > 0 else 0.0,
                "right_hand": round(info["right_zero"] / info["right_total"], 6) if info["right_total"] > 0 else 0.0,
                "face": round(info["face_zero"] / info["face_total"], 6) if info["face_total"] > 0 else 0.0,
            },
            "all_zero_left_videos": info["all_zero_left_videos"],
            "all_zero_right_videos": info["all_zero_right_videos"]
        }

    overall_summary = {
        "pose": round(overall["pose_zero"] / overall["pose_total"], 6) if overall["pose_total"] > 0 else 0.0,
        "left_hand": round(overall["left_zero"] / overall["left_total"], 6) if overall["left_total"] > 0 else 0.0,
        "right_hand": round(overall["right_zero"] / overall["right_total"], 6) if overall["right_total"] > 0 else 0.0,
        "face": round(overall["face_zero"] / overall["face_total"], 6) if overall["face_total"] > 0 else 0.0,
    }

    return {
        "num_files_analyzed": len(all_results),
        "target_shape": list(TARGET_SHAPE),
        "overall_zero_ratio": overall_summary,
        "per_label_summary": per_label_summary,
        "bad_shape_files": bad_shape_files,
        "nan_files": nan_files,
        "inf_files": inf_files,
        "all_zero_left_videos": all_zero_left_videos,
        "all_zero_right_videos": all_zero_right_videos
    }


def print_console_summary(summary):
    print("\n=== NPY ANALYSIS SUMMARY ===")
    print(f"Files analyzed: {summary['num_files_analyzed']}")
    print(f"Target shape : {tuple(summary['target_shape'])}")

    print("\nOverall zero ratio:")
    for comp, ratio in summary["overall_zero_ratio"].items():
        print(f"  {comp}: {ratio:.6f}")

    print("\nPer-label summary:")
    for label, info in summary["per_label_summary"].items():
        print(f"  Label: {label}")
        print(f"    num_files: {info['num_files']}")
        for comp, ratio in info["zero_ratio"].items():
            print(f"    {comp}: {ratio:.6f}")
        print(f"    all_zero_left_videos: {info['all_zero_left_videos']}")
        print(f"    all_zero_right_videos: {info['all_zero_right_videos']}")

    if summary["bad_shape_files"]:
        print("\nFiles with bad shape:")
        for item in summary["bad_shape_files"]:
            print(f"  - {item['file_name']} -> {item['shape']}")
    else:
        print("\nAll files have correct shape.")

    if summary["nan_files"]:
        print("\nFiles containing NaN:")
        for item in summary["nan_files"]:
            print(f"  - {item['file_name']}")
    else:
        print("No NaN found.")

    if summary["inf_files"]:
        print("\nFiles containing inf:")
        for item in summary["inf_files"]:
            print(f"  - {item['file_name']}")
    else:
        print("No inf found.")

    print(f"\nAll-zero left-hand videos : {len(summary['all_zero_left_videos'])}")
    print(f"All-zero right-hand videos: {len(summary['all_zero_right_videos'])}")


def build_dataset_index_lookup():
    dataset_index_file = os.path.join(INPUT_PATH, "dataset_index.json")
    if not os.path.exists(dataset_index_file):
        return {}

    dataset_index = load_json(dataset_index_file)
    lookup = {}

    for item in dataset_index:
        rel_path = os.path.relpath(item["npy_file"], INPUT_PATH).replace("\\", "/")
        lookup[rel_path] = item

    return lookup


def main():
    dataset_index_lookup = build_dataset_index_lookup()
    all_results = []

    for label in os.listdir(INPUT_PATH):
        label_dir = os.path.join(INPUT_PATH, label)
        if not os.path.isdir(label_dir):
            continue

        npy_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".npy")])

        print(f"\nChecking label: {label} ({len(npy_files)} files)")

        for file_name in tqdm(npy_files):
            npy_path = os.path.join(label_dir, file_name)
            result = analyze_single_npy(npy_path, dataset_index_lookup)
            all_results.append(result)

    summary = aggregate_results(all_results)

    detail_output = os.path.join(OUTPUT_PATH, "npy_analysis_detail.json")
    summary_output = os.path.join(OUTPUT_PATH, "npy_analysis_summary.json")

    with open(detail_output, "w") as f:
        json.dump(all_results, f, indent=2)

    with open(summary_output, "w") as f:
        json.dump(summary, f, indent=2)

    print_console_summary(summary)
    print(f"\nSaved detail analysis to: {detail_output}")
    print(f"Saved summary analysis to: {summary_output}")


if __name__ == "__main__":
    main()