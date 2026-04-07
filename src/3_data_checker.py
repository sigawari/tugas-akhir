import os
import json
from tqdm import tqdm
from statistics import mean

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_PATH = os.path.join(BASE_DIR, "..", "dataset", "cleaned_json")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "dataset", "analysis_2")
os.makedirs(OUTPUT_PATH, exist_ok=True)

EXPECTED_POSE_COUNT = 12
EXPECTED_HAND_COUNT = 21
EXPECTED_FACE_COUNT = 68

POSE_KEYS = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb"
]

HAND_KEYS = [str(i) for i in range(21)]
FACE_KEYS = [str(i) for i in range(68)]


def is_missing_xy(point_dict):
    return float(point_dict.get("x", 0.0)) == 0.0 and float(point_dict.get("y", 0.0)) == 0.0


def get_component_keys(component_name):
    if component_name == "pose":
        return POSE_KEYS
    if component_name in ("left_hand", "right_hand"):
        return HAND_KEYS
    if component_name == "face":
        return FACE_KEYS
    raise ValueError(f"Unknown component: {component_name}")


def compute_missing_gaps(frames, component_name, keys):
    """
    Hitung panjang gap missing beruntun per landmark.
    Return:
    {
        "left_wrist": [2, 1, 4],
        ...
    }
    """
    gap_dict = {}

    for key in keys:
        gaps = []
        current_gap = 0

        for frame in frames:
            point = frame["landmarks"][component_name][key]
            if is_missing_xy(point):
                current_gap += 1
            else:
                if current_gap > 0:
                    gaps.append(current_gap)
                    current_gap = 0

        if current_gap > 0:
            gaps.append(current_gap)

        gap_dict[key] = gaps

    return gap_dict


def summarize_gap_dict(gap_dict):
    all_gaps = []
    per_landmark = {}

    for key, gaps in gap_dict.items():
        if gaps:
            per_landmark[key] = {
                "num_gaps": len(gaps),
                "max_gap": max(gaps),
                "avg_gap": round(mean(gaps), 3)
            }
            all_gaps.extend(gaps)
        else:
            per_landmark[key] = {
                "num_gaps": 0,
                "max_gap": 0,
                "avg_gap": 0.0
            }

    if all_gaps:
        overall = {
            "num_gaps_total": len(all_gaps),
            "max_gap_overall": max(all_gaps),
            "avg_gap_overall": round(mean(all_gaps), 3)
        }
    else:
        overall = {
            "num_gaps_total": 0,
            "max_gap_overall": 0,
            "avg_gap_overall": 0.0
        }

    return {
        "overall": overall,
        "per_landmark": per_landmark
    }


def analyze_single_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    frames = data.get("frames", [])

    video_id = metadata.get("video_id", os.path.basename(file_path))
    label = metadata.get("action", "unknown")
    fps = metadata.get("fps", None)
    total_frames_meta = metadata.get("total_frames", None)

    structure_errors = []

    pose_missing = 0
    left_hand_missing = 0
    right_hand_missing = 0
    face_missing = 0

    pose_total = 0
    left_hand_total = 0
    right_hand_total = 0
    face_total = 0

    frame_missing_summary = []

    for frame_idx, frame in enumerate(frames):
        landmarks = frame.get("landmarks", {})

        pose = landmarks.get("pose", {})
        left_hand = landmarks.get("left_hand", {})
        right_hand = landmarks.get("right_hand", {})
        face = landmarks.get("face", {})

        # Struktur
        if len(pose) != EXPECTED_POSE_COUNT:
            structure_errors.append(
                f"frame {frame_idx}: pose count = {len(pose)} (expected {EXPECTED_POSE_COUNT})"
            )
        if len(left_hand) != EXPECTED_HAND_COUNT:
            structure_errors.append(
                f"frame {frame_idx}: left_hand count = {len(left_hand)} (expected {EXPECTED_HAND_COUNT})"
            )
        if len(right_hand) != EXPECTED_HAND_COUNT:
            structure_errors.append(
                f"frame {frame_idx}: right_hand count = {len(right_hand)} (expected {EXPECTED_HAND_COUNT})"
            )
        if len(face) != EXPECTED_FACE_COUNT:
            structure_errors.append(
                f"frame {frame_idx}: face count = {len(face)} (expected {EXPECTED_FACE_COUNT})"
            )

        frame_pose_missing = 0
        frame_left_missing = 0
        frame_right_missing = 0
        frame_face_missing = 0

        for key in POSE_KEYS:
            if key in pose:
                pose_total += 1
                if is_missing_xy(pose[key]):
                    pose_missing += 1
                    frame_pose_missing += 1

        for key in HAND_KEYS:
            if key in left_hand:
                left_hand_total += 1
                if is_missing_xy(left_hand[key]):
                    left_hand_missing += 1
                    frame_left_missing += 1

            if key in right_hand:
                right_hand_total += 1
                if is_missing_xy(right_hand[key]):
                    right_hand_missing += 1
                    frame_right_missing += 1

        for key in FACE_KEYS:
            if key in face:
                face_total += 1
                if is_missing_xy(face[key]):
                    face_missing += 1
                    frame_face_missing += 1

        frame_missing_summary.append({
            "frame_index": frame.get("frame_index", frame_idx),
            "timestamp_ms": frame.get("timestamp_ms", None),
            "missing_counts": {
                "pose": frame_pose_missing,
                "left_hand": frame_left_missing,
                "right_hand": frame_right_missing,
                "face": frame_face_missing
            }
        })

    # Gap analysis
    pose_gap_summary = summarize_gap_dict(compute_missing_gaps(frames, "pose", POSE_KEYS))
    left_gap_summary = summarize_gap_dict(compute_missing_gaps(frames, "left_hand", HAND_KEYS))
    right_gap_summary = summarize_gap_dict(compute_missing_gaps(frames, "right_hand", HAND_KEYS))
    face_gap_summary = summarize_gap_dict(compute_missing_gaps(frames, "face", FACE_KEYS))

    def safe_ratio(missing, total):
        return round(missing / total, 6) if total > 0 else 0.0

    result = {
        "video_id": video_id,
        "label": label,
        "file_name": os.path.basename(file_path),
        "fps": fps,
        "num_frames_found": len(frames),
        "total_frames_metadata": total_frames_meta,
        "structure_ok": len(structure_errors) == 0,
        "structure_errors": structure_errors,
        "missing_statistics": {
            "pose": {
                "missing_points": pose_missing,
                "total_points": pose_total,
                "missing_ratio": safe_ratio(pose_missing, pose_total)
            },
            "left_hand": {
                "missing_points": left_hand_missing,
                "total_points": left_hand_total,
                "missing_ratio": safe_ratio(left_hand_missing, left_hand_total)
            },
            "right_hand": {
                "missing_points": right_hand_missing,
                "total_points": right_hand_total,
                "missing_ratio": safe_ratio(right_hand_missing, right_hand_total)
            },
            "face": {
                "missing_points": face_missing,
                "total_points": face_total,
                "missing_ratio": safe_ratio(face_missing, face_total)
            }
        },
        "missing_gap_summary": {
            "pose": pose_gap_summary,
            "left_hand": left_gap_summary,
            "right_hand": right_gap_summary,
            "face": face_gap_summary
        },
        "frame_missing_summary": frame_missing_summary
    }

    return result


def aggregate_results(all_results):
    label_summary = {}
    structure_problem_files = []

    comp_missing_sum = {
        "pose": {"missing": 0, "total": 0},
        "left_hand": {"missing": 0, "total": 0},
        "right_hand": {"missing": 0, "total": 0},
        "face": {"missing": 0, "total": 0}
    }

    for result in all_results:
        label = result["label"]

        if not result["structure_ok"]:
            structure_problem_files.append({
                "file_name": result["file_name"],
                "video_id": result["video_id"],
                "errors": result["structure_errors"]
            })

        if label not in label_summary:
            label_summary[label] = {
                "num_videos": 0,
                "num_frames_total": 0,
                "missing": {
                    "pose": 0,
                    "left_hand": 0,
                    "right_hand": 0,
                    "face": 0
                },
                "total": {
                    "pose": 0,
                    "left_hand": 0,
                    "right_hand": 0,
                    "face": 0
                }
            }

        label_summary[label]["num_videos"] += 1
        label_summary[label]["num_frames_total"] += result["num_frames_found"]

        for comp in ["pose", "left_hand", "right_hand", "face"]:
            miss = result["missing_statistics"][comp]["missing_points"]
            total = result["missing_statistics"][comp]["total_points"]

            label_summary[label]["missing"][comp] += miss
            label_summary[label]["total"][comp] += total

            comp_missing_sum[comp]["missing"] += miss
            comp_missing_sum[comp]["total"] += total

    label_summary_out = {}
    for label, info in label_summary.items():
        label_summary_out[label] = {
            "num_videos": info["num_videos"],
            "num_frames_total": info["num_frames_total"],
            "missing_ratio": {
                comp: round(info["missing"][comp] / info["total"][comp], 6) if info["total"][comp] > 0 else 0.0
                for comp in ["pose", "left_hand", "right_hand", "face"]
            }
        }

    overall_missing_ratio = {
        comp: round(comp_missing_sum[comp]["missing"] / comp_missing_sum[comp]["total"], 6)
        if comp_missing_sum[comp]["total"] > 0 else 0.0
        for comp in ["pose", "left_hand", "right_hand", "face"]
    }

    return {
        "num_files_analyzed": len(all_results),
        "overall_missing_ratio": overall_missing_ratio,
        "per_label_summary": label_summary_out,
        "structure_problem_files": structure_problem_files
    }


def print_console_summary(aggregate_summary):
    print("\n=== ANALYSIS SUMMARY ===")
    print(f"Files analyzed: {aggregate_summary['num_files_analyzed']}")

    print("\nOverall missing ratio:")
    for comp, ratio in aggregate_summary["overall_missing_ratio"].items():
        print(f"  {comp}: {ratio:.6f}")

    print("\nPer-label summary:")
    for label, info in aggregate_summary["per_label_summary"].items():
        print(f"  Label: {label}")
        print(f"    num_videos: {info['num_videos']}")
        print(f"    num_frames_total: {info['num_frames_total']}")
        for comp, ratio in info["missing_ratio"].items():
            print(f"    {comp}: {ratio:.6f}")

    if aggregate_summary["structure_problem_files"]:
        print("\nFiles with structure problems:")
        for item in aggregate_summary["structure_problem_files"]:
            print(f"  - {item['file_name']}")
    else:
        print("\nNo structure problems found.")


def main():
    all_results = []

    for label in os.listdir(INPUT_PATH):
        label_dir = os.path.join(INPUT_PATH, label)
        if not os.path.isdir(label_dir):
            continue

        json_files = [f for f in os.listdir(label_dir) if f.endswith(".json")]

        print(f"\nAnalyzing label: {label} ({len(json_files)} files)")

        for file_name in tqdm(json_files):
            file_path = os.path.join(label_dir, file_name)
            result = analyze_single_file(file_path)
            all_results.append(result)

    aggregate_summary = aggregate_results(all_results)

    # Simpan file detail
    detail_output_path = os.path.join(OUTPUT_PATH, "analysis_detail.json")
    with open(detail_output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Simpan ringkasan
    summary_output_path = os.path.join(OUTPUT_PATH, "analysis_summary.json")
    with open(summary_output_path, "w") as f:
        json.dump(aggregate_summary, f, indent=2)

    print_console_summary(aggregate_summary)
    print(f"\nSaved detail analysis to: {detail_output_path}")
    print(f"Saved summary analysis to: {summary_output_path}")


if __name__ == "__main__":
    main()