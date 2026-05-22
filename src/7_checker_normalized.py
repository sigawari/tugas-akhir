import os
import json
from tqdm import tqdm
from statistics import mean

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_PATH = os.path.join(BASE_DIR, "..", "dataset", "normalized_json")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "dataset", "analysis_normalized")

os.makedirs(OUTPUT_PATH, exist_ok=True)

TARGET_FRAMES = 90
EPS = 1e-12

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


def is_zero_xy(point_dict):
    x = float(point_dict.get("x", 0.0))
    y = float(point_dict.get("y", 0.0))
    return abs(x) <= EPS and abs(y) <= EPS


def compute_zero_gaps(frames, component_name, keys):
    """
    Hitung gap nol beruntun per landmark.
    Cocok untuk melihat apakah nol muncul sesekali atau sepanjang sequence.
    """
    gap_dict = {}

    for key in keys:
        gaps = []
        current_gap = 0

        for frame in frames:
            point = frame["landmarks"][component_name][key]
            if is_zero_xy(point):
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
                "num_zero_gaps": len(gaps),
                "max_zero_gap": max(gaps),
                "avg_zero_gap": round(mean(gaps), 3)
            }
            all_gaps.extend(gaps)
        else:
            per_landmark[key] = {
                "num_zero_gaps": 0,
                "max_zero_gap": 0,
                "avg_zero_gap": 0.0
            }

    if all_gaps:
        overall = {
            "num_zero_gaps_total": len(all_gaps),
            "max_zero_gap_overall": max(all_gaps),
            "avg_zero_gap_overall": round(mean(all_gaps), 3)
        }
    else:
        overall = {
            "num_zero_gaps_total": 0,
            "max_zero_gap_overall": 0,
            "avg_zero_gap_overall": 0.0
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

    structure_errors = []
    sequence_length_ok = len(frames) == TARGET_FRAMES

    pose_zero = 0
    left_hand_zero = 0
    right_hand_zero = 0
    face_zero = 0

    pose_total = 0
    left_hand_total = 0
    right_hand_total = 0
    face_total = 0

    frame_zero_summary = []

    left_hand_all_zero_frames = 0
    right_hand_all_zero_frames = 0

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

        frame_pose_zero = 0
        frame_left_zero = 0
        frame_right_zero = 0
        frame_face_zero = 0

        # Pose
        for key in POSE_KEYS:
            if key in pose:
                pose_total += 1
                if is_zero_xy(pose[key]):
                    pose_zero += 1
                    frame_pose_zero += 1

        # Left hand
        current_left_all_zero = True
        for key in HAND_KEYS:
            if key in left_hand:
                left_hand_total += 1
                if is_zero_xy(left_hand[key]):
                    left_hand_zero += 1
                    frame_left_zero += 1
                else:
                    current_left_all_zero = False

        if current_left_all_zero:
            left_hand_all_zero_frames += 1

        # Right hand
        current_right_all_zero = True
        for key in HAND_KEYS:
            if key in right_hand:
                right_hand_total += 1
                if is_zero_xy(right_hand[key]):
                    right_hand_zero += 1
                    frame_right_zero += 1
                else:
                    current_right_all_zero = False

        if current_right_all_zero:
            right_hand_all_zero_frames += 1

        # Face
        for key in FACE_KEYS:
            if key in face:
                face_total += 1
                if is_zero_xy(face[key]):
                    face_zero += 1
                    frame_face_zero += 1

        frame_zero_summary.append({
            "frame_index": frame.get("frame_index", frame_idx),
            "timestamp_ms": frame.get("timestamp_ms", None),
            "zero_counts": {
                "pose": frame_pose_zero,
                "left_hand": frame_left_zero,
                "right_hand": frame_right_zero,
                "face": frame_face_zero
            },
            "all_zero_components": {
                "left_hand": current_left_all_zero,
                "right_hand": current_right_all_zero
            }
        })

    def safe_ratio(zero_count, total_count):
        return round(zero_count / total_count, 6) if total_count > 0 else 0.0

    result = {
        "video_id": video_id,
        "label": label,
        "file_name": os.path.basename(file_path),
        "fps": fps,
        "num_frames_found": len(frames),
        "sequence_length_ok": sequence_length_ok,
        "structure_ok": len(structure_errors) == 0,
        "structure_errors": structure_errors,
        "zero_statistics": {
            "pose": {
                "zero_points": pose_zero,
                "total_points": pose_total,
                "zero_ratio": safe_ratio(pose_zero, pose_total)
            },
            "left_hand": {
                "zero_points": left_hand_zero,
                "total_points": left_hand_total,
                "zero_ratio": safe_ratio(left_hand_zero, left_hand_total),
                "all_zero_frames": left_hand_all_zero_frames,
                "all_zero_frame_ratio": round(left_hand_all_zero_frames / len(frames), 6) if len(frames) > 0 else 0.0
            },
            "right_hand": {
                "zero_points": right_hand_zero,
                "total_points": right_hand_total,
                "zero_ratio": safe_ratio(right_hand_zero, right_hand_total),
                "all_zero_frames": right_hand_all_zero_frames,
                "all_zero_frame_ratio": round(right_hand_all_zero_frames / len(frames), 6) if len(frames) > 0 else 0.0
            },
            "face": {
                "zero_points": face_zero,
                "total_points": face_total,
                "zero_ratio": safe_ratio(face_zero, face_total)
            }
        },
        "zero_gap_summary": {
            "left_hand": summarize_gap_dict(compute_zero_gaps(frames, "left_hand", HAND_KEYS)),
            "right_hand": summarize_gap_dict(compute_zero_gaps(frames, "right_hand", HAND_KEYS)),
            "pose": summarize_gap_dict(compute_zero_gaps(frames, "pose", POSE_KEYS)),
            "face": summarize_gap_dict(compute_zero_gaps(frames, "face", FACE_KEYS))
        },
        "frame_zero_summary": frame_zero_summary
    }

    return result


def aggregate_results(all_results):
    label_summary = {}
    structure_problem_files = []
    bad_length_files = []

    comp_zero_sum = {
        "pose": {"zero": 0, "total": 0},
        "left_hand": {"zero": 0, "total": 0},
        "right_hand": {"zero": 0, "total": 0},
        "face": {"zero": 0, "total": 0},
    }

    all_zero_hand_videos = {
        "left_hand": [],
        "right_hand": []
    }

    for result in all_results:
        label = result["label"]

        if not result["structure_ok"]:
            structure_problem_files.append({
                "file_name": result["file_name"],
                "video_id": result["video_id"],
                "errors": result["structure_errors"]
            })

        if not result["sequence_length_ok"]:
            bad_length_files.append({
                "file_name": result["file_name"],
                "video_id": result["video_id"],
                "num_frames_found": result["num_frames_found"]
            })

        if label not in label_summary:
            label_summary[label] = {
                "num_videos": 0,
                "num_frames_total": 0,
                "zero": {
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
                },
                "all_zero_video_count": {
                    "left_hand": 0,
                    "right_hand": 0
                }
            }

        label_summary[label]["num_videos"] += 1
        label_summary[label]["num_frames_total"] += result["num_frames_found"]

        for comp in ["pose", "left_hand", "right_hand", "face"]:
            zero_points = result["zero_statistics"][comp]["zero_points"]
            total_points = result["zero_statistics"][comp]["total_points"]

            label_summary[label]["zero"][comp] += zero_points
            label_summary[label]["total"][comp] += total_points

            comp_zero_sum[comp]["zero"] += zero_points
            comp_zero_sum[comp]["total"] += total_points

        # video dengan satu hand kosong total sepanjang sequence
        if result["zero_statistics"]["left_hand"]["all_zero_frame_ratio"] == 1.0:
            label_summary[label]["all_zero_video_count"]["left_hand"] += 1
            all_zero_hand_videos["left_hand"].append({
                "label": result["label"],
                "video_id": result["video_id"],
                "file_name": result["file_name"]
            })

        if result["zero_statistics"]["right_hand"]["all_zero_frame_ratio"] == 1.0:
            label_summary[label]["all_zero_video_count"]["right_hand"] += 1
            all_zero_hand_videos["right_hand"].append({
                "label": result["label"],
                "video_id": result["video_id"],
                "file_name": result["file_name"]
            })

    label_summary_out = {}
    for label, info in label_summary.items():
        label_summary_out[label] = {
            "num_videos": info["num_videos"],
            "num_frames_total": info["num_frames_total"],
            "zero_ratio": {
                comp: round(info["zero"][comp] / info["total"][comp], 6) if info["total"][comp] > 0 else 0.0
                for comp in ["pose", "left_hand", "right_hand", "face"]
            },
            "all_zero_video_count": info["all_zero_video_count"]
        }

    overall_zero_ratio = {
        comp: round(comp_zero_sum[comp]["zero"] / comp_zero_sum[comp]["total"], 6)
        if comp_zero_sum[comp]["total"] > 0 else 0.0
        for comp in ["pose", "left_hand", "right_hand", "face"]
    }

    return {
        "num_files_analyzed": len(all_results),
        "target_frames": TARGET_FRAMES,
        "overall_zero_ratio": overall_zero_ratio,
        "per_label_summary": label_summary_out,
        "all_zero_hand_videos": all_zero_hand_videos,
        "bad_length_files": bad_length_files,
        "structure_problem_files": structure_problem_files
    }


def print_console_summary(summary):
    print("\n=== NORMALIZED ANALYSIS SUMMARY ===")
    print(f"Files analyzed: {summary['num_files_analyzed']}")
    print(f"Target frames : {summary['target_frames']}")

    print("\nOverall zero ratio:")
    for comp, ratio in summary["overall_zero_ratio"].items():
        print(f"  {comp}: {ratio:.6f}")

    print("\nPer-label summary:")
    for label, info in summary["per_label_summary"].items():
        print(f"  Label: {label}")
        print(f"    num_videos: {info['num_videos']}")
        print(f"    num_frames_total: {info['num_frames_total']}")
        for comp, ratio in info["zero_ratio"].items():
            print(f"    {comp}: {ratio:.6f}")
        print(f"    all_zero_left_hand_videos: {info['all_zero_video_count']['left_hand']}")
        print(f"    all_zero_right_hand_videos: {info['all_zero_video_count']['right_hand']}")

    if summary["bad_length_files"]:
        print("\nFiles with non-90 frame length:")
        for item in summary["bad_length_files"]:
            print(f"  - {item['file_name']} ({item['num_frames_found']} frames)")
    else:
        print("\nAll files have correct frame length.")

    if summary["structure_problem_files"]:
        print("\nFiles with structure problems:")
        for item in summary["structure_problem_files"]:
            print(f"  - {item['file_name']}")
    else:
        print("No structure problems found.")

    print("\nAll-zero hand videos:")
    print(f"  left_hand : {len(summary['all_zero_hand_videos']['left_hand'])}")
    print(f"  right_hand: {len(summary['all_zero_hand_videos']['right_hand'])}")


def main():
    all_results = []

    for label in os.listdir(INPUT_PATH):
        label_dir = os.path.join(INPUT_PATH, label)
        if not os.path.isdir(label_dir):
            continue

        json_files = [f for f in os.listdir(label_dir) if f.endswith(".json")]

        print(f"\nAnalyzing normalized label: {label} ({len(json_files)} files)")

        for file_name in tqdm(json_files):
            file_path = os.path.join(label_dir, file_name)
            result = analyze_single_file(file_path)
            all_results.append(result)

    summary = aggregate_results(all_results)

    detail_output_path = os.path.join(OUTPUT_PATH, "normalized_analysis_detail.json")
    summary_output_path = os.path.join(OUTPUT_PATH, "normalized_analysis_summary.json")

    with open(detail_output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    with open(summary_output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print_console_summary(summary)
    print(f"\nSaved detail analysis to: {detail_output_path}")
    print(f"Saved summary analysis to: {summary_output_path}")


if __name__ == "__main__":
    main()