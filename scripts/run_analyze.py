# scripts/run_analyze.py
"""
Analisis kondisi dataset di semua stage preprocessing.

Jalankan:
    python scripts/run_analyze.py                    # analisis semua stage
    python scripts/run_analyze.py --stage clean      # hanya stage tertentu
    python scripts/run_analyze.py --stage normalize  # hanya stage tertentu
"""
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import json
import argparse
import numpy as np
from tqdm import tqdm
from src.utils import load_config, get_logger

logger = get_logger("Analyze")

# =========================
# KONSTANTA
# =========================

TARGET_FRAMES = 90

POSE_KEYS = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
]
HAND_KEYS = [str(i) for i in range(21)]
FACE_KEYS = [str(i) for i in range(68)]

EXPECTED_COUNTS = {
    "pose": len(POSE_KEYS),        # 12
    "left_hand": len(HAND_KEYS),   # 21
    "right_hand": len(HAND_KEYS),  # 21
    "face": len(FACE_KEYS),        # 68
}


# =========================
# CORE ANALYSIS
# =========================

def is_zero(point: dict) -> bool:
    return abs(float(point.get("x", 0.0))) < 1e-9 and abs(float(point.get("y", 0.0))) < 1e-9


def analyze_file(file_path: Path) -> dict:
    with open(file_path) as f:
        data = json.load(f)

    meta   = data.get("metadata", {})
    frames = data.get("frames", [])
    n      = len(frames)

    result = {
        "file":       file_path.name,
        "label":      meta.get("action", file_path.parent.name),
        "fps":        meta.get("fps"),
        "num_frames": n,
        "frame_ok":   n == TARGET_FRAMES,
        "structure_errors": [],
        "missing": {
            "pose":       {"count": 0, "total": 0},
            "left_hand":  {"count": 0, "total": 0},
            "right_hand": {"count": 0, "total": 0},
            "face":       {"count": 0, "total": 0},
        },
        "hand_absent_frames": {   # frame di mana SELURUH tangan = 0
            "left_hand":  0,
            "right_hand": 0,
        },
        "problem": False,
        "problem_reasons": [],
    }

    for fi, frame in enumerate(frames):
        lm = frame.get("landmarks", {})

        for comp, keys in [
            ("pose",       POSE_KEYS),
            ("left_hand",  HAND_KEYS),
            ("right_hand", HAND_KEYS),
            ("face",       FACE_KEYS),
        ]:
            comp_data = lm.get(comp, {})

            # struktur
            if len(comp_data) != EXPECTED_COUNTS[comp]:
                result["structure_errors"].append(
                    f"frame {fi}: {comp} count={len(comp_data)} (expected {EXPECTED_COUNTS[comp]})"
                )

            # missing per titik
            zero_count = sum(1 for k in keys if k in comp_data and is_zero(comp_data[k]))
            result["missing"][comp]["count"] += zero_count
            result["missing"][comp]["total"] += len(keys)

            # seluruh tangan absen di frame ini?
            if comp in ("left_hand", "right_hand"):
                all_zero = all(is_zero(comp_data[k]) for k in keys if k in comp_data)
                if all_zero:
                    result["hand_absent_frames"][comp] += 1

    # hitung ratio
    for comp in result["missing"]:
        m = result["missing"][comp]
        m["ratio"] = round(m["count"] / m["total"], 4) if m["total"] > 0 else 0.0

    # tentukan apakah file bermasalah
    reasons = []
    if not result["frame_ok"]:
        reasons.append(f"frame count {n} != {TARGET_FRAMES}")
    if result["structure_errors"]:
        reasons.append(f"{len(result['structure_errors'])} structure error(s)")
    if result["missing"]["pose"]["ratio"] > 0.1:
        reasons.append(f"pose missing {result['missing']['pose']['ratio']:.1%}")
    if result["missing"]["face"]["ratio"] > 0.1:
        reasons.append(f"face missing {result['missing']['face']['ratio']:.1%}")
    lh_absent_ratio = result["hand_absent_frames"]["left_hand"]  / max(n, 1)
    rh_absent_ratio = result["hand_absent_frames"]["right_hand"] / max(n, 1)
    if lh_absent_ratio > 0.5:
        reasons.append(f"left_hand absen {lh_absent_ratio:.1%} frame")
    if rh_absent_ratio > 0.5:
        reasons.append(f"right_hand absen {rh_absent_ratio:.1%} frame")

    result["problem"]         = len(reasons) > 0
    result["problem_reasons"] = reasons
    return result


def _detect_one_handed_labels(all_results: list) -> dict:
    """
    Deteksi label yang memang one-handed secara konsisten.

    Suatu label dianggap one-handed untuk sisi tertentu jika
    SEMUA file dalam label itu absen 100% di sisi tersebut.
    Ini bukan masalah — memang gesturnya tidak pakai tangan itu.

    Return: {"left_hand": {"belum", "maaf", ...}, "right_hand": set()}
    """
    from collections import defaultdict

    # kumpulkan absent_ratio per label per sisi
    label_absent = defaultdict(lambda: {"left_hand": [], "right_hand": []})
    for r in all_results:
        lbl = r["label"]
        n   = max(r["num_frames"], 1)
        for side in ("left_hand", "right_hand"):
            ratio = r["hand_absent_frames"][side] / n
            label_absent[lbl][side].append(ratio)

    one_handed = {"left_hand": set(), "right_hand": set()}
    for lbl, sides in label_absent.items():
        for side in ("left_hand", "right_hand"):
            ratios = sides[side]
            # one-handed: semua file absen 100% di sisi ini
            if ratios and all(r == 1.0 for r in ratios):
                one_handed[side].add(lbl)

    return one_handed


def analyze_stage(stage_dir: Path, stage_name: str) -> dict:
    logger.info(f"Menganalisis stage: {stage_name} → {stage_dir}")

    if not stage_dir.exists():
        logger.warning(f"  Direktori tidak ditemukan: {stage_dir}")
        return {}

    all_results = []

    for label_dir in sorted(stage_dir.iterdir()):
        if not label_dir.is_dir():
            continue

        json_files = sorted(label_dir.glob("*.json"))
        if not json_files:
            continue

        logger.info(f"  [{label_dir.name}]  {len(json_files)} file")

        for jf in tqdm(json_files, desc=f"  {label_dir.name}", leave=False):
            all_results.append(analyze_file(jf))

    if not all_results:
        return {}

    # ── deteksi label one-handed sebelum menilai "bermasalah" ──
    one_handed = _detect_one_handed_labels(all_results)

    # tandai one-handed di tiap result, lalu re-evaluasi problem
    for r in all_results:
        lbl = r["label"]
        n   = max(r["num_frames"], 1)
        new_reasons = []

        for reason in r["problem_reasons"]:
            side = None
            if reason.startswith("left_hand"):
                side = "left_hand"
            elif reason.startswith("right_hand"):
                side = "right_hand"

            # buang reason kalau memang one-handed untuk label ini
            if side and lbl in one_handed[side]:
                continue
            new_reasons.append(reason)

        r["one_handed"] = {
            side: lbl in one_handed[side]
            for side in ("left_hand", "right_hand")
        }
        r["problem_reasons"] = new_reasons
        r["problem"]         = len(new_reasons) > 0

    problem_files = [r for r in all_results if r["problem"]]

    # ── ringkasan per label ──
    label_summary = {}
    for r in all_results:
        lbl = r["label"]
        if lbl not in label_summary:
            label_summary[lbl] = {
                "total_files":  0,
                "problem_files": 0,
                "frame_ok":     0,
                "one_handed":   {
                    "left_hand":  lbl in one_handed["left_hand"],
                    "right_hand": lbl in one_handed["right_hand"],
                },
                "missing_ratio": {c: [] for c in EXPECTED_COUNTS},
            }
        s = label_summary[lbl]
        s["total_files"]   += 1
        s["problem_files"] += int(r["problem"])
        s["frame_ok"]      += int(r["frame_ok"])
        for comp in EXPECTED_COUNTS:
            s["missing_ratio"][comp].append(r["missing"][comp]["ratio"])

    for lbl, s in label_summary.items():
        for comp in EXPECTED_COUNTS:
            ratios = s["missing_ratio"][comp]
            s["missing_ratio"][comp] = {
                "mean": round(float(np.mean(ratios)), 4),
                "max":  round(float(np.max(ratios)),  4),
            }

    summary = {
        "stage":          stage_name,
        "total_files":    len(all_results),
        "problem_files":  len(problem_files),
        "all_frame_ok":   all(r["frame_ok"] for r in all_results),
        "one_handed_labels": {
            side: sorted(one_handed[side])
            for side in ("left_hand", "right_hand")
        },
        "per_label":      label_summary,
        "problem_list":   [
            {
                "file":    r["file"],
                "label":   r["label"],
                "reasons": r["problem_reasons"],
                "frames":  r["num_frames"],
                "missing": {c: r["missing"][c]["ratio"] for c in EXPECTED_COUNTS},
            }
            for r in problem_files
        ],
    }
    return summary


# =========================
# PRINT HELPERS
# =========================

def print_summary(summary: dict):
    if not summary:
        return

    stage = summary["stage"]
    print(f"\n{'='*60}")
    print(f"  STAGE: {stage.upper()}")
    print(f"{'='*60}")
    print(f"  Total file   : {summary['total_files']}")
    print(f"  File OK      : {summary['total_files'] - summary['problem_files']}")
    print(f"  File masalah : {summary['problem_files']}")
    print(f"  Semua 90 frame: {'✓' if summary['all_frame_ok'] else '✗'}")

    print(f"\n  Per-label (missing ratio mean | max):")
    for lbl, s in summary["per_label"].items():
        ok  = s["frame_ok"]
        tot = s["total_files"]
        print(f"    {lbl:<18} frame_ok={ok}/{tot}  "
              f"pose={s['missing_ratio']['pose']['mean']:.3f}  "
              f"lh={s['missing_ratio']['left_hand']['mean']:.3f}  "
              f"rh={s['missing_ratio']['right_hand']['mean']:.3f}  "
              f"face={s['missing_ratio']['face']['mean']:.3f}")

    # one-handed labels
    oh = summary.get("one_handed_labels", {})
    oh_lh = oh.get("left_hand",  [])
    oh_rh = oh.get("right_hand", [])
    if oh_lh or oh_rh:
        print(f"\n  ℹ One-handed labels (normal, bukan masalah):")
        if oh_lh:
            print(f"    Hanya tangan kanan: {', '.join(oh_lh)}")
        if oh_rh:
            print(f"    Hanya tangan kiri : {', '.join(oh_rh)}")

    if summary["problem_list"]:
        print(f"\n  ⚠ File bermasalah (di luar one-handed):")
        for p in summary["problem_list"]:
            print(f"    [{p['label']}] {p['file']}")
            for reason in p["reasons"]:
                print(f"      → {reason}")
    else:
        print(f"\n  ✓ Tidak ada file bermasalah.")


# =========================
# MAIN
# =========================

STAGE_MAP = {
    "extract":   "landmarks_extracted",
    "select":    "landmarks_selected",
    "clean":     "landmarks_interpolated",
    "normalize": "landmarks_normalization",
}

def main():
    parser = argparse.ArgumentParser(description="Analisis kondisi dataset per stage")
    parser.add_argument(
        "--stage",
        type=str,
        choices=list(STAGE_MAP.keys()) + ["all"],
        default="all",
        help="Stage yang ingin dianalisis (default: all)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Simpan hasil analisis ke outputs/logs/analysis_<stage>.json"
    )
    args = parser.parse_args()

    config  = load_config("preprocess.yaml")
    interim = root_dir / config["paths"].get("interim_dir", "data/interim")

    stages = list(STAGE_MAP.keys()) if args.stage == "all" else [args.stage]

    all_summaries = {}
    for stage in stages:
        stage_dir = interim / STAGE_MAP[stage]
        summary   = analyze_stage(stage_dir, stage)
        if summary:
            all_summaries[stage] = summary
            print_summary(summary)

    # ── simpan ke file jika --save ──
    if args.save and all_summaries:
        log_dir = root_dir / "outputs" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        label = args.stage if args.stage != "all" else "all_stages"
        out_path = log_dir / f"analysis_{label}.json"
        with open(out_path, "w") as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)
        logger.info(f"Hasil analisis disimpan ke: {out_path}")

    # ── ringkasan akhir jika all ──
    if args.stage == "all" and all_summaries:
        total_problems = sum(s["problem_files"] for s in all_summaries.values())
        print(f"\n{'='*60}")
        print(f"  TOTAL file bermasalah di semua stage: {total_problems}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()