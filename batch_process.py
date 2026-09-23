"""
Batch process a folder of squat videos (e.g. the Mendeley dataset's Squat
subset) into a single CSV of per-rep metrics -- one row per detected rep,
with a good/bad label inferred from the folder path.

Expects videos organized so "good" or "bad" appears somewhere in each
video's folder path (case-insensitive), e.g.:
    dataset/Squat/Good/subject01_squat.mp4
    dataset/Squat/Bad/subject01_squat.mp4
If your download is organized differently, tell me and I'll adjust the
labeling logic.

Usage:
    python batch_process.py dataset/Squat output/dataset_calibration.csv
"""

import argparse
import csv
import os

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from compute_angles import knee_angle, knee_alignment, back_angle, relative_lean, LandmarkSmoother

MODEL_PATH = "pose_landmarker_heavy.task"
STANDING_ANGLE = 155
IN_SQUAT_ANGLE = 140
MIN_REP_FRAMES = 5

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")


def infer_label(path):
    lower = path.lower()
    if "good" in lower:
        return "good"
    if "bad" in lower:
        return "bad"
    return "unknown"


def primary_side(lm):
    def vis_sum(side):
        idxs = {"left": (23, 25, 27), "right": (24, 26, 28)}[side]
        return sum(lm[i]["visibility"] for i in idxs)
    return "left" if vis_sum("left") >= vis_sum("right") else "right"


def process_video(video_path, options):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [skip] could not open {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_idx = 0
    state = "standing"
    current_rep = []
    reps = []
    smoother = LandmarkSmoother(window=5)

    # A fresh landmarker per video -- VIDEO mode requires strictly increasing
    # timestamps for the life of the landmarker instance, so reusing one
    # across multiple videos crashes on the 2nd video (its timestamps start
    # back at 0, which looks like time going backwards to MediaPipe).
    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((frame_idx / fps) * 1000)
            result = landmarker.detect_for_video(mp_image, ts_ms)

            lm = None
            if result.pose_landmarks:
                lm = [{"x": p.x, "y": p.y, "z": p.z, "visibility": p.visibility}
                      for p in result.pose_landmarks[0]]
            lm = smoother.update(lm)

            if lm is not None:
                side = primary_side(lm)
                angle = knee_angle(lm, side)

                if angle is not None:
                    if state == "standing" and angle < IN_SQUAT_ANGLE:
                        state = "in_squat"
                        current_rep = [lm]
                    elif state == "in_squat":
                        current_rep.append(lm)
                        if angle >= STANDING_ANGLE:
                            if len(current_rep) >= MIN_REP_FRAMES:
                                reps.append(current_rep)
                            state = "standing"
                            current_rep = []

            frame_idx += 1

    cap.release()
    return reps


def summarize_rep(rep_frames):
    depth_vals, back_vals, align_vals, rel_lean_vals = [], [], [], []
    for lm in rep_frames:
        for side in ("left", "right"):
            a = knee_angle(lm, side)
            if a is not None:
                depth_vals.append(a)
            b = back_angle(lm, side)
            if b is not None:
                back_vals.append(b)
            al = knee_alignment(lm, side)
            if al is not None:
                align_vals.append(al)
            rl = relative_lean(lm, side)
            if rl is not None:
                rel_lean_vals.append(rl)

    return {
        "min_knee_angle": min(depth_vals) if depth_vals else None,
        "max_back_angle": max(back_vals) if back_vals else None,
        "min_knee_align": min(align_vals) if align_vals else None,
        "max_knee_align": max(align_vals) if align_vals else None,
        "max_relative_lean": max(rel_lean_vals) if rel_lean_vals else None,
        "mean_relative_lean": (sum(rel_lean_vals) / len(rel_lean_vals)) if rel_lean_vals else None,
        "n_frames": len(rep_frames),
    }


def run(input_dir, output_csv):
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    rows = []
    video_paths = []
    for root, _, files in os.walk(input_dir):
        for fn in files:
            if fn.lower().endswith(VIDEO_EXTS):
                video_paths.append(os.path.join(root, fn))

    print(f"Found {len(video_paths)} video(s) under {input_dir}")

    for i, video_path in enumerate(video_paths, 1):
        label = infer_label(video_path)
        print(f"[{i}/{len(video_paths)}] {video_path} (label={label})")
        reps = process_video(video_path, options)
        for rep_i, rep_frames in enumerate(reps, 1):
            stats = summarize_rep(rep_frames)
            rows.append({
                "source": video_path,
                "label": label,
                "rep": rep_i,
                **stats,
            })

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "source", "label", "rep", "min_knee_angle", "max_back_angle",
            "min_knee_align", "max_knee_align", "max_relative_lean",
            "mean_relative_lean", "n_frames",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rep(s) across {len(video_paths)} video(s) to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Folder containing squat videos (searched recursively)")
    parser.add_argument("output_csv", help="Path to write the combined CSV")
    args = parser.parse_args()
    run(args.input_dir, args.output_csv)