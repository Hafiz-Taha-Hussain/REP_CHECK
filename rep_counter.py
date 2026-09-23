"""
Week 3 -- Live rep counter + per-rep pass/fail report

Processes a video frame-by-frame (pose extraction happens inline, one pass --
no separate landmarks.json step needed). Runs a hysteresis state machine on
knee angle to detect each rep as it completes, then scores that rep against
the thresholds calibrated in Week 2.

This is the core of the final pipeline: Week 4 will add annotated video
output on top of this same per-frame loop.

Usage:
    python rep_counter.py input.mp4 --report output\\rep_report.json
"""

import argparse
import json
import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from compute_angles import knee_angle, knee_alignment, back_angle, LandmarkSmoother, VISIBILITY_THRESHOLD
import thresholds as th

MODEL_PATH = "pose_landmarker_heavy.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
)

STANDING_ANGLE = 155
IN_SQUAT_ANGLE = 140
MIN_REP_FRAMES = 5
MIN_VALID_FRACTION = 0.5  # below this fraction of valid frames for a metric,
                          # report "insufficient data" rather than a verdict

# Skeleton connections for drawing (same 33-point layout used throughout).
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]

COLOR_PASS = (80, 200, 80)      # green, BGR
COLOR_FAIL = (60, 60, 230)      # red, BGR
COLOR_NEUTRAL = (220, 220, 220)  # light gray, BGR
COLOR_SKELETON = (0, 255, 0)
COLOR_JOINT = (0, 0, 255)


def draw_skeleton(frame, landmarks):
    h, w = frame.shape[:2]
    points = [(int(lm["x"] * w), int(lm["y"] * h)) for lm in landmarks]
    for a, b in POSE_CONNECTIONS:
        if a < len(points) and b < len(points):
            cv2.line(frame, points[a], points[b], COLOR_SKELETON, 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 3, COLOR_JOINT, -1)


def draw_hud(frame, rep_count, last_rep, view):
    """Top-left HUD: rep count, camera view, and the most recently
    completed rep's verdict + which checks (if any) failed."""
    x, y = 12, 30
    cv2.putText(frame, f"Reps: {rep_count}  ({view} view)", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_NEUTRAL, 2, cv2.LINE_AA)

    if last_rep is None:
        return

    y += 34
    verdict = last_rep["overall"]
    color = COLOR_PASS if verdict == "pass" else (
        COLOR_FAIL if verdict == "fail" else COLOR_NEUTRAL)
    cv2.putText(frame, f"Rep {last_rep['rep']}: {verdict.upper()}", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    failed = [name for name, c in last_rep["checks"].items() if c["verdict"] == "fail"]
    if failed:
        y += 28
        cv2.putText(frame, f"Issues: {', '.join(failed)}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_FAIL, 2, cv2.LINE_AA)


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading pose landmarker model (~5MB, one-time)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def primary_side(lm):
    def vis_sum(side):
        idxs = {"left": (23, 25, 27), "right": (24, 26, 28)}[side]
        return sum(lm[i]["visibility"] for i in idxs)
    return "left" if vis_sum("left") >= vis_sum("right") else "right"


def to_landmark_dicts(mp_result_landmarks):
    return [{"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
            for lm in mp_result_landmarks]


class RepCounter:
    """Incremental hysteresis state machine -- feed it one frame at a time."""

    def __init__(self, view="front"):
        if view not in th.VIEWS:
            raise ValueError(f"Unknown view '{view}' -- must be one of {list(th.VIEWS)}")
        self.view = view
        self.state = "standing"
        self.current_rep_frames = []  # list of (frame_idx, t, landmarks_dict)
        self.completed_reps = []      # list of finalized rep dicts
        self.rep_number = 0

    def process_frame(self, frame_idx, t, landmarks):
        if landmarks is None:
            return  # no detection this frame -- nothing to update

        side = primary_side(landmarks)
        angle = knee_angle(landmarks, side)
        if angle is None:
            return  # gated out by visibility -- skip for state purposes

        if self.state == "standing" and angle < IN_SQUAT_ANGLE:
            self.state = "in_squat"
            self.current_rep_frames = [(frame_idx, t, landmarks)]

        elif self.state == "in_squat":
            self.current_rep_frames.append((frame_idx, t, landmarks))
            if angle >= STANDING_ANGLE:
                if len(self.current_rep_frames) >= MIN_REP_FRAMES:
                    self.rep_number += 1
                    self.completed_reps.append(
                        self._score_rep(self.rep_number, self.current_rep_frames, self.view)
                    )
                self.state = "standing"
                self.current_rep_frames = []

    @staticmethod
    def _score_rep(rep_number, frames, view):
        view_cfg = th.VIEWS[view]
        start_frame, start_t, _ = frames[0]
        end_frame, end_t, _ = frames[-1]

        # Depth: min knee angle across whichever side is valid, per frame
        depth_vals = []
        for _, _, lm in frames:
            for side in ("left", "right"):
                a = knee_angle(lm, side)
                if a is not None:
                    depth_vals.append(a)
        depth_valid_frac = len(depth_vals) / (len(frames) * 2)

        # Back angle: max lean across whichever side is valid, per frame
        back_vals = []
        for _, _, lm in frames:
            for side in ("left", "right"):
                a = back_angle(lm, side)
                if a is not None:
                    back_vals.append(a)
        back_valid_frac = len(back_vals) / (len(frames) * 2)

        # Knee alignment: min/max across whichever side is valid, per frame
        align_vals = []
        for _, _, lm in frames:
            for side in ("left", "right"):
                a = knee_alignment(lm, side)
                if a is not None:
                    align_vals.append(a)
        align_valid_frac = len(align_vals) / (len(frames) * 2)

        result = {
            "rep": rep_number,
            "view": view,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_t": round(start_t, 2),
            "end_t": round(end_t, 2),
            "checks": {},
        }

        # --- Depth check (both views) ---
        if not view_cfg["depth_enabled"]:
            result["checks"]["depth"] = {"verdict": "not_scored"}
        elif depth_valid_frac < MIN_VALID_FRACTION:
            result["checks"]["depth"] = {"verdict": "insufficient_data"}
        else:
            min_angle = min(depth_vals)
            passed = min_angle < th.DEPTH_THRESHOLD_DEG
            result["checks"]["depth"] = {
                "verdict": "pass" if passed else "fail",
                "min_knee_angle": round(min_angle, 1),
                "threshold": th.DEPTH_THRESHOLD_DEG,
            }

        # --- Back angle check (side view only) ---
        if not view_cfg["back_angle_enabled"]:
            result["checks"]["back_angle"] = {"verdict": "not_scored"}
        elif back_valid_frac < MIN_VALID_FRACTION:
            result["checks"]["back_angle"] = {"verdict": "insufficient_data"}
        else:
            max_angle = max(back_vals)
            passed = max_angle <= th.BACK_ANGLE_MAX_DEG_SIDE
            result["checks"]["back_angle"] = {
                "verdict": "pass" if passed else "fail",
                "max_back_angle": round(max_angle, 1),
                "threshold": th.BACK_ANGLE_MAX_DEG_SIDE,
            }

        # --- Knee alignment check (front view only) ---
        if not view_cfg["knee_alignment_enabled"]:
            result["checks"]["knee_alignment"] = {"verdict": "not_scored"}
        elif align_valid_frac < MIN_VALID_FRACTION:
            result["checks"]["knee_alignment"] = {"verdict": "insufficient_data"}
        else:
            lo, hi = min(align_vals), max(align_vals)
            passed = lo >= th.KNEE_ALIGNMENT_MIN and hi <= th.KNEE_ALIGNMENT_MAX
            result["checks"]["knee_alignment"] = {
                "verdict": "pass" if passed else "fail",
                "min": round(lo, 2),
                "max": round(hi, 2),
                "range": [th.KNEE_ALIGNMENT_MIN, th.KNEE_ALIGNMENT_MAX],
            }

        verdicts = [c["verdict"] for c in result["checks"].values()]
        if "fail" in verdicts:
            result["overall"] = "fail"
        elif all(v in ("pass", "not_scored") for v in verdicts):
            result["overall"] = "pass"
        else:
            result["overall"] = "inconclusive"

        return result


def run(video_path, report_path=None, view="front", output_video_path=None):
    ensure_model()

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_video_path:
        os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_w, frame_h))
        if not writer.isOpened():
            raise IOError(f"Could not open video writer for: {output_video_path}")

    counter = RepCounter(view=view)
    smoother = LandmarkSmoother(window=5)
    frame_idx = 0

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((frame_idx / fps) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            landmarks = None
            if result.pose_landmarks:
                landmarks = to_landmark_dicts(result.pose_landmarks[0])
            landmarks = smoother.update(landmarks)

            counter.process_frame(frame_idx, frame_idx / fps, landmarks)

            if writer is not None:
                if landmarks is not None:
                    draw_skeleton(frame, landmarks)
                last_rep = counter.completed_reps[-1] if counter.completed_reps else None
                draw_hud(frame, len(counter.completed_reps), last_rep, view)
                writer.write(frame)

            frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved annotated video to {output_video_path}")

    report = {
        "source_video": video_path,
        "view": view,
        "total_reps": len(counter.completed_reps),
        "reps": counter.completed_reps,
    }

    passed = sum(1 for r in counter.completed_reps if r["overall"] == "pass")
    failed = sum(1 for r in counter.completed_reps if r["overall"] == "fail")
    inconclusive = sum(1 for r in counter.completed_reps if r["overall"] == "inconclusive")

    print(f"\n{report['total_reps']} rep(s) detected: "
          f"{passed} pass, {failed} fail, {inconclusive} inconclusive\n")
    for r in counter.completed_reps:
        line = f"Rep {r['rep']} ({r['start_t']}s-{r['end_t']}s): {r['overall'].upper()}"
        issues = [name for name, c in r["checks"].items() if c["verdict"] == "fail"]
        if issues:
            line += f"  -- failed: {', '.join(issues)}"
        print(line)

    if report_path:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved full report to {report_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("--report", help="Path to save JSON report", default=None)
    parser.add_argument("--view", choices=["front", "side"], default="front",
                         help="Camera angle the video was filmed from. "
                              "front: scores depth + knee alignment. "
                              "side: scores depth + back-angle/lean. "
                              "(default: front)")
    parser.add_argument("--output-video", help="Path to save annotated video "
                         "(skeleton overlay + live rep/verdict HUD)", default=None)
    args = parser.parse_args()
    run(args.video_path, args.report, view=args.view, output_video_path=args.output_video)