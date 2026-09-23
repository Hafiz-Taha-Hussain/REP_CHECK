"""
Week 2 -- Angle math + rule engine (calibration pass)

Computes, per frame:
  - knee angle (hip-knee-ankle) for depth checking
  - knee alignment (knee x vs ankle x, normalized by hip width) for valgus
  - back angle (shoulder-hip line vs vertical) for excessive lean

Frames where the relevant joints have low visibility are skipped for that
metric (None), not guessed at -- this is the visibility gate.

Usage:
    python compute_angles.py output\\landmarks_1.json
"""

import argparse
import json
import math
from collections import deque

VISIBILITY_THRESHOLD = 0.5
MIN_HIP_WIDTH = 0.05  # normalized coords (0-1 range); below this, the ratio is unreliable


class LandmarkSmoother:
    """Rolling-average smoothing across the last `window` frames, per landmark
    coordinate. Reduces frame-to-frame jitter in raw MediaPipe output before
    any angle math runs on it -- ratio/difference metrics (knee_alignment,
    relative_lean) are especially sensitive to this jitter, since they
    amplify noise rather than average it out.

    Call .update(landmarks) each frame with the raw landmark list (or None
    if no detection that frame) and use the returned smoothed list instead
    of the raw one for all angle calculations.
    """

    def __init__(self, window=5, num_landmarks=33):
        self.window = window
        self.buffers = [
            {"x": deque(maxlen=window), "y": deque(maxlen=window),
             "z": deque(maxlen=window), "visibility": deque(maxlen=window)}
            for _ in range(num_landmarks)
        ]

    def update(self, landmarks):
        if landmarks is None:
            return None
        smoothed = []
        for i, lm in enumerate(landmarks):
            buf = self.buffers[i]
            buf["x"].append(lm["x"])
            buf["y"].append(lm["y"])
            buf["z"].append(lm["z"])
            buf["visibility"].append(lm["visibility"])
            smoothed.append({
                "x": sum(buf["x"]) / len(buf["x"]),
                "y": sum(buf["y"]) / len(buf["y"]),
                "z": sum(buf["z"]) / len(buf["z"]),
                # visibility isn't smoothed -- it's a per-frame confidence
                # signal, not a spatial coordinate; smoothing it would hide
                # genuine detection dropouts we want the gate to catch
                "visibility": lm["visibility"],
            })
        return smoothed

IDX = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}


def angle_at(a, b, c):
    """Angle at point b, formed by points a-b-c, in degrees."""
    ab = (a["x"] - b["x"], a["y"] - b["y"])
    cb = (c["x"] - b["x"], c["y"] - b["y"])
    dot = ab[0] * cb[0] + ab[1] * cb[1]
    mag_ab = math.hypot(*ab)
    mag_cb = math.hypot(*cb)
    if mag_ab == 0 or mag_cb == 0:
        return None
    cos_angle = max(-1.0, min(1.0, dot / (mag_ab * mag_cb)))
    return math.degrees(math.acos(cos_angle))


def visible(lm, name):
    return lm[IDX[name]]["visibility"] >= VISIBILITY_THRESHOLD


def knee_angle(lm, side):
    hip, knee, ankle = f"{side}_hip", f"{side}_knee", f"{side}_ankle"
    if not (visible(lm, hip) and visible(lm, knee) and visible(lm, ankle)):
        return None
    return angle_at(lm[IDX[hip]], lm[IDX[knee]], lm[IDX[ankle]])


def knee_alignment(lm, side):
    """Positive = knee tracking inward past ankle (valgus), relative to hip width."""
    hip_l, hip_r = IDX["left_hip"], IDX["right_hip"]
    knee, ankle = f"{side}_knee", f"{side}_ankle"
    if not (visible(lm, knee) and visible(lm, ankle)):
        return None
    hip_width = abs(lm[hip_l]["x"] - lm[hip_r]["x"])
    if hip_width < MIN_HIP_WIDTH:
        # Hips nearly stacked in x (body turned, or a jitter) -- ratio would be
        # meaningless here, not a real alignment reading. Skip rather than explode.
        return None
    knee_x, ankle_x = lm[IDX[knee]]["x"], lm[IDX[ankle]]["x"]
    # sign convention: for the left leg, knee_x < ankle_x means caving inward (to the right)
    if side == "left":
        deviation = ankle_x - knee_x
    else:
        deviation = knee_x - ankle_x
    return deviation / hip_width


def back_angle(lm, side):
    """Angle of the shoulder-hip line from vertical, in degrees. 0 = upright."""
    shoulder, hip = f"{side}_shoulder", f"{side}_hip"
    if not (visible(lm, shoulder) and visible(lm, hip)):
        return None
    dx = lm[IDX[shoulder]]["x"] - lm[IDX[hip]]["x"]
    dy = lm[IDX[shoulder]]["y"] - lm[IDX[hip]]["y"]
    # vertical reference vector is (0, -1) in image coords (up)
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def shin_angle(lm, side):
    """Angle of the knee-ankle line from vertical, in degrees. 0 = upright shin."""
    knee, ankle = f"{side}_knee", f"{side}_ankle"
    if not (visible(lm, knee) and visible(lm, ankle)):
        return None
    dx = lm[IDX[knee]]["x"] - lm[IDX[ankle]]["x"]
    dy = lm[IDX[knee]]["y"] - lm[IDX[ankle]]["y"]
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def relative_lean(lm, side):
    """Back angle minus shin angle -- how much MORE the torso leans than the
    shin does. This is what real squat coaching references (torso roughly
    tracking shin angle), not absolute vertical -- should be far less
    sensitive to individual build/mobility than back_angle alone."""
    b = back_angle(lm, side)
    s = shin_angle(lm, side)
    if b is None or s is None:
        return None
    return b - s


def analyze(path):
    with open(path) as f:
        data = json.load(f)

    rows = []
    for frame in data["frames"]:
        if frame["landmarks"] is None:
            continue
        lm = frame["landmarks"]
        rows.append({
            "frame": frame["frame"],
            "t": frame["timestamp_s"],
            "knee_angle_l": knee_angle(lm, "left"),
            "knee_angle_r": knee_angle(lm, "right"),
            "knee_align_l": knee_alignment(lm, "left"),
            "knee_align_r": knee_alignment(lm, "right"),
            "back_angle_l": back_angle(lm, "left"),
            "back_angle_r": back_angle(lm, "right"),
        })

    def summarize(key):
        vals = [r[key] for r in rows if r[key] is not None]
        if not vals:
            print(f"  {key:16s}: no valid frames (visibility gate excluded all)")
            return
        print(f"  {key:16s}: min={min(vals):6.1f}  max={max(vals):6.1f}  "
              f"avg={sum(vals)/len(vals):6.1f}  (n={len(vals)}/{len(rows)} frames)")

    print(f"File: {path}  ({len(rows)} frames with pose detected)\n")
    for key in ["knee_angle_l", "knee_angle_r", "knee_align_l", "knee_align_r",
                "back_angle_l", "back_angle_r"]:
        summarize(key)

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    args = parser.parse_args()
    analyze(args.json_path)