"""
Week 1 -- Pose extraction + data understanding
(Current MediaPipe Tasks API -- mediapipe.solutions was removed in
recent mediapipe releases, so this uses mp.tasks.vision.PoseLandmarker)

Usage:
    python extract_landmarks.py input.mp4 output_landmarks.json --preview
"""

import argparse
import json
import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = "pose_landmarker_heavy.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
)

# Skeleton connections for drawing (replaces the old solutions.POSE_CONNECTIONS)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading pose landmarker model (~5MB, one-time)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"Saved model to {MODEL_PATH}")


def draw_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in POSE_CONNECTIONS:
        if a < len(points) and b < len(points):
            cv2.line(frame, points[a], points[b], (0, 255, 0), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)


def extract_landmarks(video_path, output_path, preview=False):
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
    frame_count = 0
    all_frames = []

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((frame_count / fps) * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            frame_data = {
                "frame": frame_count,
                "timestamp_s": round(frame_count / fps, 3),
                "landmarks": None,
            }

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]  # first detected person
                frame_data["landmarks"] = [
                    {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                    for lm in landmarks
                ]
                if preview:
                    draw_landmarks(frame, landmarks)

            all_frames.append(frame_data)
            frame_count += 1

            if preview:
                cv2.imshow("Pose extraction preview (press q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    if preview:
        cv2.destroyAllWindows()

    detected = sum(1 for f in all_frames if f["landmarks"] is not None)
    pct = (detected / frame_count * 100) if frame_count else 0
    print(f"Processed {frame_count} frames, pose detected in {detected} ({pct:.1f}%)")

    with open(output_path, "w") as f:
        json.dump(
            {
                "source_video": video_path,
                "fps": fps,
                "frame_count": frame_count,
                "landmark_names": LANDMARK_NAMES,
                "frames": all_frames,
            },
            f,
        )

    print(f"Saved landmarks to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract pose landmarks from a video")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("output_path", help="Path to save landmarks JSON")
    parser.add_argument(
        "--preview", action="store_true",
        help="Show a live preview window with skeleton overlay while processing",
    )
    args = parser.parse_args()

    extract_landmarks(args.video_path, args.output_path, preview=args.preview)