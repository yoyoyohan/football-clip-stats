"""Run YOLO on a video and save an annotated clip.

YOLO goalpost boxes only appear after retraining with the fixed SoccerNet converter.
Calibrated goal regions (magenta boxes) come from calibrate_pitch.py and are used for stats.
"""

from pathlib import Path

import cv2
from ultralytics import YOLO

from utils.calibration import default_calibration_path, load_calibration

VIDEO_PATH = "input_videos/elclasico.mp4"
MODEL_PATH = "models/best.pt"
OUTPUT_PATH = "output_videos/yolo_annotated.mp4"


def draw_goal_regions(frame, goal_boxes: dict) -> None:
    for side, box in goal_boxes.items():
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.putText(
            frame,
            f"{side} goal",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2,
        )


def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {VIDEO_PATH}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    cal_path = default_calibration_path(VIDEO_PATH)
    cal = load_calibration(cal_path, w, h)
    goal_boxes = cal["goal_boxes"] if cal else None

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.predict(frame, conf=0.1, verbose=False)[0]
        annotated = result.plot()
        if goal_boxes:
            draw_goal_regions(annotated, goal_boxes)
        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Saved {frame_idx} frames to {OUTPUT_PATH}")
    if goal_boxes:
        print(f"Drawn calibrated goal regions from {cal_path}")
    else:
        print(
            f"No calibration at {cal_path}. "
            "Run: python calibrate_pitch.py --source input_videos/elclasico.mp4 --frame 50"
        )
    print("YOLO 'goalpost' labels need a retrained model; magenta boxes are calibration goal regions.")


if __name__ == "__main__":
    main()
