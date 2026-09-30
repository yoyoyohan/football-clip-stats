"""Run YOLO on a video and save an annotated clip.

Players are drawn as foot-point circles with detector confidence.
The ball always keeps a bounding box + confidence.
Goalposts stay as boxes. Calibrated magenta goal regions are optional.
"""

from pathlib import Path

import cv2
from ultralytics import YOLO

from analysis.overlay import annotate_frame
from utils.calibration import default_calibration_path, load_calibration

VIDEO_PATH = "input_videos/elclasico.mp4"
MODEL_PATH = "models/best.pt"
OUTPUT_PATH = "output_videos/yolo_annotated.mp4"


def detections_from_result(result) -> list[dict]:
    names = result.names
    boxes = result.boxes
    if boxes is None:
        return []
    detections = []
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy()
    conf = boxes.conf.cpu().numpy() if boxes.conf is not None else None
    for i in range(len(xyxy)):
        class_id = int(cls[i])
        bbox = [float(v) for v in xyxy[i]]
        detections.append(
            {
                "track_id": None,
                "class_id": class_id,
                "class_name": names.get(class_id, str(class_id)),
                "bbox": bbox,
                "cx": (bbox[0] + bbox[2]) / 2.0,
                "cy": (bbox[1] + bbox[3]) / 2.0,
                "confidence": float(conf[i]) if conf is not None else None,
                "team_id": None,
                "is_goalkeeper": names.get(class_id) == "goalkeeper",
            }
        )
    return detections


def ball_from_detections(detections: list[dict]) -> dict | None:
    balls = [d for d in detections if d.get("class_name") == "ball"]
    if not balls:
        return None
    best = max(balls, key=lambda d: d.get("confidence") or 0.0)
    return {
        "bbox": best["bbox"],
        "confidence": best.get("confidence"),
        "position": (best["cx"], best["cy"]),
    }


def draw_goal_regions(frame, goal_boxes: dict) -> None:
    for side, box in goal_boxes.items():
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(
            frame,
            f"{side} goal",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
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
        detections = detections_from_result(result)
        annotated = annotate_frame(
            frame,
            detections,
            ball_from_detections(detections),
            draw_hud_panel=False,
        )
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


if __name__ == "__main__":
    main()
