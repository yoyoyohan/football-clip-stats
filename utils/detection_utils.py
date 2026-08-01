import numpy as np

from utils.goal_regions import detection_from_raw


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def detection_from_sv(det, index: int, class_names: dict) -> dict | None:
    return detection_from_raw(det, index, class_names, require_track_id=True)


def extract_ball(det, class_names_inv: dict) -> tuple[float, float] | None:
    ball_id = class_names_inv.get("ball")
    if ball_id is None:
        return None
    best_conf = -1.0
    best_center = None
    for i, cid in enumerate(det.class_id):
        if int(cid) != ball_id:
            continue
        conf = float(det.confidence[i]) if det.confidence is not None else 0.0
        if conf > best_conf:
            x1, y1, x2, y2 = det.xyxy[i]
            best_center = (float((x1 + x2) / 2), float((y1 + y2) / 2))
            best_conf = conf
    return best_center
