from utils.goal_regions import detection_from_raw


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def detection_from_sv(det, index: int, class_names: dict) -> dict | None:
    return detection_from_raw(det, index, class_names, require_track_id=True)


def normalize_ball(ball) -> tuple[tuple[float, float] | None, float, float]:
    """Normalize cached / live ball payloads to (position, confidence, area).

    Accepts legacy ``(x, y)`` tuples and richer dicts from ``extract_ball``.
    """
    if ball is None:
        return None, 0.0, 0.0
    if isinstance(ball, dict):
        pos = ball.get("position")
        if pos is None and "x" in ball and "y" in ball:
            pos = (ball["x"], ball["y"])
        if pos is None:
            return None, 0.0, 0.0
        return (
            (float(pos[0]), float(pos[1])),
            float(ball.get("confidence", 1.0)),
            float(ball.get("area", 0.0)),
        )
    if isinstance(ball, (tuple, list)) and len(ball) >= 2:
        return (float(ball[0]), float(ball[1])), 1.0, 0.0
    return None, 0.0, 0.0


def extract_ball(det, class_names_inv: dict) -> dict | None:
    """Pick the best ball detection in a frame.

    Returns ``{"position", "confidence", "area", "bbox"}`` or ``None``.
    Ranking: highest confidence, then larger bbox area as a tie-break (helps
    reject tiny false positives when confidences are similar).
    """
    ball_id = class_names_inv.get("ball")
    if ball_id is None:
        return None

    best_conf = -1.0
    best_area = -1.0
    best: dict | None = None

    confidences = det.confidence
    for i, cid in enumerate(det.class_id):
        if int(cid) != ball_id:
            continue
        conf = float(confidences[i]) if confidences is not None else 0.0
        x1, y1, x2, y2 = (float(v) for v in det.xyxy[i])
        area = max(0.0, (x2 - x1) * (y2 - y1))
        if conf > best_conf or (conf == best_conf and area > best_area):
            best_conf = conf
            best_area = area
            best = {
                "position": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                "confidence": conf,
                "area": area,
                "bbox": [x1, y1, x2, y2],
            }
    return best
