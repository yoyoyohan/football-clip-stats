"""Extract goal regions and scoreboard overlays from detections."""

from __future__ import annotations


def detection_from_raw(
    det,
    index: int,
    class_names: dict,
    *,
    require_track_id: bool = True,
) -> dict | None:
    if require_track_id and (det.tracker_id is None or det.tracker_id[index] is None):
        return None
    x1, y1, x2, y2 = det.xyxy[index]
    class_id = int(det.class_id[index])
    class_name = class_names.get(class_id, str(class_id))
    bbox = [float(x1), float(y1), float(x2), float(y2)]
    cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
    track_id = None
    if det.tracker_id is not None:
        tid = det.tracker_id[index]
        if tid is not None:
            track_id = int(tid)
    return {
        "track_id": track_id,
        "class_id": class_id,
        "class_name": class_name,
        "bbox": bbox,
        "cx": float(cx),
        "cy": float(cy),
        "team_id": None,
        "is_goalkeeper": class_name in {
            "goalkeeper",
            "goalkeeper_team1",
            "goalkeeper_team2",
        },
    }


def extract_overlay(detections: list[dict]) -> dict:
    goalposts = [d for d in detections if d.get("class_name") == "goalpost"]
    score = next((d for d in detections if d.get("class_name") == "score"), None)
    gametime = next((d for d in detections if d.get("class_name") == "gametime"), None)
    return {
        "goalposts": goalposts,
        "score_bbox": score["bbox"] if score else None,
        "gametime_bbox": gametime["bbox"] if gametime else None,
    }


def _merge_goal_cluster(posts: list[dict], padding: float = 40.0) -> list[float] | None:
    if not posts:
        return None
    x1 = min(p["bbox"][0] for p in posts) - padding
    y1 = min(p["bbox"][1] for p in posts) - padding
    x2 = max(p["bbox"][2] for p in posts) + padding
    y2 = max(p["bbox"][3] for p in posts) + padding
    return [float(x1), float(y1), float(x2), float(y2)]


def goal_boxes_from_goalposts(
    goalposts: list[dict], frame_width: int, padding: float = 40.0
) -> dict[str, list[float]] | None:
    """Build team goal boxes from detected goalposts (left = team0, right = team1)."""
    if not goalposts:
        return None
    mid = frame_width / 2.0
    left = [g for g in goalposts if g["cx"] < mid]
    right = [g for g in goalposts if g["cx"] >= mid]
    team0 = _merge_goal_cluster(left, padding)
    team1 = _merge_goal_cluster(right, padding)
    if team0 is None or team1 is None:
        return None
    return {"team0": team0, "team1": team1}


def refine_goal_boxes_from_frames(
    frame_records: list[dict], frame_width: int, min_samples: int = 5
) -> dict[str, list[float]] | None:
    """Average goal-box estimates from frames that have goalposts on both sides."""
    team0_boxes: list[list[float]] = []
    team1_boxes: list[list[float]] = []
    for record in frame_records:
        overlay = record.get("overlay", {})
        boxes = goal_boxes_from_goalposts(overlay.get("goalposts", []), frame_width)
        if boxes:
            team0_boxes.append(boxes["team0"])
            team1_boxes.append(boxes["team1"])
    if len(team0_boxes) < min_samples:
        return None

    def _avg_box(boxes: list[list[float]]) -> list[float]:
        return [float(sum(b[i] for b in boxes) / len(boxes)) for i in range(4)]

    return {"team0": _avg_box(team0_boxes), "team1": _avg_box(team1_boxes)}


def goal_mouth_lines_from_boxes(goal_boxes: dict[str, list[float]]) -> dict[str, dict]:
    """Goal mouth line x and vertical span from merged goalpost boxes.

    team0 = left goal (mouth at right edge of box); team1 = right goal (mouth at left edge).
    """
    lines = {}
    if "team0" in goal_boxes:
        x1, y1, x2, y2 = goal_boxes["team0"]
        lines["team0"] = {"mouth_x": float(x2), "y1": float(y1), "y2": float(y2)}
    if "team1" in goal_boxes:
        x1, y1, x2, y2 = goal_boxes["team1"]
        lines["team1"] = {"mouth_x": float(x1), "y1": float(y1), "y2": float(y2)}
    return lines


def goal_mouth_lines_from_frames(
    frame_records: list[dict], frame_width: int, min_samples: int = 3
) -> dict[str, dict] | None:
    boxes = refine_goal_boxes_from_frames(frame_records, frame_width, min_samples=min_samples)
    if not boxes:
        return None
    return goal_mouth_lines_from_boxes(boxes)
