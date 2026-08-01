"""Filter YOLO / supervision detections to stats-relevant classes only."""

import numpy as np
import supervision as sv

# On-pitch objects — tracked with ByteTrack
TRACKABLE_CLASS_NAMES = frozenset({
    "ball",
    "player",
    "goalkeeper",
    "player_team1",
    "player_team_2",
    "referee",
})

# Pitch / broadcast overlay — kept for goal regions and future score-clock reads
OVERLAY_CLASS_NAMES = frozenset({
    "goalpost",
    "score",
    "gametime",
})

STATS_CLASS_NAMES = TRACKABLE_CLASS_NAMES | OVERLAY_CLASS_NAMES


def allowed_class_ids(class_names: dict) -> set[int]:
    return {
        cid for cid, name in class_names.items() if name in STATS_CLASS_NAMES
    }


def trackable_class_ids(class_names: dict) -> set[int]:
    return {
        cid for cid, name in class_names.items() if name in TRACKABLE_CLASS_NAMES
    }


def filter_supervision_detections(
    detections: sv.Detections, allowed_ids: set[int]
) -> sv.Detections:
    if len(detections) == 0 or not allowed_ids:
        return detections
    mask = np.isin(detections.class_id, list(allowed_ids))
    return detections[mask]


def split_trackable_overlay(
    detections: sv.Detections, class_names: dict
) -> tuple[sv.Detections, sv.Detections]:
    trackable_ids = trackable_class_ids(class_names)
    if len(detections) == 0:
        empty = detections
        return empty, empty
    mask = np.isin(detections.class_id, list(trackable_ids))
    return detections[mask], detections[~mask]


def filter_detection_dicts(detections: list[dict]) -> list[dict]:
    return [d for d in detections if d.get("class_name") in STATS_CLASS_NAMES]
