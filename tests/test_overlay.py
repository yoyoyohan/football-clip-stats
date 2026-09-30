from types import SimpleNamespace

import numpy as np

from analysis.overlay import (
    format_confidence,
    player_foot_point,
    player_ground_axes,
    player_label,
)
from utils.goal_regions import detection_from_raw


def test_player_foot_point_is_bottom_center():
    assert player_foot_point([10.0, 20.0, 50.0, 80.0]) == (30.0, 80.0)


def test_player_ground_oval_is_flatter_than_wide():
    rx, ry = player_ground_axes([0, 0, 40, 100])
    assert rx > ry
    assert ry / rx <= 0.45


def test_player_label_uses_detector_confidence():
    label = player_label({"track_id": 12, "confidence": 0.91})
    assert "12" in label
    assert "0.91" in label


def test_format_confidence_rejects_missing():
    assert format_confidence(None) is None
    assert format_confidence(0.74) == "0.74"


def test_detection_from_raw_keeps_confidence():
    det = SimpleNamespace(
        tracker_id=[7],
        xyxy=np.array([[10.0, 20.0, 40.0, 80.0]]),
        class_id=[1],
        confidence=[0.88],
    )
    out = detection_from_raw(det, 0, {1: "player"})
    assert out is not None
    assert out["confidence"] == 0.88
    assert out["track_id"] == 7
