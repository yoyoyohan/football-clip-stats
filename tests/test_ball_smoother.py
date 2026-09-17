"""Ball jump rejection and offline gap interpolation."""

import numpy as np

from analysis.ball_interpolator import BallInterpolator, BallObservation
from utils.detection_utils import normalize_ball


def test_normalize_ball_legacy_tuple():
    pos, conf, area = normalize_ball((10.0, 20.0))
    assert pos == (10.0, 20.0)
    assert conf == 1.0


def test_normalize_ball_dict():
    pos, conf, area = normalize_ball(
        {"position": (3.0, 4.0), "confidence": 0.7, "area": 40.0}
    )
    assert pos == (3.0, 4.0)
    assert conf == 0.7
    assert area == 40.0


def test_jump_rejection_online():
    bi = BallInterpolator(fps=30.0, frame_width=1920, frame_height=1080)
    obs0 = bi.update(0, (100.0, 100.0), confidence=0.9)
    assert obs0.observed and obs0.position == (100.0, 100.0)
    # Teleport across the frame — must be rejected.
    obs1 = bi.update(1, (1800.0, 900.0), confidence=0.95)
    assert obs1.observed is False
    assert obs1.position == (100.0, 100.0)  # held within gap


def test_gap_interpolation_offline():
    bi = BallInterpolator(fps=30.0, frame_width=1920, frame_height=1080, max_gap=10)
    detections = [
        (0, (100.0, 200.0), 0.9),
        (1, None, 0.0),
        (2, None, 0.0),
        (3, (130.0, 200.0), 0.8),
    ]
    out = bi.smooth_sequence(detections)
    assert out[0].observed is True
    assert out[0].position == (100.0, 200.0)
    assert out[1].observed is False
    assert out[1].position is not None
    assert abs(out[1].position[0] - 110.0) < 1e-6
    assert out[2].observed is False
    assert abs(out[2].position[0] - 120.0) < 1e-6
    assert out[3].observed is True


def test_offline_rejects_teleport_then_continues():
    bi = BallInterpolator(fps=30.0, frame_width=1920, frame_height=1080, max_gap=15)
    detections = [
        (0, (100.0, 100.0), 0.9),
        (1, (105.0, 100.0), 0.9),
        (2, (1700.0, 800.0), 0.99),  # teleport — reject
        (3, (110.0, 100.0), 0.85),
        (4, None, 0.0),
        (5, (120.0, 100.0), 0.8),
    ]
    out = bi.smooth_sequence(detections)
    assert out[0].observed and out[1].observed
    assert out[2].observed is False  # rejected raw detection
    # After reject, gap fill between accepted neighbors may place a point here.
    assert out[3].observed is True
    assert out[3].position[0] == 110.0
    assert out[4].observed is False
    assert out[4].position is not None
    assert abs(out[4].position[0] - 115.0) < 1e-6


def test_smooth_frame_records():
    bi = BallInterpolator(fps=25.0, frame_width=1280, frame_height=720)
    records = [
        {"frame_idx": 0, "ball": {"position": (50.0, 60.0), "confidence": 0.8, "area": 20}},
        {"frame_idx": 1, "ball": None},
        {"frame_idx": 2, "ball": (70.0, 60.0)},
    ]
    out = bi.smooth_frame_records(records)
    assert len(out) == 3
    assert isinstance(out[0], BallObservation)
    assert out[0].observed and out[2].observed
    assert out[1].observed is False
    assert out[1].position is not None


class _FakeDet:
    def __init__(self, xyxy, class_id, confidence):
        self.xyxy = np.asarray(xyxy, dtype=float)
        self.class_id = np.asarray(class_id, dtype=int)
        self.confidence = np.asarray(confidence, dtype=float)


def test_extract_ball_prefers_confidence_then_area():
    from utils.detection_utils import extract_ball

    det = _FakeDet(
        xyxy=[[0, 0, 10, 10], [0, 0, 40, 40], [100, 100, 110, 110]],
        class_id=[0, 0, 1],
        confidence=[0.4, 0.4, 0.9],
    )
    # Two balls at same conf — larger area wins; non-ball ignored.
    hit = extract_ball(det, {"ball": 0})
    assert hit is not None
    assert hit["confidence"] == 0.4
    assert hit["area"] == 1600.0
    assert hit["position"] == (20.0, 20.0)
