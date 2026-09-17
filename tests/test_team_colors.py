"""Dark-jersey feature extraction and stable cluster labeling."""

import numpy as np

from analysis.player_color_assignment import TeamColorAssigner


def _solid_frame(bgr, h=80, w=60):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = bgr
    return frame


def test_dark_jersey_features_not_empty():
    assigner = TeamColorAssigner()
    # Near-black jersey (B,G,R)
    frame = _solid_frame((15, 15, 15))
    bbox = [5, 5, 55, 75]
    feats = assigner._extract_jersey_features(frame, bbox)
    assert feats is not None
    assert feats.shape == (4,)
    # Low value / lightness for black.
    assert feats[0] < 60  # mean V
    assert feats[3] < 60  # mean L


def test_yellow_jersey_higher_chroma_than_black():
    assigner = TeamColorAssigner()
    black = assigner._extract_jersey_features(_solid_frame((10, 10, 10)), [5, 5, 55, 75])
    yellow = assigner._extract_jersey_features(_solid_frame((0, 220, 240)), [5, 5, 55, 75])
    assert black is not None and yellow is not None
    assert yellow[2] > black[2]  # chroma
    assert yellow[0] > black[0]  # V


def test_cluster_labels_bright_vs_dark_stable():
    assigner = TeamColorAssigner()
    # Simulate collected features: V, S, chroma, L
    # Tracks 1-3 yellow-ish, 4-6 black-ish (larger black cluster — old bug used size).
    yellow = np.array([180.0, 160.0, 200.0, 170.0], dtype=np.float32)
    black = np.array([25.0, 20.0, 15.0, 20.0], dtype=np.float32)
    for tid in (1, 2, 3):
        assigner._track_colors[tid] = [yellow.copy() for _ in range(4)]
    for tid in (4, 5, 6, 7, 8):
        assigner._track_colors[tid] = [black.copy() for _ in range(4)]

    assigner.fit_teams(min_samples_per_track=3)

    for tid in (1, 2, 3):
        assert assigner._track_teams[tid] == 0, "brighter kit must be team0"
    for tid in (4, 5, 6, 7, 8):
        assert assigner._track_teams[tid] == 1, "darker kit must be team1"


def test_referee_excluded_from_collect():
    assigner = TeamColorAssigner()
    frame = _solid_frame((0, 0, 255))  # red
    dets = [
        {"track_id": 99, "class_name": "referee", "bbox": [5, 5, 55, 75]},
        {"track_id": 1, "class_name": "player", "bbox": [5, 5, 55, 75]},
    ]
    assigner.collect(frame, dets)
    assert 99 not in assigner._track_colors
    assert 1 in assigner._track_colors


def test_sticky_assignment_after_fit():
    assigner = TeamColorAssigner()
    yellow = np.array([180.0, 160.0, 200.0, 170.0], dtype=np.float32)
    black = np.array([25.0, 20.0, 15.0, 20.0], dtype=np.float32)
    assigner._track_colors[1] = [yellow for _ in range(3)]
    assigner._track_colors[2] = [black for _ in range(3)]
    assigner.fit_teams(min_samples_per_track=3)
    dets = [
        {"track_id": 1, "class_name": "player", "bbox": [0, 0, 10, 10], "team_id": None},
        {"track_id": 2, "class_name": "player", "bbox": [0, 0, 10, 10], "team_id": None},
    ]
    out = assigner.assign_teams(_solid_frame((0, 0, 0)), dets)
    assert out[0]["team_id"] == 0
    assert out[1]["team_id"] == 1


def test_too_few_tracks_leaves_teams_unset():
    assigner = TeamColorAssigner()
    yellow = np.array([180.0, 160.0, 200.0, 170.0], dtype=np.float32)
    assigner._track_colors[1] = [yellow for _ in range(5)]
    assigner.fit_teams(min_samples_per_track=3)
    assert assigner._fitted is True
    assert 1 not in assigner._track_teams
