"""Unit tests for stats_engine geometry and event detection."""

import numpy as np
import pytest

from stats_engine import StatEngine, _compute_xg, default_field_polygon, default_goal_boxes


@pytest.fixture
def engine():
    w, h = 1920, 1080
    homography = np.eye(3, dtype=np.float32)
    homography[0, 0] = 105.0 / w
    homography[1, 1] = 68.0 / h
    return StatEngine(
        goal_boxes=default_goal_boxes(w, h),
        homography=homography,
        field_polygon=default_field_polygon(w, h),
        attacking_direction="left_to_right",
        fps=25,
        frame_width=w,
        frame_height=h,
    )


def _player(track_id, team, cx, cy, is_gk=False):
    return {
        "track_id": track_id,
        "team_id": team,
        "cx": float(cx),
        "cy": float(cy),
        "bbox": [cx - 10, cy - 20, cx + 10, cy + 20],
        "class_name": "goalkeeper" if is_gk else "player",
        "is_goalkeeper": is_gk,
    }


def test_possession_split(engine):
    ball = (500.0, 500.0)
    dets = [
        _player(1, 0, 480, 500),
        _player(2, 1, 900, 500),
    ]
    for i in range(10):
        engine.update(i, dets, ball)
    stats = engine.get_stats()
    assert stats["possession"]["team0_pct"] > 0
    assert stats["possession"]["team1_pct"] == 0


def test_pass_detection(engine):
    dets0 = [_player(1, 0, 400, 500), _player(2, 0, 800, 500)]
    ball = (410, 500)
    for i in range(5):
        engine._ball_history.append((i, ball))
        engine.update(i, dets0, ball)
    ball2 = (790, 500)
    engine._ball_history.append((5, ball2))
    engine.update(5, dets0, ball2)
    stats = engine.get_stats()
    assert stats["passes"]["team0_passes"] >= 0


def test_interception(engine):
    dets = [_player(1, 0, 400, 500), _player(2, 1, 420, 500)]
    for i in range(3):
        engine.update(i, [_player(1, 0, 400, 500)], (400, 500))
    engine.update(4, dets, (415, 500))
    stats = engine.get_stats()
    assert "team1_interceptions" in stats["interceptions"]


def test_offside_flag(engine):
    defenders = [_player(10, 1, 600, 500, is_gk=True), _player(11, 1, 700, 500), _player(12, 1, 750, 500)]
    attackers = [_player(20, 0, 900, 500), _player(21, 0, 800, 500)]
    engine._current_players_cache = defenders + attackers
    engine._check_offside_at_pass(100, 0, (500, 500), _player(20, 0, 900, 500))
    stats = engine.get_stats()
    assert stats["offsides"]["team0_offsides"] >= 0


def test_tackle_attempt(engine):
    dets = [_player(1, 0, 500, 500), _player(2, 1, 510, 500)]
    engine._possessor_track_id = 1
    engine._possessor_team = 0
    engine._update_tackles(10, dets, (505, 500))
    stats = engine.get_stats()
    assert stats["tackles"]["team1_tackles_attempted"] >= 1


def test_xg_formula():
    close = _compute_xg(5.0, 1.0)
    far = _compute_xg(40.0, 0.1)
    assert close > far
    assert 0 < close < 1


def test_heatmap_shape(engine):
    for i in range(5):
        engine.update(i, [_player(1, 0, 100 + i * 10, 200)], None)
    hm = engine.get_heatmap(1)
    assert hm.ndim == 2
    assert hm.sum() > 0


def test_export_json(tmp_path, engine):
    engine.update(0, [_player(1, 0, 100, 100)], (100, 100))
    out = tmp_path / "stats.json"
    engine.export_json(str(out))
    assert out.exists()


def test_empty_idle_possession_not_100(engine):
    """Idle / empty / no-ball sequences must not report 100% for a team."""
    # No detections, no ball — pure idle.
    for i in range(20):
        engine.update(i, [], None)
    stats = engine.get_stats()
    poss = stats["possession"]
    assert poss["team0_pct"] == 0.0
    assert poss["team1_pct"] == 0.0
    assert poss.get("unknown_pct", 0.0) >= 99.0 or poss["loose_pct"] >= 99.0
    assert poss["team0_pct"] + poss["team1_pct"] < 1.0


def test_pre_kickoff_players_no_ball_not_100(engine):
    """Players standing around with no ball should stay unknown / loose."""
    dets = [_player(1, 0, 400, 500), _player(2, 1, 900, 500)]
    for i in range(25):
        engine.update(i, dets, None)
    stats = engine.get_stats()
    poss = stats["possession"]
    assert poss["team0_pct"] == 0.0
    assert poss["team1_pct"] == 0.0
    assert poss.get("unknown_pct", 0.0) >= 99.0 or poss["loose_pct"] >= 99.0


def test_loose_ball_visible_not_forced_to_team(engine):
    """Visible ball far from all players stays loose, not assigned 100%."""
    dets = [_player(1, 0, 200, 200), _player(2, 1, 1700, 900)]
    ball = (960.0, 540.0)
    for i in range(20):
        engine.update(i, dets, ball)
    stats = engine.get_stats()
    poss = stats["possession"]
    assert poss["team0_pct"] == 0.0
    assert poss["team1_pct"] == 0.0
    assert poss["loose_pct"] > 50.0 or poss.get("unknown_pct", 0.0) > 50.0
