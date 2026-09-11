"""Tests for velocity-based pass, shot, and goal-line detection."""

import numpy as np

from analysis.goal_detector import GoalDetector
from analysis.pass_detector import PassDetector
from analysis.pitch_coordinates import PitchCoordinateMapper
from analysis.shot_detector import ShotDetector


def _player(track_id, team, cx, cy):
    return {
        "track_id": track_id,
        "team_id": team,
        "cx": cx,
        "cy": cy,
        "bbox": [cx - 12, cy - 30, cx + 12, cy + 10],
        "class_name": "player",
    }


def _pitch(w=1920, h=1080, fps=25.0):
    homography = np.eye(3, dtype=np.float32)
    homography[0, 0] = 105.0 / w
    homography[1, 1] = 68.0 / h
    return PitchCoordinateMapper(homography, fps=fps)


def _m_to_px(x_m, y_m, w=1920, h=1080):
    return (x_m * w / 105.0, y_m * h / 68.0)


def test_velocity_pass_detected():
    det = PassDetector(high_speed=14.0, cooldown_frames=2)
    passer = [_player(1, 0, 400, 500)]
    receiver = [_player(2, 0, 800, 500)]

    event = None
    for i in range(6):
        event = det.update(i, (400, 500), 4.0, passer)
    event = det.update(7, (550, 500), 22.0, [])
    event = det.update(8, (700, 500), 20.0, [])
    event = det.update(9, (800, 500), 5.0, receiver)
    if event is None:
        event = det.update(10, (800, 500), 3.0, receiver)

    assert event is not None
    assert event.from_track_id == 1
    assert event.to_track_id == 2
    assert det.counts()["team0_passes"] == 1


def test_goal_crossing_right_goal():
    # team1 goal on right side of frame
    goal_boxes = {"team0": [0, 400, 80, 680], "team1": [1200, 400, 1280, 680]}
    goal_lines = {
        "team0": {"mouth_x": 80, "y1": 400, "y2": 680},
        "team1": {"mouth_x": 1200, "y1": 400, "y2": 680},
    }
    gd = GoalDetector(goal_boxes=goal_boxes, goal_lines=goal_lines, min_cross_speed=10.0)

    gd.update(0, (1100, 540), 5.0, 0)
    event = gd.update(1, (1250, 540), 20.0, 0)

    assert event is not None
    assert event.scoring_team == 0
    assert gd.counts()["team0_goals"] == 1


def test_soft_clear_goal_cross_with_goal_lines():
    """Unambiguous goal-line crosses should count even below the old 12px bar."""
    goal_boxes = {"team0": [0, 400, 80, 680], "team1": [1200, 400, 1280, 680]}
    goal_lines = {
        "team0": {"mouth_x": 80, "y1": 400, "y2": 680},
        "team1": {"mouth_x": 1200, "y1": 400, "y2": 680},
    }
    gd = GoalDetector(goal_boxes=goal_boxes, goal_lines=goal_lines)

    gd.update(0, (1180, 540), 2.0, 0)
    # Soft finish: clear mouth cross but only ~6 px/frame.
    event = gd.update(1, (1210, 540), 6.0, 0)

    assert event is not None
    assert event.scoring_team == 0
    assert gd.counts()["team0_goals"] == 1


def test_through_ball_not_counted_as_shot():
    """Fast attacking-third ball aimed wide / at a teammate is not a shot."""
    pitch = _pitch()
    det = ShotDetector(pitch=pitch, min_shot_speed_mps=12.0, confirm_frames=2, cooldown_frames=1)

    # Through-ball down the right channel (extrapolates outside goal mouth).
    path = [
        _m_to_px(70.0, 18.0),
        _m_to_px(74.0, 14.0),
        _m_to_px(78.0, 10.0),
        _m_to_px(82.0, 6.0),
    ]
    # ~4m / (1/25s) = 100 m/s — well above threshold; geometry should reject.
    event = None
    for i, pt in enumerate(path):
        vx = 0.0 if i == 0 else pt[0] - path[i - 1][0]
        event = det.update(
            i,
            pt,
            speed_mps=100.0,
            vx_sign=vx,
            possessor_track_id=1,
            possessor_team=0,
            ball_observed=True,
            players=[_player(1, 0, path[0][0], path[0][1]), _player(2, 0, path[-1][0], path[-1][1])],
        )

    assert event is None
    assert det.counts()["team0_shots"] == 0


def test_through_ball_to_teammate_not_shot():
    """Even when aimed near the mouth, a clear teammate destination is a through-ball."""
    pitch = _pitch()
    det = ShotDetector(pitch=pitch, min_shot_speed_mps=12.0, confirm_frames=2, cooldown_frames=1)

    path = [
        _m_to_px(70.0, 34.0),
        _m_to_px(74.0, 34.0),
        _m_to_px(78.0, 34.0),
        _m_to_px(82.0, 34.0),
    ]
    # Teammate ahead on the ball path (~12m ahead of early frames).
    receiver_px = _m_to_px(90.0, 34.0)
    players = [
        _player(1, 0, path[0][0], path[0][1]),
        _player(2, 0, receiver_px[0], receiver_px[1]),
    ]

    event = None
    for i, pt in enumerate(path):
        vx = 0.0 if i == 0 else pt[0] - path[i - 1][0]
        event = det.update(
            i,
            pt,
            speed_mps=100.0,
            vx_sign=vx,
            possessor_track_id=1,
            possessor_team=0,
            ball_observed=True,
            players=players,
        )

    assert event is None
    assert det.counts()["team0_shots"] == 0


def test_real_shot_still_counted():
    """Fast ball in the attacking third aimed at the goal mouth is a shot."""
    pitch = _pitch()
    det = ShotDetector(pitch=pitch, min_shot_speed_mps=12.0, confirm_frames=2, cooldown_frames=1)

    path = [
        _m_to_px(70.0, 34.0),
        _m_to_px(74.0, 34.0),
        _m_to_px(78.0, 34.0),
        _m_to_px(82.0, 34.0),
    ]
    # Shooter behind the ball; no teammate on the path toward goal.
    players = [_player(1, 0, _m_to_px(68.0, 34.0)[0], _m_to_px(68.0, 34.0)[1])]

    event = None
    for i, pt in enumerate(path):
        vx = 0.0 if i == 0 else pt[0] - path[i - 1][0]
        event = det.update(
            i,
            pt,
            speed_mps=100.0,
            vx_sign=vx,
            possessor_track_id=1,
            possessor_team=0,
            ball_observed=True,
            players=players,
        )
        if event is not None:
            break

    assert event is not None
    assert event.team_id == 0
    assert det.counts()["team0_shots"] == 1


def test_shot_not_confirmed_on_interpolated_frames():
    """Interpolated / unobserved ball frames must clear pending shot confirmation."""
    pitch = _pitch()
    det = ShotDetector(pitch=pitch, min_shot_speed_mps=12.0, confirm_frames=2, cooldown_frames=1)

    p0 = _m_to_px(70.0, 34.0)
    p1 = _m_to_px(74.0, 34.0)
    p2 = _m_to_px(78.0, 34.0)

    det.update(0, p0, 100.0, 0.0, 1, 0, ball_observed=True)
    det.update(1, p1, 100.0, p1[0] - p0[0], 1, 0, ball_observed=True)
    # Interpolated frame mid-confirmation should reset streak.
    det.update(2, p2, 100.0, p2[0] - p1[0], 1, 0, ball_observed=False)
    event = det.update(3, p2, 100.0, 1.0, 1, 0, ball_observed=True)

    assert event is None
    assert det.counts()["team0_shots"] == 0
