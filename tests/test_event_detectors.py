"""Tests for velocity-based pass and goal-line detection."""

from analysis.goal_detector import GoalDetector
from analysis.pass_detector import PassDetector


def _player(track_id, team, cx, cy):
    return {
        "track_id": track_id,
        "team_id": team,
        "cx": cx,
        "cy": cy,
        "bbox": [cx - 12, cy - 30, cx + 12, cy + 10],
        "class_name": "player",
    }


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
