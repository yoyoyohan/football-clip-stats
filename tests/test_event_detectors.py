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


def test_shot_keeps_pending_across_interpolated_frames():
    """Interpolated frames must not wipe a toward-goal confirmation streak."""
    pitch = _pitch()
    det = ShotDetector(pitch=pitch, min_shot_speed_mps=12.0, confirm_frames=2, cooldown_frames=1)

    p0 = _m_to_px(70.0, 34.0)
    p1 = _m_to_px(74.0, 34.0)
    p2 = _m_to_px(78.0, 34.0)
    p3 = _m_to_px(82.0, 34.0)

    det.update(0, p0, 100.0, 0.0, 1, 0, ball_observed=True)
    det.update(1, p1, 100.0, p1[0] - p0[0], 1, 0, ball_observed=True)
    # Gap fill mid-flight — pending should survive.
    det.update(2, p2, 100.0, p2[0] - p1[0], 1, 0, ball_observed=False)
    event = det.update(3, p3, 100.0, p3[0] - p2[0], 1, 0, ball_observed=True)

    assert event is not None
    assert det.counts()["team0_shots"] == 1


def test_undercounted_style_pass_sequence():
    """Short clip with two clear teammate transfers should not undercount."""
    pitch = _pitch()
    det = PassDetector(
        fps=25.0,
        frame_width=1920,
        high_speed=10.0,
        min_velocity_peak=8.0,
        cooldown_frames=6,
        pitch=pitch,
    )

    p1 = _player(1, 0, 400, 500)
    p2 = _player(2, 0, 700, 500)
    p3 = _player(3, 0, 1000, 500)

    # Establish control with passer 1.
    for i in range(5):
        det.update(i, (400, 510), 3.0, [p1, p2, p3])

    # Pass 1 flight: ~25 px/frame (~36 m/s) toward player 2.
    event = None
    x = 400.0
    frame = 5
    while x < 700:
        x += 25.0
        frame += 1
        players = [p1, p2, p3] if x >= 680 else [p1, p2, p3]
        # Leave passer feet early so release triggers.
        near = [] if 430 < x < 680 else [p1, p2, p3]
        if x >= 680:
            near = [p1, p2, p3]
        event = det.update(frame, (x, 510), 20.0, near, ball_observed=True)
    if event is None:
        event = det.update(frame + 1, (700, 510), 3.0, [p1, p2, p3], ball_observed=True)
        frame += 1
    assert event is not None
    assert event.from_track_id == 1
    assert event.to_track_id == 2

    # Re-establish control with receiver 2, then pass to 3.
    for j in range(5):
        frame += 1
        det.update(frame, (700, 510), 3.0, [p1, p2, p3])

    event2 = None
    x = 700.0
    while x < 1000:
        x += 25.0
        frame += 1
        near = [p1, p2, p3] if x >= 980 else ([p2, p3] if x > 730 else [p1, p2, p3])
        if 730 < x < 980:
            near = [p1, p3]  # ball in flight away from feet
        event2 = det.update(frame, (x, 510), 20.0, near, ball_observed=True)
    if event2 is None:
        event2 = det.update(frame + 1, (1000, 510), 3.0, [p1, p2, p3], ball_observed=True)
    assert event2 is not None
    assert event2.from_track_id == 2
    assert event2.to_track_id == 3
    assert det.counts()["team0_passes"] == 2


def test_pass_rejects_teleport_jump():
    """Implausible ball teleports must not become passes."""
    pitch = _pitch()
    det = PassDetector(fps=25.0, frame_width=1920, pitch=pitch, cooldown_frames=2)
    passer = [_player(1, 0, 200, 500)]
    receiver = [_player(2, 0, 1700, 500)]

    for i in range(4):
        det.update(i, (200, 500), 2.0, passer)
    # Teleport across the frame in one tick.
    det.update(4, (1700, 500), 200.0, [], ball_observed=True)
    event = det.update(5, (1700, 500), 2.0, receiver, ball_observed=True)

    assert event is None
    assert det.counts()["team0_passes"] == 0


def test_shot_toward_goal_without_possessor_team():
    """Clear goal-bound trajectory with player context but no possession team."""
    pitch = _pitch()
    det = ShotDetector(
        pitch=pitch,
        min_shot_speed_mps=10.0,
        confirm_frames=2,
        cooldown_frames=1,
        fast_shot_speed_mps=18.0,
        fast_confirm_frames=1,
    )

    path = [
        _m_to_px(72.0, 34.0),
        _m_to_px(78.0, 34.0),
        _m_to_px(84.0, 34.0),
        _m_to_px(90.0, 34.0),
    ]
    players = [_player(7, 0, _m_to_px(70.0, 34.0)[0], _m_to_px(70.0, 34.0)[1])]

    event = None
    for i, pt in enumerate(path):
        vx = 0.0 if i == 0 else pt[0] - path[i - 1][0]
        event = det.update(
            i,
            pt,
            speed_mps=25.0,
            vx_sign=vx,
            possessor_track_id=None,
            possessor_team=None,
            ball_observed=True,
            players=players,
        )
        if event is not None:
            break

    assert event is not None
    assert event.team_id == 0
    assert det.counts()["team0_shots"] == 1


def test_pass_survives_wide_fov_meter_speed():
    """Ordinary ~30 px/frame flight must count even when default H maps it >42 m/s."""
    pitch = _pitch(fps=30.0)
    det = PassDetector(
        fps=30.0,
        frame_width=1920,
        high_speed=10.0,
        min_velocity_peak=8.0,
        cooldown_frames=6,
        pitch=pitch,
    )
    p1 = _player(1, 0, 400, 500)
    p2 = _player(2, 0, 700, 500)

    for i in range(5):
        det.update(i, (400, 510), 3.0, [p1, p2])

    event = None
    x = 400.0
    frame = 5
    while x < 700:
        x += 30.0
        frame += 1
        near = [p1, p2] if x >= 680 else []
        event = det.update(frame, (x, 510), 20.0, near, ball_observed=True)
    if event is None:
        event = det.update(frame + 1, (700, 510), 3.0, [p1, p2], ball_observed=True)

    assert event is not None
    assert event.from_track_id == 1
    assert event.to_track_id == 2
    # Sanity: this step rate is >42 m/s under default-style H @ 30fps.
    assert pitch.speed_mps((0.0, 0.0), (30.0, 0.0), frame_gap=1) > 42.0

def test_low_px_speed_pass_via_meter_peak():
    """Wide-FOV style: low px/frame but pass-like m/s still counts as a pass.

    Default identity-scale pitch maps ~3 px/frame to only ~4.9 m/s @ 30fps.
    Use a coarser meters-per-pixel scale so 3 px/frame ≈ 9 m/s while remaining
    well below the pixel min_velocity_peak (8).
    """
    import math

    class _FakePitch:
        def __init__(self, m_per_px: float, fps: float = 30.0):
            self.m_per_px = m_per_px
            self.fps = fps

        def to_meters(self, point_px):
            return (point_px[0] * self.m_per_px, point_px[1] * self.m_per_px)

        def distance_m(self, a_px, b_px):
            return math.hypot(a_px[0] - b_px[0], a_px[1] - b_px[1]) * self.m_per_px

        def speed_mps(self, a_px, b_px, frame_gap: int = 1):
            return self.distance_m(a_px, b_px) / (max(1, frame_gap) / self.fps)

        def in_pitch(self, x_m, y_m, margin: float = 2.0):
            return True

    # 3 px/frame * 0.1 m/px * 30 fps = 9 m/s (>= 5.5); norm px speed = 3 (< 8).
    pitch = _FakePitch(m_per_px=0.1, fps=30.0)
    det = PassDetector(
        fps=30.0,
        frame_width=1920,
        high_speed=10.0,
        min_velocity_peak=8.0,
        min_velocity_peak_mps=5.5,
        high_speed_mps=6.5,
        cooldown_frames=2,
        pitch=pitch,
    )
    p1 = _player(1, 0, 400, 500)
    p2 = _player(2, 0, 520, 500)  # ~120 px ≈ 12 m separation

    for i in range(5):
        det.update(i, (400, 510), 1.0, [p1, p2])

    event = None
    x = 400.0
    frame = 5
    # Creep at 3 px/frame — never clears pixel gates alone.
    while x < 520 and event is None:
        x += 3.0
        frame += 1
        near = [p1, p2] if x >= 505 else []
        event = det.update(frame, (x, 510), 3.0, near, ball_observed=True)
    settle = 0
    while event is None and settle < 5:
        frame += 1
        settle += 1
        event = det.update(frame, (min(x, 520.0), 510), 1.0, [p1, p2], ball_observed=True)

    assert event is not None, (
        f"expected meter-space pass; peak_px={det._peak_speed} peak_mps={det._peak_speed_mps}"
    )
    assert event.from_track_id == 1
    assert event.to_track_id == 2
    assert event.ball_speed_peak < det.min_velocity_peak
    assert det.counts()["team0_passes"] == 1


def test_stale_flight_aborts_without_receive():
    """Orphan release must not leave the detector stuck in_flight forever."""
    det = PassDetector(fps=30.0, frame_width=1920, max_flight_frames=60, cooldown_frames=2)
    passer = [_player(1, 0, 400, 500)]

    for i in range(5):
        det.update(i, (400, 510), 2.0, passer)
    # Force a release at high pixel speed with no receiver.
    det.update(5, (450, 510), 20.0, [], ball_observed=True)
    assert det._in_flight

    # Advance past the stale-flight timeout with no successful receive.
    for i in range(6, 6 + 70):
        det.update(i, (450 + (i - 5), 510), 2.0, [], ball_observed=True)

    assert not det._in_flight
    assert det.counts()["team0_passes"] == 0


def test_low_px_without_pitch_still_requires_pixel_peak():
    """Pixel-only mode (pitch=None) must not accept a slow 3 px/frame transfer."""
    det = PassDetector(
        fps=30.0,
        frame_width=1920,
        high_speed=10.0,
        min_velocity_peak=8.0,
        cooldown_frames=2,
        pitch=None,
    )
    p1 = _player(1, 0, 400, 500)
    p2 = _player(2, 0, 520, 500)

    for i in range(5):
        det.update(i, (400, 510), 1.0, [p1, p2])

    x = 400.0
    frame = 5
    event = None
    while x < 520:
        x += 3.0
        frame += 1
        near = [p1, p2] if x >= 505 else []
        event = det.update(frame, (x, 510), 3.0, near, ball_observed=True)
    if event is None:
        frame += 1
        event = det.update(frame, (520, 510), 1.0, [p1, p2], ball_observed=True)

    assert event is None
    assert det.counts()["team0_passes"] == 0

