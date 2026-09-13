import numpy as np

from utils.calibration import (
    LANDMARK_CATALOG,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    auto_calibration_from_goalposts,
    build_calibration_from_landmarks,
    build_calibration_from_one_goal,
    is_auto_calibration,
    is_manual_calibration,
    robust_auto_calibration_from_frames,
    _pixels_to_meters,
)


def test_landmark_homography_from_four_points():
    # Synthetic: identity-ish mapping
    px = [(0, 720), (1280, 720), (1280, 0), (0, 0)]
    m = [
        LANDMARK_CATALOG["left_near_corner"],
        LANDMARK_CATALOG["right_near_corner"],
        LANDMARK_CATALOG["right_far_corner"],
        LANDMARK_CATALOG["left_far_corner"],
    ]
    cal = build_calibration_from_landmarks(px, m)
    H = np.asarray(cal["homography"], dtype=np.float32)
    assert H.shape == (3, 3)


def test_single_goal_calibration_with_depth():
    # Left goal posts + penalty spot (synthetic perspective)
    post_l = (400.0, 500.0)
    post_r = (520.0, 500.0)
    depth = (460.0, 420.0)
    cal = build_calibration_from_one_goal(post_l, post_r, "left", depth_px=depth)
    H = np.asarray(cal["homography"], dtype=np.float32)
    assert H.shape == (3, 3)
    assert cal["method"] == "single_goal"
    assert cal["depth_estimated"] is False


def test_single_goal_calibration_two_clicks_only():
    post_l = (300.0, 600.0)
    post_r = (450.0, 600.0)
    cal = build_calibration_from_one_goal(post_l, post_r, "right")
    assert cal["depth_estimated"] is True
    assert cal["landmark_count"] == 2


def _post(cx: float, y2: float, half_w: float = 8.0, height: float = 80.0) -> dict:
    return {
        "class_name": "goalpost",
        "bbox": [cx - half_w, y2 - height, cx + half_w, y2],
        "cx": cx,
        "cy": y2 - height / 2.0,
    }


def test_method_helpers():
    assert is_manual_calibration("landmarks")
    assert is_manual_calibration("single_goal")
    assert is_manual_calibration(None)
    assert is_auto_calibration("goalpost_auto")
    assert is_auto_calibration("goalpost_auto_multiframe")
    assert not is_auto_calibration("landmarks")


def test_robust_auto_cal_single_goal_left_multiframe():
    """Stable left-goal posts across frames → usable auto calibration."""
    w, h = 1280, 720
    # True left-goal feet with small jitter
    true_l, true_r = 180.0, 290.0
    y = 520.0
    records = []
    rng = np.random.default_rng(0)
    for i in range(12):
        jl = float(rng.normal(0, 1.5))
        jr = float(rng.normal(0, 1.5))
        posts = [_post(true_l + jl, y + float(rng.normal(0, 1.0))),
                 _post(true_r + jr, y + float(rng.normal(0, 1.0)))]
        records.append({"frame_idx": i, "overlay": {"goalposts": posts}, "detections": posts})

    cal = robust_auto_calibration_from_frames(records, w, h, min_samples_per_post=3)
    assert cal is not None
    assert cal["method"] == "goalpost_auto_multiframe"
    assert cal["goal_on_screen"] == "left"
    assert cal["quality"] >= 0.25
    H = np.asarray(cal["homography"], dtype=np.float64)
    # Post feet should map near the left goal line (x≈0) and goal-width y's.
    meters = _pixels_to_meters(H, np.float32([[true_l, y], [true_r, y]]))
    assert abs(meters[0, 0]) < 12.0
    assert abs(meters[1, 0]) < 12.0
    assert -8.0 <= meters[0, 1] <= PITCH_WIDTH_M + 8.0
    # A point into the pitch from the goal should move toward midfield (increasing x).
    into = _pixels_to_meters(H, np.float32([[(true_l + true_r) / 2.0, y - 80.0]]))[0]
    assert into[0] > 3.0


def test_robust_auto_cal_both_goals():
    w, h = 1280, 720
    # Four posts: left goal ~x=100/200, right goal ~x=1080/1180
    layout = [(100.0, 500.0), (200.0, 500.0), (1080.0, 500.0), (1180.0, 500.0)]
    records = []
    rng = np.random.default_rng(1)
    for i in range(10):
        posts = [
            _post(x + float(rng.normal(0, 1.0)), y + float(rng.normal(0, 1.0)))
            for x, y in layout
        ]
        records.append({"frame_idx": i, "overlay": {"goalposts": posts}})

    cal = robust_auto_calibration_from_frames(records, w, h, min_samples_per_post=3)
    assert cal is not None
    assert cal["method"] == "goalpost_auto_multiframe"
    assert cal.get("goal_on_screen") == "both"
    assert "post_feet_px" in cal


def test_robust_auto_cal_rejects_insufficient_posts():
    w, h = 1280, 720
    records = [
        {"frame_idx": 0, "overlay": {"goalposts": [_post(100, 500)]}},
        {"frame_idx": 1, "detections": [_post(110, 500)]},
    ]
    assert robust_auto_calibration_from_frames(records, w, h) is None


def test_robust_auto_cal_rejects_implausible_separation():
    w, h = 1280, 720
    # Posts almost on top of each other → should fail quality gate / separation.
    records = []
    for i in range(8):
        posts = [_post(400.0, 500.0), _post(405.0, 500.0)]
        records.append({"frame_idx": i, "overlay": {"goalposts": posts}})
    cal = robust_auto_calibration_from_frames(
        records, w, h, min_samples_per_post=3, min_post_separation_px=25.0
    )
    assert cal is None


def test_robust_auto_cal_reads_detections_fallback():
    """When overlay.goalposts is missing, use detections class_name=goalpost."""
    w, h = 1280, 720
    records = []
    for i in range(8):
        posts = [_post(900.0, 540.0), _post(1010.0, 540.0)]
        records.append({"frame_idx": i, "detections": posts, "overlay": {}})
    cal = robust_auto_calibration_from_frames(records, w, h, min_samples_per_post=3)
    assert cal is not None
    assert cal["goal_on_screen"] == "right"


def test_legacy_single_frame_auto_still_works():
    posts = [_post(150, 500), _post(260, 500), _post(1020, 500), _post(1130, 500)]
    cal = auto_calibration_from_goalposts(posts, 1280, 720)
    assert cal is not None
    assert cal["method"] == "goalpost_auto"
