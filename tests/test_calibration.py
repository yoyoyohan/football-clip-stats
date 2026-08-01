import numpy as np

from utils.calibration import (
    LANDMARK_CATALOG,
    build_calibration_from_landmarks,
    build_calibration_from_one_goal,
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
