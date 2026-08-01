"""Load / save pitch calibration (homography, goal boxes, field polygon)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
GOAL_WIDTH_M = 7.32
GOAL_DEPTH_M = 3.0
GOAL_POST_Y0 = (PITCH_WIDTH_M - GOAL_WIDTH_M) / 2.0
GOAL_POST_Y1 = (PITCH_WIDTH_M + GOAL_WIDTH_M) / 2.0
PENALTY_DEPTH_M = 16.5
PENALTY_WIDTH_M = 40.32
PENALTY_Y0 = (PITCH_WIDTH_M - PENALTY_WIDTH_M) / 2.0
PENALTY_Y1 = (PITCH_WIDTH_M + PENALTY_WIDTH_M) / 2.0
CENTER_X = PITCH_LENGTH_M / 2.0
CENTER_Y = PITCH_WIDTH_M / 2.0

# Named pitch landmarks → (x_m, y_m). Pick any 4+ you can see in the broadcast frame.
LANDMARK_CATALOG: dict[str, tuple[float, float]] = {
    "left_goal_left_post": (0.0, GOAL_POST_Y0),
    "left_goal_right_post": (0.0, GOAL_POST_Y1),
    "right_goal_left_post": (PITCH_LENGTH_M, GOAL_POST_Y0),
    "right_goal_right_post": (PITCH_LENGTH_M, GOAL_POST_Y1),
    "center_spot": (CENTER_X, CENTER_Y),
    "left_penalty_spot": (11.0, CENTER_Y),
    "right_penalty_spot": (PITCH_LENGTH_M - 11.0, CENTER_Y),
    "left_near_corner": (0.0, PITCH_WIDTH_M),
    "right_near_corner": (PITCH_LENGTH_M, PITCH_WIDTH_M),
    "left_far_corner": (0.0, 0.0),
    "right_far_corner": (PITCH_LENGTH_M, 0.0),
    "left_penalty_box_front_near": (PENALTY_DEPTH_M, PENALTY_Y1),
    "left_penalty_box_front_far": (PENALTY_DEPTH_M, PENALTY_Y0),
    "right_penalty_box_front_near": (PITCH_LENGTH_M - PENALTY_DEPTH_M, PENALTY_Y1),
    "right_penalty_box_front_far": (PITCH_LENGTH_M - PENALTY_DEPTH_M, PENALTY_Y0),
}

# Plain-English help for each landmark (shown in UI + terminal).
# LEFT/RIGHT = left/right side of YOUR TV screen.  NEAR = bottom of screen (closer to camera).  FAR = top.
LANDMARK_HELP: dict[str, str] = {
    "left_goal_left_post": "LEFT goal (left side of screen) — click the ground at the LEFT post",
    "left_goal_right_post": "LEFT goal (left side of screen) — click the ground at the RIGHT post",
    "right_goal_left_post": "RIGHT goal (right side of screen) — click the ground at the LEFT post",
    "right_goal_right_post": "RIGHT goal (right side of screen) — click the ground at the RIGHT post",
    "center_spot": "CENTER CIRCLE — the kick-off dot in the middle of the pitch",
    "left_penalty_spot": "PENALTY SPOT in front of the LEFT goal (the white dot ~11m out)",
    "right_penalty_spot": "PENALTY SPOT in front of the RIGHT goal",
    "left_near_corner": "BOTTOM-LEFT corner flag — where left goal-line meets the BOTTOM touchline",
    "right_near_corner": "BOTTOM-RIGHT corner flag — right goal-line meets BOTTOM touchline",
    "left_far_corner": "TOP-LEFT corner flag — where left goal-line meets the TOP touchline (far from camera)",
    "right_far_corner": "TOP-RIGHT corner flag — right goal-line meets TOP touchline (far from camera)",
    "left_penalty_box_front_near": "LEFT penalty-box front line — meets BOTTOM sideline (18-yard line)",
    "left_penalty_box_front_far": "LEFT penalty-box front line — meets TOP sideline",
    "right_penalty_box_front_near": "RIGHT penalty-box front line — meets BOTTOM sideline",
    "right_penalty_box_front_far": "RIGHT penalty-box front line — meets TOP sideline",
}

LANDMARK_SHORTCUTS: list[tuple[str, str]] = [
    ("1", "left_goal_left_post"),
    ("2", "left_goal_right_post"),
    ("3", "right_goal_left_post"),
    ("4", "right_goal_right_post"),
    ("5", "center_spot"),
    ("6", "left_penalty_spot"),
    ("7", "right_penalty_spot"),
    ("8", "left_near_corner"),
    ("9", "right_near_corner"),
    ("0", "left_far_corner"),
    ("-", "right_far_corner"),
    ("q", "left_penalty_box_front_near"),
    ("w", "left_penalty_box_front_far"),
    ("e", "right_penalty_box_front_near"),
    ("r", "right_penalty_box_front_far"),
]


def meters_to_pixels(homography: np.ndarray, points_m: np.ndarray) -> np.ndarray:
    H_inv = np.linalg.inv(homography)
    pts = np.asarray(points_m, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)


def goal_box_from_homography(homography: np.ndarray, side: str) -> list[float]:
    y0 = (PITCH_WIDTH_M - GOAL_WIDTH_M) / 2.0
    y1 = (PITCH_WIDTH_M + GOAL_WIDTH_M) / 2.0
    if side == "team0":
        corners_m = np.float32([
            [0, y0],
            [GOAL_DEPTH_M, y0],
            [GOAL_DEPTH_M, y1],
            [0, y1],
        ])
    else:
        corners_m = np.float32([
            [PITCH_LENGTH_M, y0],
            [PITCH_LENGTH_M - GOAL_DEPTH_M, y0],
            [PITCH_LENGTH_M - GOAL_DEPTH_M, y1],
            [PITCH_LENGTH_M, y1],
        ])
    corners_px = meters_to_pixels(homography, corners_m)
    return [
        float(corners_px[:, 0].min()),
        float(corners_px[:, 1].min()),
        float(corners_px[:, 0].max()),
        float(corners_px[:, 1].max()),
    ]


def field_polygon_from_homography(homography: np.ndarray) -> list[list[float]]:
    corners_m = np.float32([
        [0, 0],
        [PITCH_LENGTH_M, 0],
        [PITCH_LENGTH_M, PITCH_WIDTH_M],
        [0, PITCH_WIDTH_M],
    ])
    corners_px = meters_to_pixels(homography, corners_m)
    return corners_px.tolist()


def _goal_post_meter_coords(goal_on_screen: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """goal_on_screen: which goal is visible ('left' or 'right' of TV)."""
    if goal_on_screen == "left":
        return LANDMARK_CATALOG["left_goal_left_post"], LANDMARK_CATALOG["left_goal_right_post"]
    return LANDMARK_CATALOG["right_goal_left_post"], LANDMARK_CATALOG["right_goal_right_post"]


def _depth_meter_point(goal_on_screen: str, depth_type: str) -> tuple[float, float]:
    if depth_type == "penalty_box_front_center":
        x = PENALTY_DEPTH_M if goal_on_screen == "left" else PITCH_LENGTH_M - PENALTY_DEPTH_M
        return (x, CENTER_Y)
    x = 11.0 if goal_on_screen == "left" else PITCH_LENGTH_M - 11.0
    return (x, CENTER_Y)


def _fourth_meter_point(goal_on_screen: str, post_right_m: tuple[float, float], depth_m: tuple[float, float]) -> tuple[float, float]:
    """Penalty-box front corner on the same side as the right post (non-collinear with goal line)."""
    if goal_on_screen == "left":
        return (PENALTY_DEPTH_M, post_right_m[1])
    return (PITCH_LENGTH_M - PENALTY_DEPTH_M, post_right_m[1])


def _estimate_depth_pixel(
    post_left_px: np.ndarray,
    post_right_px: np.ndarray,
    goal_on_screen: str,
    depth_type: str,
) -> tuple[float, float]:
    """Guess penalty-spot pixel from goal width when user cannot see a 3rd mark."""
    center = (post_left_px + post_right_px) / 2.0
    along_goal = post_right_px - post_left_px
    goal_len = float(np.linalg.norm(along_goal))
    if goal_len < 1.0:
        raise ValueError("Goal posts are too close together — re-click the two posts")

    # Perpendicular into the pitch (broadcast: usually upward on screen).
    perp = np.array([-along_goal[1], along_goal[0]], dtype=np.float64)
    if perp[1] > 0:
        perp = -perp

    depth_m = 11.0 if depth_type == "penalty_spot" else PENALTY_DEPTH_M
    scale = goal_len / GOAL_WIDTH_M
    depth_px = center + perp * (depth_m * scale)
    return (float(depth_px[0]), float(depth_px[1]))


def _fourth_pixel_from_geometry(
    post_left_px: np.ndarray,
    post_right_px: np.ndarray,
    depth_px: np.ndarray,
    post_left_m: tuple[float, float],
    post_right_m: tuple[float, float],
    depth_m: tuple[float, float],
    fourth_m: tuple[float, float],
) -> tuple[float, float]:
    """Extrapolate 4th pixel from local goal↔pitch geometry (3 clicks + 1 inferred)."""
    center_px = (post_left_px + post_right_px) / 2.0
    center_m = ((post_left_m[0] + post_right_m[0]) / 2.0, (post_left_m[1] + post_right_m[1]) / 2.0)
    g_px = post_right_px - post_left_px
    g_m = (post_right_m[0] - post_left_m[0], post_right_m[1] - post_left_m[1])
    d_px = depth_px - center_px
    d_m = (depth_m[0] - center_m[0], depth_m[1] - center_m[1])

    def _solve_2x2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        det = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
        if abs(det) < 1e-9:
            return b * 0.5
        inv = np.array([[a[1, 1], -a[0, 1]], [-a[1, 0], a[0, 0]]], dtype=np.float64) / det
        return inv @ b

    a = np.array([[g_m[0], d_m[0]], [g_m[1], d_m[1]]], dtype=np.float64)
    off_m = (fourth_m[0] - post_left_m[0], fourth_m[1] - post_left_m[1])
    coeff = _solve_2x2(a, np.array(off_m, dtype=np.float64))
    fourth_px = post_left_px + coeff[0] * g_px + coeff[1] * d_px
    return (float(fourth_px[0]), float(fourth_px[1]))


def build_calibration_from_one_goal(
    post_a_px: tuple[float, float],
    post_b_px: tuple[float, float],
    goal_on_screen: str,
    depth_px: tuple[float, float] | None = None,
    depth_type: str = "penalty_spot",
    depth_estimated: bool = False,
) -> dict:
    """
    Calibrate when only ONE goal is visible (typical tight broadcast).

    Minimum: 2 goal-post clicks. Optional 3rd click = penalty spot or 18-yard line.
    Without a 3rd click, depth is estimated from the 7.32 m goal width (less accurate).
    """
    if goal_on_screen not in ("left", "right"):
        raise ValueError("goal_on_screen must be 'left' or 'right'")

    pa = np.array(post_a_px, dtype=np.float64)
    pb = np.array(post_b_px, dtype=np.float64)
    if pa[0] <= pb[0]:
        post_left_px, post_right_px = pa, pb
    else:
        post_left_px, post_right_px = pb, pa

    post_left_m, post_right_m = _goal_post_meter_coords(goal_on_screen)
    depth_m = _depth_meter_point(goal_on_screen, depth_type)
    fourth_m = _fourth_meter_point(goal_on_screen, post_right_m, depth_m)

    if depth_px is None:
        depth_px = _estimate_depth_pixel(post_left_px, post_right_px, goal_on_screen, depth_type)
        depth_estimated = True

    depth_arr = np.array(depth_px, dtype=np.float64)
    fourth_px = _fourth_pixel_from_geometry(
        post_left_px, post_right_px, depth_arr,
        post_left_m, post_right_m, depth_m, fourth_m,
    )

    pixel_points = [
        (float(post_left_px[0]), float(post_left_px[1])),
        (float(post_right_px[0]), float(post_right_px[1])),
        depth_px,
        fourth_px,
    ]
    meter_points = [post_left_m, post_right_m, depth_m, fourth_m]

    src = np.float32(pixel_points)
    dst = np.float32(meter_points)
    homography, _ = cv2.findHomography(src, dst, 0)
    if homography is None:
        raise ValueError("Could not compute homography — check goal clicks")

    goal_boxes = {
        "team0": goal_box_from_homography(homography, "team0"),
        "team1": goal_box_from_homography(homography, "team1"),
    }
    return {
        "homography": homography.tolist(),
        "goal_boxes": goal_boxes,
        "field_polygon": field_polygon_from_homography(homography),
        "pitch_meters": {"length": PITCH_LENGTH_M, "width": PITCH_WIDTH_M},
        "method": "single_goal",
        "goal_on_screen": goal_on_screen,
        "depth_type": depth_type,
        "depth_estimated": depth_estimated,
        "landmark_count": 2 if depth_estimated else 3,
    }


def build_calibration_from_landmarks(
    pixel_points: list[tuple[float, float]],
    meter_points: list[tuple[float, float]],
) -> dict:
    """Build homography from >=4 matched point pairs (partial broadcast views OK)."""
    if len(pixel_points) < 4 or len(pixel_points) != len(meter_points):
        raise ValueError("Need at least 4 matched pixel/meter point pairs")

    src = np.float32(pixel_points)
    dst = np.float32(meter_points)
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if homography is None:
        raise ValueError("Could not compute homography — check landmark clicks")

    goal_boxes = {
        "team0": goal_box_from_homography(homography, "team0"),
        "team1": goal_box_from_homography(homography, "team1"),
    }
    field_polygon = field_polygon_from_homography(homography)
    return {
        "homography": homography.tolist(),
        "goal_boxes": goal_boxes,
        "field_polygon": field_polygon,
        "pitch_meters": {"length": PITCH_LENGTH_M, "width": PITCH_WIDTH_M},
        "landmark_count": len(pixel_points),
    }


def estimate_homography_from_goalposts(
    goalposts: list[dict],
    frame_width: int,
) -> np.ndarray | None:
    """Auto homography when goalposts are visible (no full pitch corners needed)."""
    if len(goalposts) < 2:
        return None

    mid = frame_width / 2.0
    left = sorted([g for g in goalposts if g["cx"] < mid], key=lambda g: g["bbox"][1])
    right = sorted([g for g in goalposts if g["cx"] >= mid], key=lambda g: g["bbox"][1])

    src_px: list[tuple[float, float]] = []
    dst_m: list[tuple[float, float]] = []

    def _add_posts(posts: list[dict], side: str) -> None:
        if len(posts) < 2:
            return
        bottom = posts[-2:]
        bottom = sorted(bottom, key=lambda g: g["bbox"][0])
        if side == "left":
            keys = ("left_goal_left_post", "left_goal_right_post")
        else:
            keys = ("right_goal_left_post", "right_goal_right_post")
        for post, key in zip(bottom, keys):
            x = (post["bbox"][0] + post["bbox"][2]) / 2.0
            y = post["bbox"][3]
            src_px.append((float(x), float(y)))
            dst_m.append(LANDMARK_CATALOG[key])

    _add_posts(left, "left")
    _add_posts(right, "right")

    if len(src_px) < 4:
        return None

    H, _ = cv2.findHomography(np.float32(src_px), np.float32(dst_m), cv2.RANSAC, 5.0)
    return H


def auto_calibration_from_goalposts(
    goalposts: list[dict],
    frame_width: int,
    frame_height: int,
) -> dict | None:
    H = estimate_homography_from_goalposts(goalposts, frame_width)
    if H is None:
        return None
    return {
        "homography": H.tolist(),
        "goal_boxes": {
            "team0": goal_box_from_homography(H, "team0"),
            "team1": goal_box_from_homography(H, "team1"),
        },
        "field_polygon": field_polygon_from_homography(H),
        "pitch_meters": {"length": PITCH_LENGTH_M, "width": PITCH_WIDTH_M},
        "method": "goalpost_auto",
        "frame_size": {"width": frame_width, "height": frame_height},
    }


def build_calibration(
    src_corners_px: list[tuple[float, float]],
) -> dict:
    """src_corners_px: bottom-left, bottom-right, top-right, top-left of visible pitch."""
    src = np.float32(src_corners_px)
    dst = np.float32([
        [0, PITCH_WIDTH_M],
        [PITCH_LENGTH_M, PITCH_WIDTH_M],
        [PITCH_LENGTH_M, 0],
        [0, 0],
    ])
    homography = cv2.getPerspectiveTransform(src, dst)
    goal_boxes = {
        "team0": goal_box_from_homography(homography, "team0"),
        "team1": goal_box_from_homography(homography, "team1"),
    }
    field_polygon = field_polygon_from_homography(homography)
    return {
        "homography": homography.tolist(),
        "goal_boxes": goal_boxes,
        "field_polygon": field_polygon,
        "pitch_meters": {"length": PITCH_LENGTH_M, "width": PITCH_WIDTH_M},
    }


def save_calibration(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_calibration(
    path: str | Path,
    frame_width: int,
    frame_height: int,
) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = {
        "homography": np.asarray(data["homography"], dtype=np.float32),
        "goal_boxes": data["goal_boxes"],
        "field_polygon": data["field_polygon"],
        "attacking_direction": data.get("attacking_direction", "left_to_right"),
    }
    # Only pass through fps if calibration stored it — never invent 30fps.
    if "fps" in data and data["fps"]:
        out["fps"] = float(data["fps"])
    return out


def default_calibration_path(video_path: str) -> Path:
    stem = Path(video_path).stem
    return Path("calibration") / f"{stem}.json"
