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
    perp = perp / float(np.linalg.norm(perp))

    depth_m = 11.0 if depth_type == "penalty_spot" else PENALTY_DEPTH_M
    scale = goal_len / GOAL_WIDTH_M  # pixels per meter along the goal line
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
    """Single-frame auto calibration (legacy). Prefer robust_auto_calibration_from_frames."""
    if len(goalposts) < 2:
        return None

    feet = sorted((_post_foot(g) for g in goalposts), key=lambda p: p[0])
    mid = frame_width / 2.0
    left_feet = [f for f in feet if f[0] < mid]
    right_feet = [f for f in feet if f[0] >= mid]

    # Prefer dual-goal with depth augmentation when both sides have ≥2 posts.
    if len(left_feet) >= 2 and len(right_feet) >= 2:
        src_px = [left_feet[0], left_feet[-1], right_feet[0], right_feet[-1]]
        dst_m = [
            LANDMARK_CATALOG["left_goal_left_post"],
            LANDMARK_CATALOG["left_goal_right_post"],
            LANDMARK_CATALOG["right_goal_left_post"],
            LANDMARK_CATALOG["right_goal_right_post"],
        ]
        src_aug = list(src_px)
        dst_aug = list(dst_m)
        for side, a, b in (("left", src_px[0], src_px[1]), ("right", src_px[2], src_px[3])):
            try:
                depth_px = _estimate_depth_pixel(
                    np.array(a, dtype=np.float64),
                    np.array(b, dtype=np.float64),
                    side,
                    "penalty_spot",
                )
                src_aug.append(depth_px)
                dst_aug.append(_depth_meter_point(side, "penalty_spot"))
            except ValueError:
                pass
        H, _ = cv2.findHomography(np.float32(src_aug), np.float32(dst_aug), cv2.RANSAC, 5.0)
        if H is not None:
            cal = {
                "homography": H.tolist(),
                "goal_boxes": {
                    "team0": goal_box_from_homography(H, "team0"),
                    "team1": goal_box_from_homography(H, "team1"),
                },
                "field_polygon": field_polygon_from_homography(H),
                "pitch_meters": {"length": PITCH_LENGTH_M, "width": PITCH_WIDTH_M},
                "method": "goalpost_auto",
                "goal_on_screen": "both",
                "frame_size": {"width": frame_width, "height": frame_height},
                "quality": 0.45,
                "quality_notes": ["single_frame_four_post"],
                "post_feet_px": {
                    "left_goal_left": list(src_px[0]),
                    "left_goal_right": list(src_px[1]),
                    "right_goal_left": list(src_px[2]),
                    "right_goal_right": list(src_px[3]),
                },
                "depth_estimated": True,
            }
            return cal if _calibration_passes_sanity(cal, frame_width, frame_height) else None

    # Single goal (2+ posts on one side, or overall).
    post_a, post_b = feet[0], feet[-1]
    if abs(post_b[0] - post_a[0]) < max(20.0, frame_width * 0.015):
        return None
    mid_x = (post_a[0] + post_b[0]) / 2.0
    goal_on_screen = "left" if mid_x < frame_width / 2.0 else "right"
    try:
        cal = build_calibration_from_one_goal(post_a, post_b, goal_on_screen)
    except ValueError:
        return None
    cal["method"] = "goalpost_auto"
    cal["frame_size"] = {"width": frame_width, "height": frame_height}
    cal["quality"] = 0.35
    cal["quality_notes"] = ["single_frame_two_post_fallback"]
    cal["post_feet_px"] = {"post_left": list(post_a), "post_right": list(post_b)}
    return cal if _calibration_passes_sanity(cal, frame_width, frame_height) else None


def _post_foot(post: dict) -> tuple[float, float]:
    """Ground contact approximation: bottom-center of the goalpost bbox."""
    bbox = post["bbox"]
    return (float(bbox[0] + bbox[2]) / 2.0, float(bbox[3]))


def _extract_goalposts_from_record(record: dict) -> list[dict]:
    overlay = record.get("overlay") or {}
    posts = overlay.get("goalposts")
    if posts:
        return list(posts)
    detections = record.get("detections") or []
    return [d for d in detections if d.get("class_name") == "goalpost"]


def _median_xy(points: list[tuple[float, float]]) -> tuple[float, float]:
    xs = sorted(p[0] for p in points)
    ys = sorted(p[1] for p in points)
    n = len(points)
    mid = n // 2
    if n % 2:
        return (xs[mid], ys[mid])
    return ((xs[mid - 1] + xs[mid]) / 2.0, (ys[mid - 1] + ys[mid]) / 2.0)


def _cluster_feet_by_x(
    feet: list[tuple[float, float]],
    max_clusters: int = 4,
    gap_frac: float = 0.04,
    frame_width: int = 1280,
) -> list[list[tuple[float, float]]]:
    """Greedy 1-D gap clustering on foot x (stable posts across frames)."""
    if not feet:
        return []
    ordered = sorted(feet, key=lambda p: p[0])
    min_gap = max(18.0, frame_width * gap_frac)
    clusters: list[list[tuple[float, float]]] = [[ordered[0]]]
    for pt in ordered[1:]:
        if pt[0] - clusters[-1][-1][0] >= min_gap:
            clusters.append([pt])
        else:
            clusters[-1].append(pt)
    # Merge smallest gaps until <= max_clusters
    while len(clusters) > max_clusters:
        gaps = [
            (clusters[i + 1][0][0] - clusters[i][-1][0], i)
            for i in range(len(clusters) - 1)
        ]
        _, idx = min(gaps, key=lambda t: t[0])
        clusters[idx].extend(clusters[idx + 1])
        del clusters[idx + 1]
    return clusters


def _infer_goal_side(
    post_left: tuple[float, float],
    post_right: tuple[float, float],
    frame_width: int,
) -> str:
    mid_x = (post_left[0] + post_right[0]) / 2.0
    return "left" if mid_x < frame_width / 2.0 else "right"


def _pixels_to_meters(homography: np.ndarray, points_px: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_px, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, homography).reshape(-1, 2)


def _calibration_passes_sanity(
    cal: dict,
    frame_width: int,
    frame_height: int,
    *,
    margin_m: float = 25.0,
) -> bool:
    """Reject clearly degenerate or explosive homographies."""
    try:
        H = np.asarray(cal["homography"], dtype=np.float64)
        if H.shape != (3, 3) or not np.isfinite(H).all():
            return False
        det = float(np.linalg.det(H))
        if abs(det) < 1e-12:
            return False

        # Prefer checking known post feet when present.
        feet: list[tuple[float, float]] = []
        post_feet = cal.get("post_feet_px") or {}
        for key in (
            "post_left", "post_right",
            "left_goal_left", "left_goal_right",
            "right_goal_left", "right_goal_right",
        ):
            if key in post_feet:
                feet.append(tuple(post_feet[key]))
        if len(feet) >= 2:
            meters = _pixels_to_meters(H, np.float32(feet))
            if not np.isfinite(meters).all():
                return False
            # Posts should land near a goal line (x≈0 or x≈105) and within pitch y.
            near_left = np.abs(meters[:, 0] - 0.0) < margin_m
            near_right = np.abs(meters[:, 0] - PITCH_LENGTH_M) < margin_m
            y_ok = (meters[:, 1] >= -margin_m) & (meters[:, 1] <= PITCH_WIDTH_M + margin_m)
            if not bool(np.all(y_ok)):
                return False
            if not bool(np.all(near_left | near_right)):
                return False
            # Separation in meters should resemble a real goal width (or two goals).
            # For a single pair, y-span should be near GOAL_WIDTH_M.
            if len(feet) == 2:
                y_span = float(abs(meters[0, 1] - meters[1, 1]))
                if y_span < GOAL_WIDTH_M * 0.35 or y_span > GOAL_WIDTH_M * 2.5:
                    return False
            return True

        # Fallback: lower-third samples should not all explode outside the pitch.
        samples = np.float32([
            [frame_width * 0.30, frame_height * 0.70],
            [frame_width * 0.50, frame_height * 0.72],
            [frame_width * 0.70, frame_height * 0.70],
        ])
        meters = _pixels_to_meters(H, samples)
        if not np.isfinite(meters).all():
            return False
        inside = (
            (meters[:, 0] >= -margin_m * 2)
            & (meters[:, 0] <= PITCH_LENGTH_M + margin_m * 2)
            & (meters[:, 1] >= -margin_m * 2)
            & (meters[:, 1] <= PITCH_WIDTH_M + margin_m * 2)
        )
        if int(inside.sum()) < 1:
            return False
        span_x = float(meters[:, 0].max() - meters[:, 0].min())
        span_y = float(meters[:, 1].max() - meters[:, 1].min())
        if span_x + span_y < 3.0:
            return False
        if span_x > PITCH_LENGTH_M * 3.0 or span_y > PITCH_WIDTH_M * 3.0:
            return False
        return True
    except Exception:
        return False


def _reprojection_score(
    H: np.ndarray,
    pixel_points: list[tuple[float, float]],
    meter_points: list[tuple[float, float]],
) -> float:
    """1.0 = perfect; decays with mean pixel reprojection error."""
    if len(pixel_points) < 2:
        return 0.0
    pred = meters_to_pixels(H, np.float32(meter_points))
    err = np.linalg.norm(pred - np.float32(pixel_points), axis=1)
    mean_err = float(np.mean(err))
    return float(max(0.0, 1.0 - mean_err / 40.0))


def _score_post_separation(
    sep_px: float,
    frame_width: int,
    min_sep: float,
    max_sep: float,
) -> float:
    if sep_px < min_sep or sep_px > max_sep:
        return 0.0
    # Prefer separations that look like a real goal width in broadcast (~3–12% of width).
    ideal = frame_width * 0.07
    return float(max(0.0, 1.0 - abs(sep_px - ideal) / (ideal * 2.5)))


MANUAL_CALIBRATION_METHODS = frozenset({
    "single_goal",
    "landmarks",
    "corners",
    "manual",
})
AUTO_CALIBRATION_METHODS = frozenset({
    "goalpost_auto",
    "goalpost_auto_multiframe",
})


def is_manual_calibration(method: str | None) -> bool:
    if not method:
        # Legacy files without method are treated as user-provided.
        return True
    return method in MANUAL_CALIBRATION_METHODS


def is_auto_calibration(method: str | None) -> bool:
    return bool(method) and (
        method in AUTO_CALIBRATION_METHODS or str(method).startswith("goalpost_auto")
    )


def robust_auto_calibration_from_frames(
    frame_records: list[dict],
    frame_width: int,
    frame_height: int,
    *,
    min_samples_per_post: int = 3,
    min_post_separation_px: float | None = None,
    max_post_separation_px: float | None = None,
) -> dict | None:
    """
    Multi-frame automatic pitch calibration from YOLO goalpost detections.

    Aggregates stable left/right post feet across frames (median), infers goal
    side, builds a homography, and quality-gates the result. Returns None if
    detections are insufficient or the homography fails sanity checks.
    """
    if not frame_records:
        return None

    min_sep = min_post_separation_px if min_post_separation_px is not None else max(20.0, frame_width * 0.015)
    max_sep = max_post_separation_px if max_post_separation_px is not None else frame_width * 0.45

    # Per-frame post feet (sorted left→right).
    frame_feet: list[list[tuple[float, float]]] = []
    all_feet: list[tuple[float, float]] = []
    for record in frame_records:
        posts = _extract_goalposts_from_record(record)
        if len(posts) < 2:
            continue
        feet = sorted((_post_foot(p) for p in posts), key=lambda p: p[0])
        # Deduplicate near-identical detections in one frame.
        deduped: list[tuple[float, float]] = [feet[0]]
        for ft in feet[1:]:
            if abs(ft[0] - deduped[-1][0]) >= min_sep * 0.5:
                deduped.append(ft)
        if len(deduped) < 2:
            continue
        frame_feet.append(deduped)
        all_feet.extend(deduped)

    if len(frame_feet) < min_samples_per_post and len(all_feet) < min_samples_per_post * 2:
        # Last-ditch: try any single frame with the legacy path (still enforce separation).
        for record in frame_records:
            posts = _extract_goalposts_from_record(record)
            if len(posts) < 2:
                continue
            feet = sorted((_post_foot(p) for p in posts), key=lambda p: p[0])
            if abs(feet[-1][0] - feet[0][0]) < min_sep:
                continue
            cal = auto_calibration_from_goalposts(posts, frame_width, frame_height)
            if cal is not None:
                cal["method"] = "goalpost_auto"
                cal["quality"] = min(float(cal.get("quality", 0.3)), 0.4)
                notes = list(cal.get("quality_notes") or [])
                notes.append("sparse_frames_legacy_fallback")
                cal["quality_notes"] = notes
                return cal
        return None

    clusters = _cluster_feet_by_x(all_feet, max_clusters=4, frame_width=frame_width)
    # Keep clusters with enough support.
    strong = [c for c in clusters if len(c) >= min_samples_per_post]
    notes: list[str] = []
    cal: dict | None = None
    quality = 0.0

    if len(strong) >= 4:
        # Two goals visible — use outermost pair on each side of mid-frame.
        medians = [_median_xy(c) for c in strong]
        left_side = sorted([m for m in medians if m[0] < frame_width / 2.0], key=lambda p: p[0])
        right_side = sorted([m for m in medians if m[0] >= frame_width / 2.0], key=lambda p: p[0])
        if len(left_side) >= 2 and len(right_side) >= 2:
            src_px = [left_side[0], left_side[-1], right_side[0], right_side[-1]]
            dst_m = [
                LANDMARK_CATALOG["left_goal_left_post"],
                LANDMARK_CATALOG["left_goal_right_post"],
                LANDMARK_CATALOG["right_goal_left_post"],
                LANDMARK_CATALOG["right_goal_right_post"],
            ]
            sep_l = abs(src_px[1][0] - src_px[0][0])
            sep_r = abs(src_px[3][0] - src_px[2][0])
            if sep_l < min_sep or sep_r < min_sep or sep_l > max_sep or sep_r > max_sep:
                notes.append("both_goals_implausible_separation")
            else:
                # Augment each goal with an estimated depth point so H is not
                # under-constrained (goal-line-only correspondences are weak).
                src_aug = list(src_px)
                dst_aug = list(dst_m)
                for side, a, b in (
                    ("left", src_px[0], src_px[1]),
                    ("right", src_px[2], src_px[3]),
                ):
                    try:
                        depth_px = _estimate_depth_pixel(
                            np.array(a, dtype=np.float64),
                            np.array(b, dtype=np.float64),
                            side,
                            "penalty_spot",
                        )
                    except ValueError:
                        continue
                    src_aug.append(depth_px)
                    dst_aug.append(_depth_meter_point(side, "penalty_spot"))
                H, _ = cv2.findHomography(
                    np.float32(src_aug), np.float32(dst_aug), cv2.RANSAC, 5.0
                )
                if H is not None:
                    repro = _reprojection_score(H, src_px, dst_m)
                    sep_score = 0.5 * (
                        _score_post_separation(sep_l, frame_width, min_sep, max_sep)
                        + _score_post_separation(sep_r, frame_width, min_sep, max_sep)
                    )
                    sample_score = min(1.0, len(frame_feet) / 20.0)
                    quality = 0.45 * repro + 0.30 * sep_score + 0.25 * sample_score
                    cal = {
                        "homography": H.tolist(),
                        "goal_boxes": {
                            "team0": goal_box_from_homography(H, "team0"),
                            "team1": goal_box_from_homography(H, "team1"),
                        },
                        "field_polygon": field_polygon_from_homography(H),
                        "pitch_meters": {"length": PITCH_LENGTH_M, "width": PITCH_WIDTH_M},
                        "method": "goalpost_auto_multiframe",
                        "goal_on_screen": "both",
                        "frame_size": {"width": frame_width, "height": frame_height},
                        "quality": float(quality),
                        "post_feet_px": {
                            "left_goal_left": list(src_px[0]),
                            "left_goal_right": list(src_px[1]),
                            "right_goal_left": list(src_px[2]),
                            "right_goal_right": list(src_px[3]),
                        },
                        "samples_frames": len(frame_feet),
                        "depth_estimated": True,
                    }
                    notes.append("both_goals_aggregated")

    if cal is None and len(strong) >= 2:
        # Single goal: take the two strongest neighboring clusters.
        # Prefer the pair with the most combined samples and plausible separation.
        candidates: list[tuple[float, tuple[float, float], tuple[float, float], int, int]] = []
        med_with_n = [(_median_xy(c), len(c)) for c in strong]
        for i in range(len(med_with_n) - 1):
            for j in range(i + 1, len(med_with_n)):
                a, na = med_with_n[i]
                b, nb = med_with_n[j]
                if a[0] > b[0]:
                    a, b = b, a
                    na, nb = nb, na
                sep = b[0] - a[0]
                if sep < min_sep or sep > max_sep:
                    continue
                score = na + nb + _score_post_separation(sep, frame_width, min_sep, max_sep) * 10.0
                candidates.append((score, a, b, na, nb))
        if candidates:
            candidates.sort(key=lambda t: t[0], reverse=True)
            _, post_l, post_r, n_l, n_r = candidates[0]
            goal_on_screen = _infer_goal_side(post_l, post_r, frame_width)
            try:
                built = build_calibration_from_one_goal(post_l, post_r, goal_on_screen)
            except ValueError:
                built = None
            if built is not None:
                H = np.asarray(built["homography"], dtype=np.float64)
                meter_posts = list(_goal_post_meter_coords(goal_on_screen))
                repro = _reprojection_score(H, [post_l, post_r], meter_posts)
                sep = abs(post_r[0] - post_l[0])
                sep_score = _score_post_separation(sep, frame_width, min_sep, max_sep)
                sample_score = min(1.0, (n_l + n_r) / 30.0)
                quality = 0.40 * repro + 0.35 * sep_score + 0.25 * sample_score
                # Depth was estimated — slightly lower confidence.
                if built.get("depth_estimated"):
                    quality *= 0.9
                    notes.append("depth_estimated_from_goal_width")
                built["method"] = "goalpost_auto_multiframe"
                built["frame_size"] = {"width": frame_width, "height": frame_height}
                built["quality"] = float(quality)
                built["post_feet_px"] = {
                    "post_left": list(post_l),
                    "post_right": list(post_r),
                }
                built["samples_per_post"] = {"left": int(n_l), "right": int(n_r)}
                built["samples_frames"] = len(frame_feet)
                notes.append(f"single_goal_{goal_on_screen}")
                cal = built

    if cal is None:
        # Pair-wise median from frames that show exactly two posts (very common TV shot).
        left_feet: list[tuple[float, float]] = []
        right_feet: list[tuple[float, float]] = []
        for feet in frame_feet:
            if len(feet) == 2:
                left_feet.append(feet[0])
                right_feet.append(feet[1])
            elif len(feet) > 2:
                # Use outermost two if they are close enough to be one goal.
                if feet[-1][0] - feet[0][0] <= max_sep:
                    left_feet.append(feet[0])
                    right_feet.append(feet[-1])
        if len(left_feet) >= min_samples_per_post and len(right_feet) >= min_samples_per_post:
            post_l = _median_xy(left_feet)
            post_r = _median_xy(right_feet)
            sep = abs(post_r[0] - post_l[0])
            if min_sep <= sep <= max_sep:
                goal_on_screen = _infer_goal_side(post_l, post_r, frame_width)
                try:
                    built = build_calibration_from_one_goal(post_l, post_r, goal_on_screen)
                except ValueError:
                    built = None
                if built is not None:
                    H = np.asarray(built["homography"], dtype=np.float64)
                    meter_posts = list(_goal_post_meter_coords(goal_on_screen))
                    repro = _reprojection_score(H, [post_l, post_r], meter_posts)
                    sep_score = _score_post_separation(sep, frame_width, min_sep, max_sep)
                    sample_score = min(1.0, (len(left_feet) + len(right_feet)) / 30.0)
                    quality = 0.40 * repro + 0.35 * sep_score + 0.25 * sample_score
                    if built.get("depth_estimated"):
                        quality *= 0.9
                        notes.append("depth_estimated_from_goal_width")
                    built["method"] = "goalpost_auto_multiframe"
                    built["frame_size"] = {"width": frame_width, "height": frame_height}
                    built["quality"] = float(quality)
                    built["post_feet_px"] = {
                        "post_left": list(post_l),
                        "post_right": list(post_r),
                    }
                    built["samples_per_post"] = {
                        "left": len(left_feet),
                        "right": len(right_feet),
                    }
                    built["samples_frames"] = len(frame_feet)
                    notes.append(f"paired_frames_single_goal_{goal_on_screen}")
                    cal = built

    if cal is None:
        return None

    cal["quality_notes"] = notes
    if float(cal.get("quality", 0.0)) < 0.25:
        return None
    if not _calibration_passes_sanity(cal, frame_width, frame_height):
        return None
    return cal


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
        "method": data.get("method"),
        "quality": data.get("quality"),
        "raw": data,
    }
    # Only pass through fps if calibration stored it — never invent 30fps.
    if "fps" in data and data["fps"]:
        out["fps"] = float(data["fps"])
    return out


def default_calibration_path(video_path: str) -> Path:
    stem = Path(video_path).stem
    return Path("calibration") / f"{stem}.json"
