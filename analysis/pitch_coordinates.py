"""Map pixel detections to FIFA pitch coordinates (meters) via homography.

This is the standard broadcast-football approach: a 3D pitch is projected to a
2D plane (105m x 68m). Full multi-view 3D reconstruction is not required for
pass distance, shot location, or possession zones.
"""

from __future__ import annotations

import numpy as np

from analysis.perspective_transformer import PerspectiveTransformer
from utils.calibration import GOAL_WIDTH_M, PITCH_LENGTH_M, PITCH_WIDTH_M


def goal_mouth_bounds_m(defended_side: str) -> tuple[float, float, float]:
    """Return (goal_line_x, y_min, y_max) in pitch meters for a defended goal."""
    y0 = (PITCH_WIDTH_M - GOAL_WIDTH_M) / 2.0
    y1 = (PITCH_WIDTH_M + GOAL_WIDTH_M) / 2.0
    if defended_side == "team0":
        return 0.0, y0, y1
    return PITCH_LENGTH_M, y0, y1


class PitchCoordinateMapper:
    def __init__(self, homography: np.ndarray, fps: float = 30.0):
        self.transformer = PerspectiveTransformer(homography)
        self.fps = fps

    def foot_px(self, bbox: list[float]) -> tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2.0, bbox[3])

    def to_meters(self, point_px: tuple[float, float]) -> tuple[float, float]:
        m = self.transformer.pixels_to_meters(np.array([point_px], dtype=np.float32))[0]
        return float(m[0]), float(m[1])

    def bbox_to_meters(self, bbox: list[float]) -> dict:
        foot = self.foot_px(bbox)
        x_m, y_m = self.to_meters(foot)
        return {
            "bbox_px": [float(v) for v in bbox],
            "foot_px": [foot[0], foot[1]],
            "x_m": x_m,
            "y_m": y_m,
        }

    def distance_m(self, a_px: tuple[float, float], b_px: tuple[float, float]) -> float:
        return self.transformer.distance_meters(a_px, b_px)

    def speed_mps(
        self, a_px: tuple[float, float], b_px: tuple[float, float], frame_gap: int = 1
    ) -> float:
        dt = max(1, frame_gap) / self.fps
        return self.distance_m(a_px, b_px) / dt

    def in_pitch(self, x_m: float, y_m: float, margin: float = 2.0) -> bool:
        return (
            -margin <= x_m <= PITCH_LENGTH_M + margin
            and -margin <= y_m <= PITCH_WIDTH_M + margin
        )

    def in_goal_mouth(self, x_m: float, y_m: float, defended_side: str, depth_m: float = 2.0) -> bool:
        line_x, y0, y1 = goal_mouth_bounds_m(defended_side)
        if defended_side == "team0":
            return y0 <= y_m <= y1 and -depth_m <= x_m <= depth_m
        return y0 <= y_m <= y1 and PITCH_LENGTH_M - depth_m <= x_m <= PITCH_LENGTH_M + depth_m
