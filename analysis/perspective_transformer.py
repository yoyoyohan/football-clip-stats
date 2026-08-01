import cv2
import numpy as np


class PerspectiveTransformer:
    """Map pixel coordinates to pitch coordinates (meters) via homography."""

    def __init__(self, homography: np.ndarray):
        self.homography = np.asarray(homography, dtype=np.float32)

    def pixels_to_meters(self, points: np.ndarray | list) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self.homography)
        return transformed.reshape(-1, 2).astype(np.float32)

    def distance_meters(
        self, p1: tuple[float, float], p2: tuple[float, float]
    ) -> float:
        m = self.pixels_to_meters(np.array([p1, p2], dtype=np.float32))
        return float(np.linalg.norm(m[1] - m[0]))

    def velocity_mps(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
        fps: int,
    ) -> float:
        return self.distance_meters(p1, p2) * fps
