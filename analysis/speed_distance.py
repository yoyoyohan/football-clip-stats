import numpy as np

from .perspective_transformer import PerspectiveTransformer


class SpeedDistanceEstimator:
    """Track per-player distance (meters) and instantaneous speed."""

    def __init__(self, transformer: PerspectiveTransformer, fps: int = 25):
        self.transformer = transformer
        self.fps = fps
        self.distances: dict[int, float] = {}
        self.speeds: dict[int, float] = {}
        self._last_pos: dict[int, tuple[float, float]] = {}

    def update(self, track_id: int, center: tuple[float, float]) -> dict:
        dist_delta = 0.0
        speed = 0.0
        if track_id in self._last_pos:
            dist_delta = self.transformer.distance_meters(self._last_pos[track_id], center)
            speed = dist_delta * self.fps
            self.distances[track_id] = self.distances.get(track_id, 0.0) + dist_delta
        self._last_pos[track_id] = center
        self.speeds[track_id] = speed
        return {"distance_m": self.distances.get(track_id, 0.0), "speed_mps": speed}

    def get_distances(self) -> dict[int, float]:
        return dict(self.distances)
