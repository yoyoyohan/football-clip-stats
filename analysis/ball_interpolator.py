from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BallObservation:
    position: tuple[float, float] | None
    observed: bool


class BallInterpolator:
    """Fill missing ball positions via linear interpolation between detections."""

    REF_FPS = 30.0
    MAX_GAP_AT_REF = 20

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self._max_gap = max(1, round(self.MAX_GAP_AT_REF * fps / self.REF_FPS))
        self._history: list[tuple[int, tuple[float, float]]] = []

    def update(self, frame_idx: int, ball: tuple[float, float] | None) -> BallObservation:
        if ball is not None:
            self._history.append((frame_idx, ball))
            return BallObservation(position=ball, observed=True)
        if len(self._history) < 2:
            return BallObservation(position=None, observed=False)
        prev_idx, prev_pos = self._history[-2]
        curr_idx, curr_pos = self._history[-1]
        if frame_idx <= curr_idx:
            return BallObservation(position=curr_pos, observed=False)
        if frame_idx - curr_idx > self._max_gap:
            return BallObservation(position=None, observed=False)
        gap = curr_idx - prev_idx
        if gap <= 0:
            return BallObservation(position=curr_pos, observed=False)
        t = (frame_idx - prev_idx) / gap
        x = prev_pos[0] + t * (curr_pos[0] - prev_pos[0])
        y = prev_pos[1] + t * (curr_pos[1] - prev_pos[1])
        return BallObservation(position=(float(x), float(y)), observed=False)

    def reset(self):
        self._history.clear()
