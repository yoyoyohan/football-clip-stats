from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BallObservation:
    position: tuple[float, float] | None
    observed: bool
    confidence: float = 0.0


def _hypot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class BallInterpolator:
    """Associate ball detections over time, reject teleports, fill gaps.

    Online ``update`` applies a distance gate vs the predicted position.
    ``smooth_sequence`` / ``smooth_frame_records`` run an offline two-pass
    (gate then linear gap fill) — preferred for full-clip pipelines.
    """

    REF_FPS = 30.0
    MAX_GAP_AT_REF = 20
    # ~900 px/s on a 1920×1080 frame ≈ fast on-screen kick; scaled by diagonal.
    REF_DIAG = math.hypot(1920.0, 1080.0)
    REF_MAX_SPEED_PX_S = 900.0

    def __init__(
        self,
        fps: float = 30.0,
        frame_width: int = 1920,
        frame_height: int = 1080,
        max_gap: int | None = None,
        max_speed_px_s: float | None = None,
    ):
        self.fps = max(1e-3, float(fps))
        diag = math.hypot(float(frame_width), float(frame_height))
        scale = diag / self.REF_DIAG if self.REF_DIAG > 0 else 1.0
        self._max_speed_px_s = (
            float(max_speed_px_s)
            if max_speed_px_s is not None
            else self.REF_MAX_SPEED_PX_S * scale
        )
        self._max_px_per_frame = self._max_speed_px_s / self.fps
        self._max_gap = (
            int(max_gap)
            if max_gap is not None
            else max(1, round(self.MAX_GAP_AT_REF * self.fps / self.REF_FPS))
        )
        self._last_accepted_idx: int | None = None
        self._last_accepted_pos: tuple[float, float] | None = None
        self._prev_accepted_idx: int | None = None
        self._prev_accepted_pos: tuple[float, float] | None = None
        self._history: list[tuple[int, tuple[float, float]]] = []

    def _max_jump(self, dt_frames: int) -> float:
        dt = max(1, int(dt_frames))
        return self._max_px_per_frame * dt

    def _predict(self, frame_idx: int) -> tuple[float, float] | None:
        if self._last_accepted_pos is None or self._last_accepted_idx is None:
            return None
        if (
            self._prev_accepted_pos is None
            or self._prev_accepted_idx is None
            or self._last_accepted_idx == self._prev_accepted_idx
        ):
            return self._last_accepted_pos
        dt = self._last_accepted_idx - self._prev_accepted_idx
        if dt <= 0:
            return self._last_accepted_pos
        vx = (self._last_accepted_pos[0] - self._prev_accepted_pos[0]) / dt
        vy = (self._last_accepted_pos[1] - self._prev_accepted_pos[1]) / dt
        ahead = frame_idx - self._last_accepted_idx
        return (
            self._last_accepted_pos[0] + vx * ahead,
            self._last_accepted_pos[1] + vy * ahead,
        )

    def _accept(self, frame_idx: int, pos: tuple[float, float]) -> None:
        self._prev_accepted_idx = self._last_accepted_idx
        self._prev_accepted_pos = self._last_accepted_pos
        self._last_accepted_idx = frame_idx
        self._last_accepted_pos = pos
        self._history.append((frame_idx, pos))

    def update(
        self,
        frame_idx: int,
        ball: tuple[float, float] | None,
        confidence: float = 1.0,
    ) -> BallObservation:
        """Online update with jump gating (no look-ahead gap fill)."""
        if ball is not None:
            if self._last_accepted_pos is not None and self._last_accepted_idx is not None:
                dt = frame_idx - self._last_accepted_idx
                pred = self._predict(frame_idx)
                gate_ref = pred if pred is not None else self._last_accepted_pos
                if _hypot(ball, gate_ref) > self._max_jump(dt):
                    # Teleport — treat as missing this frame.
                    ball = None
                else:
                    self._accept(frame_idx, ball)
                    return BallObservation(position=ball, observed=True, confidence=float(confidence))
            else:
                self._accept(frame_idx, ball)
                return BallObservation(position=ball, observed=True, confidence=float(confidence))

        # Missing / rejected: hold last position only within max_gap (no velocity
        # extrapolation here — offline smooth_sequence does proper gap fill).
        if (
            self._last_accepted_pos is not None
            and self._last_accepted_idx is not None
            and 0 < frame_idx - self._last_accepted_idx <= self._max_gap
        ):
            return BallObservation(
                position=self._last_accepted_pos, observed=False, confidence=0.0
            )
        return BallObservation(position=None, observed=False, confidence=0.0)

    def smooth_sequence(
        self,
        detections: list[tuple[int, tuple[float, float] | None, float]],
    ) -> list[BallObservation]:
        """Two-pass offline smoother: gate teleports, then interpolate gaps."""
        n = len(detections)
        if n == 0:
            return []

        accepted: list[tuple[float, float, float] | None] = [None] * n
        last_idx: int | None = None
        last_pos: tuple[float, float] | None = None
        prev_idx: int | None = None
        prev_pos: tuple[float, float] | None = None

        for i, (frame_idx, pos, conf) in enumerate(detections):
            if pos is None:
                continue
            if last_pos is not None and last_idx is not None:
                dt = frame_idx - last_idx
                if dt <= 0:
                    continue
                # Gate vs constant-velocity prediction when we have two points.
                if prev_pos is not None and prev_idx is not None and last_idx != prev_idx:
                    span = last_idx - prev_idx
                    vx = (last_pos[0] - prev_pos[0]) / span
                    vy = (last_pos[1] - prev_pos[1]) / span
                    pred = (last_pos[0] + vx * dt, last_pos[1] + vy * dt)
                else:
                    pred = last_pos
                if _hypot(pos, pred) > self._max_jump(dt):
                    continue
            accepted[i] = (float(pos[0]), float(pos[1]), float(conf))
            prev_idx, prev_pos = last_idx, last_pos
            last_idx, last_pos = frame_idx, (float(pos[0]), float(pos[1]))

        # Precompute nearest accepted neighbors for gap fill.
        prev_acc = [None] * n
        next_acc = [None] * n
        last = None
        for i in range(n):
            if accepted[i] is not None:
                last = i
            prev_acc[i] = last
        last = None
        for i in range(n - 1, -1, -1):
            if accepted[i] is not None:
                last = i
            next_acc[i] = last

        out: list[BallObservation] = []
        for i, (frame_idx, _raw, _conf) in enumerate(detections):
            hit = accepted[i]
            if hit is not None:
                out.append(
                    BallObservation(
                        position=(hit[0], hit[1]),
                        observed=True,
                        confidence=hit[2],
                    )
                )
                continue

            pi, ni = prev_acc[i], next_acc[i]
            if pi is None or ni is None or pi == ni:
                out.append(BallObservation(position=None, observed=False))
                continue

            prev_frame = detections[pi][0]
            next_frame = detections[ni][0]
            gap = next_frame - prev_frame
            if gap <= 0 or gap > self._max_gap:
                out.append(BallObservation(position=None, observed=False))
                continue
            if frame_idx - prev_frame > self._max_gap or next_frame - frame_idx > self._max_gap:
                out.append(BallObservation(position=None, observed=False))
                continue

            t = (frame_idx - prev_frame) / gap
            p0 = accepted[pi]
            p1 = accepted[ni]
            assert p0 is not None and p1 is not None
            x = p0[0] + t * (p1[0] - p0[0])
            y = p0[1] + t * (p1[1] - p0[1])
            out.append(
                BallObservation(position=(float(x), float(y)), observed=False, confidence=0.0)
            )
        return out

    def smooth_frame_records(
        self, frame_records: list[dict]
    ) -> list[BallObservation]:
        """Smooth ``frame_records[*]['ball']`` (tuple or extract_ball dict)."""
        from utils.detection_utils import normalize_ball

        seq: list[tuple[int, tuple[float, float] | None, float]] = []
        for rec in frame_records:
            pos, conf, _area = normalize_ball(rec.get("ball"))
            seq.append((int(rec["frame_idx"]), pos, conf))
        return self.smooth_sequence(seq)

    def reset(self) -> None:
        self._last_accepted_idx = None
        self._last_accepted_pos = None
        self._prev_accepted_idx = None
        self._prev_accepted_pos = None
        self._history.clear()
