"""Detect passes when a teammate stably receives the ball at their feet."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from analysis.hybrid_event_classifier import HybridEventClassifier
    from analysis.pitch_coordinates import PitchCoordinateMapper


REF_WIDTH = 1280.0
REF_FPS = 30.0

# Soft caps for rejecting tracking teleports.
# Instantaneous FIFA max (~40 m/s) is NOT used as a per-frame gate: wide-FOV
# / default homographies map ordinary on-screen motion above that. Jump
# distance is the reliable teleport signal; avg flight m/s uses a looser cap.
MAX_BALL_SPEED_MPS = 42.0
MAX_BALL_JUMP_M = 8.0  # per frame at typical fps — beyond this is a teleport


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _foot(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, bbox[3])


@dataclass
class PassEvent:
    frame: int
    team_id: int
    from_track_id: int
    to_track_id: int
    ball_speed_peak: float
    release_frame: int = 0
    release_ball_px: tuple[float, float] | None = None
    receive_ball_px: tuple[float, float] | None = None


@dataclass
class PassDetector:
    fps: float = REF_FPS
    frame_width: float = REF_WIDTH
    possession_radius: float = 100.0
    control_radius: float = 28.0
    long_receive_radius: float = 80.0
    nearby_radius: float = 130.0
    high_speed: float = 10.0
    release_dist: float = 32.0
    long_pass_peak: float = 16.0
    min_velocity_peak: float = 8.0
    # Meter-space velocity gates (used when pitch is available). Wide-FOV /
    # distant cameras yield low px/frame even for real passes; meters stay
    # pass-like (~5–10 m/s). Either pixel OR meter peak may clear the gate.
    min_velocity_peak_mps: float = 5.5
    high_speed_mps: float = 6.5
    long_pass_peak_mps: float = 10.0
    # Abort in-flight state after this many ref frames without a receive so a
    # weak/orphan release cannot lock the detector for the rest of the clip.
    max_flight_frames: int = 60
    cooldown_frames: int = 10
    min_receiver_dist: float = 22.0
    min_ball_travel: float = 28.0
    control_stable_frames: int = 2
    min_long_receive_frames: int = 18
    # Only block near-instant bounce-backs (noise). Real give-and-gos are kept.
    return_window_frames: int = 22  # ~0.75s at 30fps ref
    # Meter-space thresholds (used when pitch mapper is available).
    min_ball_travel_m: float = 2.0
    min_receiver_dist_m: float = 1.2
    release_dist_m: float = 2.0
    max_ball_speed_mps: float = MAX_BALL_SPEED_MPS
    max_ball_jump_m: float = MAX_BALL_JUMP_M
    pitch: PitchCoordinateMapper | None = None
    event_classifier: HybridEventClassifier | None = None

    events: list[PassEvent] = field(default_factory=list)
    _last_pass_frame: int = -999
    _prev_ball: tuple[float, float] | None = None
    _prev_ball_frame: int = -1
    _peak_speed: float = 0.0
    _peak_speed_mps: float = 0.0
    _in_flight: bool = False
    _release_frame: int = 0
    _ball_at_release: tuple[float, float] | None = None
    _passer_at_release: int | None = None
    _passer_team: int | None = None
    _passer_pos: tuple[float, float] | None = None
    _last_control_tid: int | None = None
    _last_control_team: int | None = None
    _last_control_pos: tuple[float, float] | None = None
    _control_tid: int | None = None
    _control_streak: int = 0
    _nearby_tid: int | None = None
    _nearby_team: int | None = None
    _long_receive_done: bool = False
    _flight_observations: int = 0
    _last_pass_ball: tuple[float, float] | None = None
    _current_players: list[dict] = field(default_factory=list, repr=False)

    def _px(self, value_at_ref: float) -> float:
        return value_at_ref * self.frame_width / REF_WIDTH

    def _frames(self, count_at_ref: int) -> int:
        return max(1, round(count_at_ref * self.fps / REF_FPS))

    def _norm_speed(self, px_per_frame: float) -> float:
        return px_per_frame * self.fps / REF_FPS

    def _meters(self, a: tuple[float, float], b: tuple[float, float]) -> float | None:
        if self.pitch is None:
            return None
        return self.pitch.distance_m(a, b)

    def _speed_mps(self, a: tuple[float, float], b: tuple[float, float], gap: int = 1) -> float | None:
        if self.pitch is None:
            return None
        return self.pitch.speed_mps(a, b, frame_gap=gap)

    def _on_pitch(self, ball: tuple[float, float]) -> bool:
        if self.pitch is None:
            return True
        x_m, y_m = self.pitch.to_meters(ball)
        return self.pitch.in_pitch(x_m, y_m)

    def _implausible_segment(
        self, prev: tuple[float, float], curr: tuple[float, float], gap: int
    ) -> bool:
        """Reject tracking teleports that would inflate pass counts / speeds.

        Use spatial jump (meters or pixels), not instantaneous m/s.

        Wide-FOV / default homographies map ~30 px/frame to >42 m/s even for
        ordinary kicks, so a FIFA max-speed gate falsely aborts every real
        flight and yields 0 passes. True teleports are huge jumps (many meters
        or a large fraction of the frame) — those still get rejected here.
        """
        gap = max(1, gap)
        px_per_frame = _dist(prev, curr) / gap
        if px_per_frame > self.frame_width * 0.35:
            return True
        jump = self._meters(prev, curr)
        if jump is not None and jump > self.max_ball_jump_m * gap:
            return True
        return False

    def _travel_ok(self, release: tuple[float, float], receive: tuple[float, float]) -> bool:
        travel_m = self._meters(release, receive)
        if travel_m is not None:
            return travel_m >= self.min_ball_travel_m
        return _dist(release, receive) >= self._px(self.min_ball_travel)

    def _receiver_sep_ok(self, passer_pos: tuple[float, float], recv_pos: tuple[float, float]) -> bool:
        sep_m = self._meters(passer_pos, recv_pos)
        if sep_m is not None:
            return sep_m >= self.min_receiver_dist_m
        return _dist(passer_pos, recv_pos) >= self._px(self.min_receiver_dist)

    def _release_dist_ok(self, control_pos: tuple[float, float], ball: tuple[float, float]) -> bool:
        d_m = self._meters(control_pos, ball)
        if d_m is not None:
            return d_m >= self.release_dist_m
        return _dist(control_pos, ball) >= self._px(self.release_dist)

    def _closest_within(
        self, players: list[dict], ball: tuple[float, float], radius: float
    ) -> tuple[dict | None, float]:
        best, best_d = None, float("inf")
        for p in players:
            if p.get("team_id") is None or p.get("track_id") is None:
                continue
            d = _dist(_foot(p["bbox"]), ball)
            if d < best_d:
                best_d = d
                best = p
        if best is not None and best_d <= radius:
            return best, best_d
        return None, best_d

    def _receiver_candidate(
        self, players: list[dict], ball: tuple[float, float], radius: float
    ) -> tuple[dict | None, float]:
        if self._passer_at_release is None:
            return None, float("inf")
        best, best_d = None, float("inf")
        for p in players:
            if p.get("team_id") != self._passer_team:
                continue
            tid = p.get("track_id")
            if tid is None or tid == self._passer_at_release:
                continue
            d = _dist(_foot(p["bbox"]), ball)
            if d < best_d:
                best_d = d
                best = p
        if best is not None and best_d <= radius:
            return best, best_d
        return None, best_d

    def _frame_speed(self, frame_idx: int, ball: tuple[float, float], ball_speed: float) -> float:
        if self._prev_ball is not None and frame_idx == self._prev_ball_frame + 1:
            return self._norm_speed(_dist(self._prev_ball, ball))
        if ball_speed > 0:
            return self._norm_speed(min(ball_speed, 60.0))
        return 0.0


    def _peak_velocity_ok(
        self,
        min_px: float | None = None,
        min_mps: float | None = None,
    ) -> bool:
        """True if pixel-normalized peak OR meter peak clears the threshold.

        Pixel-only when pitch is None (unit tests / uncalibrated runs).
        """
        min_px = self.min_velocity_peak if min_px is None else min_px
        min_mps = self.min_velocity_peak_mps if min_mps is None else min_mps
        if self._peak_speed >= min_px:
            return True
        if self.pitch is not None and self._peak_speed_mps >= min_mps:
            return True
        return False

    def _cooldown_ok(self, frame_idx: int, from_tid: int, to_tid: int) -> bool:
        gap = frame_idx - self._last_pass_frame
        needed = self._frames(self.cooldown_frames)
        if self.events:
            last = self.events[-1]
            if (
                from_tid == last.to_track_id
                and to_tid == last.from_track_id
                and last.ball_speed_peak >= self.long_pass_peak
            ):
                needed = self._frames(6)
        return gap >= needed

    def _is_return_pass(self, from_tid: int, to_tid: int, frame_idx: int) -> bool:
        """Block only noise bounce-backs, not real give-and-go passes."""
        if not self.events:
            return False
        last = self.events[-1]
        if last.from_track_id != to_tid or last.to_track_id != from_tid:
            return False

        gap_frames = frame_idx - last.frame
        gap_secs = gap_frames / self.fps
        if gap_secs >= self.return_window_frames / REF_FPS:
            return False

        travel = 0.0
        if self._ball_at_release is not None and self._last_pass_ball is not None:
            travel = _dist(self._last_pass_ball, self._ball_at_release)

        # Near-instant tiny returns are almost always tracking noise.
        if gap_frames <= self._frames(8) and travel < self._px(self.min_ball_travel * 0.5):
            return True

        # Confirmed ball flight with meaningful travel → real give-and-go.
        if self._in_flight and (
            self._peak_velocity_ok() or self._flight_observations >= 1
        ):
            if travel >= self._px(self.min_ball_travel * 0.5):
                return False
            # Flight claimed but ball barely moved: still treat as bounce-back noise.
            return True

        if travel >= self._px(self.min_ball_travel * 0.5):
            return False

        return True

    def _begin_flight(self, frame_idx: int, ball: tuple[float, float]) -> None:
        if self._in_flight:
            return
        self._in_flight = True
        self._release_frame = frame_idx
        self._ball_at_release = ball
        self._passer_at_release = self._last_control_tid or self._nearby_tid
        self._passer_team = (
            self._last_control_team
            if self._last_control_team is not None
            else self._nearby_team
        )
        self._passer_pos = self._last_control_pos
        if self._nearby_tid is not None and self._passer_at_release is None:
            self._passer_at_release = self._nearby_tid
            self._passer_team = self._nearby_team
        self._peak_speed = 0.0
        self._peak_speed_mps = 0.0
        self._long_receive_done = False
        self._flight_observations = 0

    def _abort_flight(self) -> None:
        self._in_flight = False
        self._passer_at_release = None
        self._long_receive_done = False
        self._flight_observations = 0
        self._peak_speed = 0.0
        self._peak_speed_mps = 0.0

    def _record_pass(
        self,
        frame_idx: int,
        team_id: int,
        from_tid: int,
        to_tid: int,
        recv_pos: tuple[float, float],
        ball: tuple[float, float],
    ) -> PassEvent:
        event = PassEvent(
            frame=frame_idx,
            team_id=team_id,
            from_track_id=from_tid,
            to_track_id=to_tid,
            ball_speed_peak=self._peak_speed,
            release_frame=self._release_frame,
            release_ball_px=self._ball_at_release,
            receive_ball_px=ball,
        )
        self.events.append(event)
        self._last_pass_frame = frame_idx
        self._last_pass_ball = ball
        self._last_control_tid = to_tid
        self._last_control_team = team_id
        self._last_control_pos = recv_pos
        self._in_flight = False
        self._passer_at_release = None
        self._long_receive_done = False
        self._flight_observations = 0
        self._peak_speed = 0.0
        self._peak_speed_mps = 0.0
        return event

    def _build_pass_features(
        self,
        frame_idx: int,
        ball: tuple[float, float],
        receiver: dict,
        recv_pos: tuple[float, float],
        mode: str,
    ):
        from analysis.event_features import PassCandidateFeatures, count_players_near

        travel = 0.0
        if self._ball_at_release is not None:
            travel = _dist(self._ball_at_release, ball)
        pr_dist = _dist(self._passer_pos, recv_pos) if self._passer_pos else 0.0
        flight_frames = max(0, frame_idx - self._release_frame)
        return PassCandidateFeatures(
            frame=frame_idx,
            release_frame=self._release_frame,
            fps=self.fps,
            frame_width=self.frame_width,
            from_track_id=self._passer_at_release or 0,
            to_track_id=receiver["track_id"],
            team_id=receiver["team_id"],
            ball_speed_peak=self._peak_speed,
            ball_travel_px=travel,
            passer_receiver_dist_px=pr_dist,
            control_streak=self._control_streak,
            flight_frames=flight_frames,
            ball_observed_fraction=self._flight_observations / max(1, flight_frames),
            players_near_ball=count_players_near(
                self._current_players, ball, self._px(self.nearby_radius)
            ),
            recv_mode=mode,
            geometry_score=1.0,
        )

    def _try_pass_to(
        self,
        frame_idx: int,
        ball: tuple[float, float],
        receiver: dict,
        recv_dist: float,
        mode: str,
        ball_observed: bool,
    ) -> PassEvent | None:
        to_tid = receiver["track_id"]
        team = receiver["team_id"]
        from_tid = self._passer_at_release
        if from_tid is None or from_tid == to_tid:
            return None
        if not self._cooldown_ok(frame_idx, from_tid, to_tid):
            return None
        if not self._in_flight:
            return None
        if not self._peak_velocity_ok():
            return None
        # Require at least one observed in-flight frame (blocks interpolated FPs).
        if self._flight_observations < 1:
            return None
        if self._passer_team is not None and team != self._passer_team:
            return None
        if self._is_return_pass(from_tid, to_tid, frame_idx):
            return None
        if not self._on_pitch(ball):
            return None
        if self._ball_at_release is not None and not self._on_pitch(self._ball_at_release):
            return None

        recv_pos = _foot(receiver["bbox"])
        if self._passer_pos is not None and not self._receiver_sep_ok(self._passer_pos, recv_pos):
            return None
        if self._ball_at_release is not None and not self._travel_ok(self._ball_at_release, ball):
            return None

        # Reject only absurd average flight speeds. Keep this loose: default /
        # wide-FOV homographies inflate m/s, and short hard passes are often
        # >42 m/s even with good calibration.
        if self._ball_at_release is not None and self.pitch is not None:
            gap = max(1, frame_idx - self._release_frame)
            avg_mps = self.pitch.speed_mps(self._ball_at_release, ball, frame_gap=gap)
            if avg_mps > self.max_ball_speed_mps * 2.0:
                return None

        if mode == "control":
            if recv_dist > self._px(self.control_radius):
                return None
            if self._control_streak < self._frames(self.control_stable_frames):
                return None
            if not ball_observed:
                return None
        else:
            if self._long_receive_done:
                return None
            if not self._peak_velocity_ok(
                min_px=self.long_pass_peak, min_mps=self.long_pass_peak_mps
            ):
                return None
            if frame_idx - self._release_frame < self._frames(self.min_long_receive_frames):
                return None
            if recv_dist > self._px(self.long_receive_radius):
                return None
            if not ball_observed:
                return None
            self._long_receive_done = True

        if self.event_classifier is not None:
            features = self._build_pass_features(frame_idx, ball, receiver, recv_pos, mode)
            result = self.event_classifier.classify_pass(features)
            if not result.accepted:
                return None

        return self._record_pass(frame_idx, team, from_tid, to_tid, recv_pos, ball)

    def update(
        self,
        frame_idx: int,
        ball: tuple[float, float] | None,
        ball_speed: float,
        players: list[dict],
        ball_observed: bool = True,
    ) -> PassEvent | None:
        if ball is None:
            return None

        self._current_players = players

        # Drop / ignore teleport jumps so they never open a fake flight.
        if (
            self._prev_ball is not None
            and frame_idx > self._prev_ball_frame
            and self._implausible_segment(
                self._prev_ball, ball, max(1, frame_idx - self._prev_ball_frame)
            )
        ):
            if self._in_flight:
                self._abort_flight()
            self._prev_ball = ball if ball_observed else self._prev_ball
            self._prev_ball_frame = frame_idx
            return None

        speed = self._frame_speed(frame_idx, ball, ball_speed)
        if speed > 0:
            self._peak_speed = max(self._peak_speed, speed)
        if (
            self._prev_ball is not None
            and frame_idx == self._prev_ball_frame + 1
            and self.pitch is not None
        ):
            mps = self.pitch.speed_mps(self._prev_ball, ball, frame_gap=1)
            self._peak_speed_mps = max(self._peak_speed_mps, mps)

        if ball_observed and self._in_flight:
            self._flight_observations += 1

        control_radius = self._px(self.control_radius)
        controller, control_dist = self._closest_within(players, ball, control_radius)
        nearby, _ = self._closest_within(players, ball, self._px(self.nearby_radius))

        if nearby is not None:
            self._nearby_tid = nearby["track_id"]
            self._nearby_team = nearby["team_id"]

        if controller is not None:
            tid = controller["track_id"]
            if tid == self._control_tid:
                self._control_streak += 1
            else:
                self._control_tid = tid
                self._control_streak = 1
        else:
            self._control_tid = None
            self._control_streak = 0

        stable_control = self._frames(self.control_stable_frames)
        # Require a clearer release: high speed alone, or moderate speed plus
        # leaving the last controller's feet. Avoids slow dribble "flights".
        # When pitch is available, meter speed can also open a release even if
        # pixel speed is low (wide-FOV / distant cameras).
        dist_release = (
            self._last_control_pos is not None
            and self._release_dist_ok(self._last_control_pos, ball)
        )
        mps_now = 0.0
        if (
            self.pitch is not None
            and self._prev_ball is not None
            and frame_idx == self._prev_ball_frame + 1
        ):
            mps_now = self.pitch.speed_mps(self._prev_ball, ball, frame_gap=1)
        releasing = (
            speed >= self.high_speed
            or (dist_release and speed >= self.min_velocity_peak * 0.45)
            or (self.pitch is not None and mps_now >= self.high_speed_mps)
            or (
                self.pitch is not None
                and dist_release
                and mps_now >= self.min_velocity_peak_mps * 0.7
            )
        )
        has_passer_context = (
            self._last_control_tid is not None or self._nearby_tid is not None
        )
        # Allow release even while the ball is still inside the (generous) control
        # radius of the same passer, once it has clearly left their feet at
        # pass-like speed. Otherwise meter-space teleport caps prevent any
        # single-frame exit from control_radius.
        same_controller_leaving = (
            controller is not None
            and self._last_control_tid is not None
            and controller["track_id"] == self._last_control_tid
            and dist_release
            and (
                speed >= self.min_velocity_peak
                or (self.pitch is not None and mps_now >= self.min_velocity_peak_mps)
            )
        )
        open_release = controller is None or self._control_streak < stable_control
        if releasing and has_passer_context and (open_release or same_controller_leaving):
            # begin_flight zeroes peaks; keep the release-frame speed so a
            # sparse-observation flight can still clear velocity gates.
            self._begin_flight(frame_idx, ball)
            self._peak_speed = max(self._peak_speed, speed)
            if self.pitch is not None:
                self._peak_speed_mps = max(self._peak_speed_mps, mps_now)

        if self._in_flight:
            # Stale flight: one weak release must not lock the detector forever.
            if frame_idx - self._release_frame >= self._frames(self.max_flight_frames):
                self._abort_flight()
            elif controller is not None:
                event = self._try_pass_to(
                    frame_idx, ball, controller, control_dist, "control", ball_observed
                )
                if event is not None:
                    self._prev_ball = ball
                    self._prev_ball_frame = frame_idx
                    return event

            if self._in_flight:
                long_recv, long_dist = self._receiver_candidate(
                    players, ball, self._px(self.long_receive_radius)
                )
                long_peak_ok = self._peak_velocity_ok(
                    min_px=self.long_pass_peak, min_mps=self.long_pass_peak_mps
                )
                if long_recv is not None and long_peak_ok:
                    event = self._try_pass_to(
                        frame_idx, ball, long_recv, long_dist, "long", ball_observed
                    )
                    if event is not None:
                        self._prev_ball = ball
                        self._prev_ball_frame = frame_idx
                        return event

        if not self._in_flight and controller is not None and self._control_streak >= stable_control:
            tid = controller["track_id"]
            team = controller["team_id"]
            pos = _foot(controller["bbox"])
            if self._last_control_tid is None:
                self._last_control_tid = tid
                self._last_control_team = team
                self._last_control_pos = pos
            elif tid != self._last_control_tid:
                self._last_control_tid = tid
                self._last_control_team = team
                self._last_control_pos = pos
            else:
                self._last_control_pos = pos

        self._prev_ball = ball
        self._prev_ball_frame = frame_idx
        return None

    def counts(self) -> dict[str, int]:
        return {
            "team0_passes": sum(1 for e in self.events if e.team_id == 0),
            "team1_passes": sum(1 for e in self.events if e.team_id == 1),
        }
