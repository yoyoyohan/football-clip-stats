"""Detect shots and shots on target using pitch coordinates + ball speed."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from analysis.pitch_coordinates import PitchCoordinateMapper, goal_mouth_bounds_m
from utils.calibration import PITCH_LENGTH_M


@dataclass
class ShotEvent:
    frame: int
    team_id: int
    shooter_track_id: int | None
    ball_speed_mps: float
    distance_to_goal_m: float
    on_target: bool
    xg: float
    ball_x_m: float
    ball_y_m: float


@dataclass
class ShotDetector:
    pitch: PitchCoordinateMapper
    attacking_direction: str = "left_to_right"
    min_shot_speed_mps: float = 12.0
    min_attack_x_team0: float = 55.0
    max_attack_x_team1: float = 50.0
    cooldown_frames: int = 45
    confirm_frames: int = 2

    events: list[ShotEvent] = field(default_factory=list)
    _last_shot_frame: int = -999
    _pending: dict | None = None
    _confirm_streak: int = 0

    def _defended_goal(self, team_id: int) -> str:
        return "team1" if team_id == 0 else "team0"

    def _distance_to_goal(self, x_m: float, y_m: float, team_id: int) -> float:
        goal_x, y0, y1 = goal_mouth_bounds_m(self._defended_goal(team_id))
        mouth_y = (y0 + y1) / 2.0
        return math.hypot(x_m - goal_x, y_m - mouth_y)

    def _compute_xg(self, dist_m: float) -> float:
        return float(max(0.02, min(0.75, 0.35 * math.exp(-0.08 * dist_m))))

    def _attacking_team(
        self,
        x_m: float,
        speed_mps: float,
        vx_sign: float,
        possessor_team: int | None,
    ) -> int | None:
        if x_m >= self.min_attack_x_team0 and (vx_sign > 0 or possessor_team == 0):
            return 0
        if x_m <= self.max_attack_x_team1 and (vx_sign < 0 or possessor_team == 1):
            return 1
        if possessor_team == 0 and x_m > PITCH_LENGTH_M * 0.55:
            return 0
        if possessor_team == 1 and x_m < PITCH_LENGTH_M * 0.45:
            return 1
        return None

    def update(
        self,
        frame_idx: int,
        ball_px: tuple[float, float] | None,
        speed_mps: float,
        vx_sign: float,
        possessor_track_id: int | None,
        possessor_team: int | None,
        ball_observed: bool = True,
    ) -> ShotEvent | None:
        if ball_px is None or not ball_observed:
            self._pending = None
            self._confirm_streak = 0
            return None

        x_m, y_m = self.pitch.to_meters(ball_px)
        if not self.pitch.in_pitch(x_m, y_m):
            return None

        team = self._attacking_team(x_m, speed_mps, vx_sign, possessor_team)
        if team is None or speed_mps < self.min_shot_speed_mps:
            self._pending = None
            self._confirm_streak = 0
            return None

        defended = self._defended_goal(team)
        on_target = self.pitch.in_goal_mouth(x_m, y_m, defended)
        dist = self._distance_to_goal(x_m, y_m, team)

        if self._pending and self._pending["team"] == team:
            self._confirm_streak += 1
            self._pending["on_target"] = self._pending["on_target"] or on_target
        else:
            self._pending = {
                "team": team,
                "frame": frame_idx,
                "shooter": possessor_track_id,
                "speed": speed_mps,
                "dist": dist,
                "on_target": on_target,
                "x_m": x_m,
                "y_m": y_m,
            }
            self._confirm_streak = 1

        if self._confirm_streak < self.confirm_frames:
            return None
        if frame_idx - self._last_shot_frame < self.cooldown_frames:
            return None

        event = ShotEvent(
            frame=frame_idx,
            team_id=team,
            shooter_track_id=possessor_track_id,
            ball_speed_mps=float(self._pending["speed"]),
            distance_to_goal_m=float(self._pending["dist"]),
            on_target=bool(self._pending["on_target"]),
            xg=self._compute_xg(float(self._pending["dist"])),
            ball_x_m=float(self._pending["x_m"]),
            ball_y_m=float(self._pending["y_m"]),
        )
        self.events.append(event)
        self._last_shot_frame = frame_idx
        self._pending = None
        self._confirm_streak = 0
        return event

    def counts(self) -> dict:
        return {
            "team0_shots": sum(1 for e in self.events if e.team_id == 0),
            "team1_shots": sum(1 for e in self.events if e.team_id == 1),
            "team0_shots_on_target": sum(1 for e in self.events if e.team_id == 0 and e.on_target),
            "team1_shots_on_target": sum(1 for e in self.events if e.team_id == 1 and e.on_target),
        }
