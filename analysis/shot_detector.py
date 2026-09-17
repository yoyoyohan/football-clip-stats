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
    min_shot_speed_mps: float = 10.0
    # High-speed shots may use a slightly shorter confirm window (still >=1 prior frame).
    fast_shot_speed_mps: float = 18.0
    fast_confirm_frames: int = 2
    min_attack_x_team0: float = 55.0
    max_attack_x_team1: float = 50.0
    cooldown_frames: int = 45
    confirm_frames: int = 2
    # Goal-mouth projection tolerance beyond posts (meters).
    goal_mouth_y_margin_m: float = 6.5
    # Reject shot if a teammate sits on the ball path within this corridor.
    teammate_path_lateral_m: float = 3.5
    teammate_path_min_ahead_m: float = 3.0
    teammate_path_max_ahead_m: float = 25.0

    events: list[ShotEvent] = field(default_factory=list)
    _last_shot_frame: int = -999
    _pending: dict | None = None
    _confirm_streak: int = 0
    _prev_xy: tuple[float, float] | None = None

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

    def _vx_toward_goal(self, team_id: int, vx_sign: float) -> bool:
        if team_id == 0:
            return vx_sign > 0
        return vx_sign < 0

    def _trajectory_toward_goal(
        self,
        x_m: float,
        y_m: float,
        prev_x: float,
        prev_y: float,
        team_id: int,
    ) -> bool:
        """True when the ball segment extrapolates into/near the defended goal mouth."""
        goal_x, y0, y1 = goal_mouth_bounds_m(self._defended_goal(team_id))
        dx = x_m - prev_x
        dy = y_m - prev_y
        if team_id == 0 and dx <= 0:
            return False
        if team_id == 1 and dx >= 0:
            return False
        if abs(dx) < 1e-6:
            return False
        t = (goal_x - x_m) / dx
        if t < 0:
            return False
        y_at_goal = y_m + t * dy
        margin = self.goal_mouth_y_margin_m
        return (y0 - margin) <= y_at_goal <= (y1 + margin)

    def _distance_decreasing(
        self,
        x_m: float,
        y_m: float,
        prev_x: float,
        prev_y: float,
        team_id: int,
    ) -> bool:
        prev_dist = self._distance_to_goal(prev_x, prev_y, team_id)
        dist = self._distance_to_goal(x_m, y_m, team_id)
        # Allow tiny noise; reject clear movement away from goal.
        return dist <= prev_dist + 0.5

    def _teammate_is_destination(
        self,
        x_m: float,
        y_m: float,
        prev_x: float,
        prev_y: float,
        team_id: int,
        players: list[dict] | None,
        possessor_track_id: int | None,
    ) -> bool:
        """True when a same-team player lies clearly on the ball's path (through-ball)."""
        if not players:
            return False
        dx = x_m - prev_x
        dy = y_m - prev_y
        mag = math.hypot(dx, dy)
        if mag < 1e-6:
            return False
        ux, uy = dx / mag, dy / mag
        for p in players:
            if p.get("team_id") != team_id:
                continue
            tid = p.get("track_id")
            if tid is None or tid == possessor_track_id:
                continue
            bbox = p.get("bbox")
            if bbox is None:
                continue
            foot_px = self.pitch.foot_px(bbox)
            mx, my = self.pitch.to_meters(foot_px)
            to_x, to_y = mx - x_m, my - y_m
            ahead = to_x * ux + to_y * uy
            if ahead < self.teammate_path_min_ahead_m or ahead > self.teammate_path_max_ahead_m:
                continue
            lateral = abs(to_x * uy - to_y * ux)
            if lateral <= self.teammate_path_lateral_m:
                return True
        return False


    def _nearest_player_team(
        self,
        ball_px: tuple[float, float],
        players: list[dict] | None,
        max_dist_m: float = 6.0,
    ) -> tuple[int | None, int | None]:
        """Infer (track_id, team_id) from nearest on-pitch player when possession is flaky."""
        if not players:
            return None, None
        best_tid, best_team, best_d = None, None, float("inf")
        for p in players:
            team = p.get("team_id")
            tid = p.get("track_id")
            bbox = p.get("bbox")
            if team is None or tid is None or bbox is None:
                continue
            foot_px = self.pitch.foot_px(bbox)
            d = self.pitch.distance_m(foot_px, ball_px)
            if d < best_d:
                best_d = d
                best_tid = tid
                best_team = team
        if best_d <= max_dist_m:
            return best_tid, best_team
        return None, None

    def _toward_goal(
        self,
        x_m: float,
        y_m: float,
        prev_xy: tuple[float, float],
        team_id: int,
        vx_sign: float,
        players: list[dict] | None,
        possessor_track_id: int | None,
    ) -> bool:
        if not self._vx_toward_goal(team_id, vx_sign):
            return False
        prev_x, prev_y = prev_xy
        if not self._trajectory_toward_goal(x_m, y_m, prev_x, prev_y, team_id):
            return False
        if not self._distance_decreasing(x_m, y_m, prev_x, prev_y, team_id):
            return False
        if self._teammate_is_destination(
            x_m, y_m, prev_x, prev_y, team_id, players, possessor_track_id
        ):
            return False
        return True

    def _clear_pending(self) -> None:
        self._pending = None
        self._confirm_streak = 0

    def update(
        self,
        frame_idx: int,
        ball_px: tuple[float, float] | None,
        speed_mps: float,
        vx_sign: float,
        possessor_track_id: int | None,
        possessor_team: int | None,
        ball_observed: bool = True,
        players: list[dict] | None = None,
    ) -> ShotEvent | None:
        if ball_px is None:
            self._clear_pending()
            self._prev_xy = None
            return None

        # Interpolated / held frames: keep pending confirmation alive. Sparse
        # ball detectors (common after finetunes) otherwise never get two
        # consecutive observed toward-goal frames and report 0 shots.
        if not ball_observed:
            x_m, y_m = self.pitch.to_meters(ball_px)
            if self.pitch.in_pitch(x_m, y_m):
                self._prev_xy = (x_m, y_m)
            return None

        x_m, y_m = self.pitch.to_meters(ball_px)
        if not self.pitch.in_pitch(x_m, y_m):
            self._prev_xy = (x_m, y_m)
            return None

        prev_xy = self._prev_xy
        self._prev_xy = (x_m, y_m)

        # When jersey/possession IDs are flaky (e.g. all-black kits), fall back to
        # nearest player context so clear goal-bound trajectories still register.
        shooter_tid = possessor_track_id
        team_hint = possessor_team
        if team_hint is None:
            near_tid, near_team = self._nearest_player_team(ball_px, players)
            if near_team is not None:
                team_hint = near_team
                if shooter_tid is None:
                    shooter_tid = near_tid

        team = self._attacking_team(x_m, speed_mps, vx_sign, team_hint)
        if team is None and team_hint is not None and speed_mps >= self.min_shot_speed_mps:
            # Trust player context in the final third even if vx_sign is noisy.
            if team_hint == 0 and x_m >= self.min_attack_x_team0 * 0.9:
                team = 0
            elif team_hint == 1 and x_m <= self.max_attack_x_team1 * 1.1:
                team = 1
        if team is None or speed_mps < self.min_shot_speed_mps:
            self._clear_pending()
            return None

        # Need a prior observed point to judge trajectory (avoids single-frame FPs).
        if prev_xy is None:
            return None

        if not self._toward_goal(
            x_m, y_m, prev_xy, team, vx_sign, players, shooter_tid
        ):
            self._clear_pending()
            return None

        defended = self._defended_goal(team)
        on_target = self.pitch.in_goal_mouth(x_m, y_m, defended)
        dist = self._distance_to_goal(x_m, y_m, team)

        if self._pending and self._pending["team"] == team:
            # Confirm frames must keep closing on goal.
            if dist <= float(self._pending.get("last_dist", dist)) + 0.5:
                self._confirm_streak += 1
                self._pending["on_target"] = self._pending["on_target"] or on_target
                self._pending["last_dist"] = dist
                self._pending["speed"] = max(float(self._pending["speed"]), speed_mps)
            else:
                self._pending = {
                    "team": team,
                    "frame": frame_idx,
                    "shooter": possessor_track_id,
                    "speed": speed_mps,
                    "dist": dist,
                    "last_dist": dist,
                    "on_target": on_target,
                    "x_m": x_m,
                    "y_m": y_m,
                }
                self._confirm_streak = 1
        else:
            self._pending = {
                "team": team,
                "frame": frame_idx,
                "shooter": possessor_track_id,
                "speed": speed_mps,
                "dist": dist,
                "last_dist": dist,
                "on_target": on_target,
                "x_m": x_m,
                "y_m": y_m,
            }
            self._confirm_streak = 1

        needed = self.confirm_frames
        if float(self._pending["speed"]) >= self.fast_shot_speed_mps:
            # Still require at least 2 toward-goal frames so interpolated gaps cannot
            # single-frame a shot after pending was cleared.
            needed = max(2, min(needed, self.fast_confirm_frames))
        if self._confirm_streak < needed:
            return None
        if frame_idx - self._last_shot_frame < self.cooldown_frames:
            return None

        event = ShotEvent(
            frame=frame_idx,
            team_id=team,
            shooter_track_id=shooter_tid if shooter_tid is not None else possessor_track_id,
            ball_speed_mps=float(self._pending["speed"]),
            distance_to_goal_m=float(self._pending["dist"]),
            on_target=bool(self._pending["on_target"]),
            xg=self._compute_xg(float(self._pending["dist"])),
            ball_x_m=float(self._pending["x_m"]),
            ball_y_m=float(self._pending["y_m"]),
        )
        self.events.append(event)
        self._last_shot_frame = frame_idx
        self._clear_pending()
        return event

    def counts(self) -> dict:
        return {
            "team0_shots": sum(1 for e in self.events if e.team_id == 0),
            "team1_shots": sum(1 for e in self.events if e.team_id == 1),
            "team0_shots_on_target": sum(1 for e in self.events if e.team_id == 0 and e.on_target),
            "team1_shots_on_target": sum(1 for e in self.events if e.team_id == 1 and e.on_target),
        }
