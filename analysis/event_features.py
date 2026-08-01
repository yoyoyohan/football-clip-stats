"""Feature vectors for hybrid geometry + ML event classification."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class PassCandidateFeatures:
    """Features describing a proposed pass at receive time."""

    event_type: str = "pass"
    frame: int = 0
    release_frame: int = 0
    fps: float = 30.0
    frame_width: float = 1280.0
    from_track_id: int = 0
    to_track_id: int = 0
    team_id: int = 0
    ball_speed_peak: float = 0.0
    ball_travel_px: float = 0.0
    passer_receiver_dist_px: float = 0.0
    control_streak: int = 0
    possession_dwell_frames: int = 0
    flight_frames: int = 0
    ball_observed_fraction: float = 0.0
    players_near_ball: int = 0
    same_team: int = 1
    recv_mode: str = "control"  # control | long
    geometry_score: float = 1.0

    def to_vector(self) -> list[float]:
        gap_secs = (self.frame - self.release_frame) / max(self.fps, 1.0)
        return [
            self.ball_speed_peak,
            self.ball_travel_px / max(self.frame_width, 1.0),
            self.passer_receiver_dist_px / max(self.frame_width, 1.0),
            float(self.control_streak),
            float(self.possession_dwell_frames),
            float(self.flight_frames),
            self.ball_observed_fraction,
            float(self.players_near_ball),
            float(self.same_team),
            gap_secs,
            1.0 if self.recv_mode == "long" else 0.0,
            self.geometry_score,
            self.fps / 30.0,
            self.frame_width / 1280.0,
        ]

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "ball_speed_peak",
            "ball_travel_norm",
            "passer_receiver_dist_norm",
            "control_streak",
            "possession_dwell_frames",
            "flight_frames",
            "ball_observed_fraction",
            "players_near_ball",
            "same_team",
            "flight_secs",
            "is_long_receive",
            "geometry_score",
            "fps_norm",
            "frame_width_norm",
        ]


@dataclass
class GoalCandidateFeatures:
    event_type: str = "goal"
    frame: int = 0
    fps: float = 30.0
    ball_speed: float = 0.0
    cross_depth_px: float = 0.0
    last_touch_team: int = -1
    scoring_team: int = 0
    geometry_score: float = 1.0

    def to_vector(self) -> list[float]:
        return [
            self.ball_speed,
            self.cross_depth_px,
            float(max(self.last_touch_team, 0)),
            float(self.scoring_team),
            self.geometry_score,
            self.fps / 30.0,
        ]

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "ball_speed",
            "cross_depth_px",
            "last_touch_team",
            "scoring_team",
            "geometry_score",
            "fps_norm",
        ]


@dataclass
class ShotCandidateFeatures:
    event_type: str = "shot"
    frame: int = 0
    fps: float = 30.0
    ball_speed_peak: float = 0.0
    distance_to_goal_px: float = 0.0
    shooter_track_id: int = 0
    team_id: int = 0
    toward_goal: int = 1
    geometry_score: float = 1.0

    def to_vector(self) -> list[float]:
        return [
            self.ball_speed_peak,
            self.distance_to_goal_px,
            float(self.shooter_track_id > 0),
            float(self.team_id),
            float(self.toward_goal),
            self.geometry_score,
            self.fps / 30.0,
        ]

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "ball_speed_peak",
            "distance_to_goal_px",
            "has_shooter",
            "team_id",
            "toward_goal",
            "geometry_score",
            "fps_norm",
        ]


@dataclass
class CandidateContext:
    """Rolling context used when building candidate features."""

    fps: float = 30.0
    frame_width: float = 1280.0
    flight_start_frame: int = 0
    flight_observations: int = 0
    flight_total_frames: int = 0
    ball_observed_in_flight: int = 0

    def ball_observed_fraction(self) -> float:
        if self.flight_total_frames <= 0:
            return 0.0
        return self.ball_observed_in_flight / self.flight_total_frames


def count_players_near(
    players: list[dict],
    ball: tuple[float, float],
    radius_px: float,
) -> int:
    count = 0
    for p in players:
        foot = ((p["bbox"][0] + p["bbox"][2]) / 2.0, p["bbox"][3])
        if _dist(foot, ball) <= radius_px:
            count += 1
    return count
