"""
Pure-geometry football statistics engine.
Ingests per-frame detections and accumulates match statistics in real time.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans

from analysis import (
    BallInterpolator,
    GoalDetector,
    PassDetector,
    PerspectiveTransformer,
    SpeedDistanceEstimator,
    TeamColorAssigner,
)
from analysis.pitch_coordinates import PitchCoordinateMapper
from analysis.shot_detector import ShotDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _bbox_foot(bbox: list[float]) -> tuple[float, float]:
    """Bottom-center of bbox — closer to where the ball is at a player's feet."""
    return ((bbox[0] + bbox[2]) / 2.0, bbox[3])


def _goal_center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _point_in_bbox(point: tuple[float, float], box: list[float]) -> bool:
    x, y = point
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _velocity_px(
    history: deque[tuple[int, tuple[float, float]]],
) -> tuple[float, float]:
    if len(history) < 2:
        return (0.0, 0.0)
    f0, p0 = history[-2]
    f1, p1 = history[-1]
    dt = max(1, f1 - f0)
    return ((p1[0] - p0[0]) / dt, (p1[1] - p0[1]) / dt)


def _formation_string(players: list[dict], attacking_direction: str) -> str:
    if len(players) < 4:
        return "unknown"
    coords = np.array([[p["cx"], p["cy"]] for p in players], dtype=np.float32)
    axis = 0 if attacking_direction == "left_to_right" else 0
    values = coords[:, axis]
    k = min(4, len(players))
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(
        values.reshape(-1, 1)
    )
    counts = np.bincount(labels, minlength=k)
    order = np.argsort(
        [values[labels == i].mean() if np.any(labels == i) else 0 for i in range(k)]
    )
    ordered = [int(counts[i]) for i in order if counts[i] > 0]
    if len(ordered) > 3:
        ordered = ordered[-3:]
    return "-".join(str(c) for c in ordered) if ordered else "unknown"


def _compute_xg(distance_m: float, angle_rad: float) -> float:
    return float(1.0 / (1.0 + np.exp(0.5 * distance_m - 3.0 * angle_rad)))


# ---------------------------------------------------------------------------
# StatEngine
# ---------------------------------------------------------------------------

class StatEngine:
    POSSESSION_RADIUS = 100.0
    REF_WIDTH = 1280.0
    PASS_MAX_FRAMES = 90
    PASS_MIN_PLAYER_DIST = 25.0
    PASS_MIN_BALL_MOVE = 15.0
    PASS_COOLDOWN_FRAMES = 1
    LOOSE_CARRY_FRAMES = 9999
    INTERCEPTION_RADIUS = 120.0
    SHOT_VELOCITY = 25.0
    SHOT_FRAME_WINDOW = 30
    TACKLE_RADIUS = 40.0
    DRIBBLE_MIN_FRAMES = 10
    DRIBBLE_MIN_SPEED = 15.0
    DRIBBLE_TEAMMATE_RADIUS = 60.0
    PRESS_WINDOW = 150
    FORMATION_INTERVAL = 150
    SPRINT_SPEED_MPS = 7.0
    SPRINT_MIN_FRAMES = 8
    SAVE_VELOCITY = 20.0
    GK_SAVE_FRAMES = 15
    HEATMAP_BIN = 20

    def __init__(
        self,
        goal_boxes: dict,
        homography: np.ndarray,
        field_polygon: list,
        attacking_direction: str,
        fps: int = 25,
        frame_width: int = 1920,
        frame_height: int = 1080,
        halftime_frame: int | None = None,
        goal_lines: dict | None = None,
    ):
        self.goal_boxes = goal_boxes
        self.goal_lines = goal_lines
        self.homography = np.asarray(homography, dtype=np.float32)
        self.field_polygon = np.asarray(field_polygon, dtype=np.float32)
        self.attacking_direction = attacking_direction
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.halftime_frame = halftime_frame

        self.transformer = PerspectiveTransformer(self.homography)
        self.pitch = PitchCoordinateMapper(self.homography, fps=float(fps))
        self.speed_estimator = SpeedDistanceEstimator(self.transformer, fps)

        # Possession
        self._possession_frames = {"team0": 0, "team1": 0, "loose": 0}
        self._possessor_track_id: int | None = None
        self._possessor_team: int | None = None
        self._possession_start_frame = 0
        self._last_touch_frame = 0

        self.pass_detector = PassDetector(
            fps=fps,
            frame_width=frame_width,
            possession_radius=self.POSSESSION_RADIUS,
        )
        self.goal_detector = GoalDetector(
            goal_boxes=goal_boxes,
            goal_lines=goal_lines,
        )
        self.shot_detector = ShotDetector(
            pitch=self.pitch,
            attacking_direction=attacking_direction,
        )

        # Passes / interceptions / offside
        self._team0_interceptions = 0
        self._team1_interceptions = 0
        self._team0_offsides = 0
        self._team1_offsides = 0

        # Shots / xG / saves
        self._team0_shots = 0
        self._team1_shots = 0
        self._team0_xg = 0.0
        self._team1_xg = 0.0
        self._team0_saves = 0
        self._team1_saves = 0
        self._pending_shots: list[dict] = []

        # Tackles / dribbles
        self._tackles = {
            "team0": {"attempted": 0, "won": 0},
            "team1": {"attempted": 0, "won": 0},
        }
        self._pending_tackles: list[dict] = []
        self._team0_dribbles = 0
        self._team1_dribbles = 0
        self._top_dribbler_track_id: int | None = None
        self._top_dribble_frames = 0
        self._dribble_streak: dict | None = None

        # Set pieces
        self._throw_ins = 0
        self._corners = 0
        self._goal_kicks = 0
        self._last_touch_team: int | None = None
        self._ball_was_in_field = True

        # Heatmaps / sprints
        self._heatmap_positions: dict[int, list[tuple[float, float]]] = {}
        self._sprint_state: dict[int, dict] = {}
        self._sprint_stats: dict[int, dict] = {}

        # Formation
        self._formation_history: list[dict] = []

        # Pressing
        self._press_buffer: deque[dict] = deque(maxlen=self.PRESS_WINDOW)
        self._team0_press_values: deque[float] = deque(maxlen=self.PRESS_WINDOW)
        self._team1_press_values: deque[float] = deque(maxlen=self.PRESS_WINDOW)

        # Ball tracking
        self._ball_history: deque[tuple[int, tuple[float, float]]] = deque(maxlen=10)
        self._ball_velocity_smooth: deque[tuple[float, float]] = deque(maxlen=5)

        # Per-frame CSV log
        self._frame_log: list[dict] = []

    # ------------------------------------------------------------------
    def set_halftime(self, frame_idx: int):
        if self.halftime_frame is None:
            self.halftime_frame = frame_idx
        if self.attacking_direction == "left_to_right":
            self.attacking_direction = "right_to_left"
        else:
            self.attacking_direction = "left_to_right"

    def _players(self, detections: list[dict]) -> list[dict]:
        skip = {"ball", "referee", "linesman", "goalpost", "score", "team_name", "gametime", "playername"}
        return [d for d in detections if d.get("class_name") not in skip and d.get("team_id") is not None]

    def _px(self, value_at_ref: float) -> float:
        return value_at_ref * self.frame_width / self.REF_WIDTH

    def _closest_player_to_ball(
        self, players: list[dict], ball: tuple[float, float]
    ) -> dict | None:
        best, best_dist = None, float("inf")
        for p in players:
            foot = _bbox_foot(p["bbox"])
            dist = _euclidean(foot, ball)
            if dist < best_dist:
                best_dist = dist
                best = p
        if best is not None and best_dist <= self._px(self.POSSESSION_RADIUS):
            return best
        return None

    def _ball_speed(self) -> float:
        vel = _velocity_px(self._ball_history)
        return float(np.linalg.norm(vel))

    def _update_possession(self, frame_idx: int, players: list[dict], ball: tuple[float, float] | None):
        if ball is None:
            if self._last_touch_team is not None:
                self._possession_frames[f"team{self._last_touch_team}"] += 1
            else:
                self._possession_frames["loose"] += 1
            return

        owner = self._closest_player_to_ball(players, ball)
        prev_track = self._possessor_track_id
        prev_team = self._possessor_team

        if owner is None:
            if self._last_touch_team is not None:
                self._possession_frames[f"team{self._last_touch_team}"] += 1
            else:
                self._possession_frames["loose"] += 1
            self._check_dribble_end(frame_idx, False)
            self._possessor_track_id = None
            self._possessor_team = None
            return

        team = owner["team_id"]
        track_id = owner["track_id"]
        key = f"team{team}"
        self._possession_frames[key] += 1
        self._last_touch_team = team
        self._last_touch_frame = frame_idx

        dwell_frames = frame_idx - self._possession_start_frame if prev_track is not None else 0

        if prev_track is not None and prev_track != track_id:
            self._handle_possession_switch(
                frame_idx, prev_track, prev_team, owner, ball, dwell_frames
            )
            self._possession_start_frame = frame_idx
        elif prev_track is None:
            self._possession_start_frame = frame_idx

        if prev_track == track_id:
            self._update_dribble_streak(frame_idx, owner, players)
        else:
            self._check_dribble_end(frame_idx, False)
            self._dribble_streak = {
                "track_id": track_id,
                "team": team,
                "start": frame_idx,
                "positions": [(owner["cx"], owner["cy"])],
            }

        self._possessor_track_id = track_id
        self._possessor_team = team

    def _handle_possession_switch(
        self,
        frame_idx: int,
        prev_track: int,
        prev_team: int | None,
        new_owner: dict,
        ball: tuple[float, float],
        dwell_frames: int,
    ):
        if prev_team is None:
            return
        new_team = new_owner["team_id"]
        new_track = new_owner["track_id"]

        if prev_team == new_team and prev_track != new_track:
            pass  # passes detected via ball-flight receive in PassDetector.update

        if prev_team != new_team:
            dist = _euclidean((new_owner["cx"], new_owner["cy"]), ball)
            if dist <= self._px(self.INTERCEPTION_RADIUS):
                if new_team == 0:
                    self._team0_interceptions += 1
                else:
                    self._team1_interceptions += 1

    def _get_last_position(self, track_id: int) -> tuple[float, float] | None:
        positions = self._heatmap_positions.get(track_id)
        if positions:
            return positions[-1]
        return None

    def _check_offside_at_pass(
        self,
        frame_idx: int,
        attacking_team: int,
        ball: tuple[float, float],
        receiver: dict,
    ):
        defenders = [
            p for p in self._current_players_cache
            if p.get("team_id") != attacking_team and not p.get("is_goalkeeper")
        ]
        attackers = [
            p for p in self._current_players_cache
            if p.get("team_id") == attacking_team and not p.get("is_goalkeeper")
        ]
        if len(defenders) < 2:
            return
        defenders_sorted = sorted(defenders, key=lambda p: p["cx"])
        second_last_x = defenders_sorted[-2]["cx"]
        ball_x = ball[0]
        for a in attackers:
            if a["track_id"] == receiver["track_id"]:
                continue
            ahead = a["cx"] > ball_x if self.attacking_direction == "left_to_right" else a["cx"] < ball_x
            beyond = a["cx"] > second_last_x if self.attacking_direction == "left_to_right" else a["cx"] < second_last_x
            if ahead and beyond:
                if attacking_team == 0:
                    self._team0_offsides += 1
                else:
                    self._team1_offsides += 1
                break

    def _update_ball_events(self, frame_idx: int, ball: tuple[float, float] | None):
        if ball is None:
            return
        vel = _velocity_px(self._ball_history)
        self._ball_velocity_smooth.append(vel)
        speed = float(np.linalg.norm(vel))

        for goal_team, box in self.goal_boxes.items():
            attacking = 0 if goal_team == "team1" else 1
            goal_c = _goal_center(box)
            toward = (goal_c[0] - ball[0], goal_c[1] - ball[1])
            dot = toward[0] * vel[0] + toward[1] * vel[1]
            if speed > self.SHOT_VELOCITY and dot > 0:
                self._pending_shots.append(
                    {
                        "frame": frame_idx,
                        "team": attacking,
                        "ball": ball,
                        "goal_box": box,
                        "speed": speed,
                    }
                )

        self._pending_shots = [s for s in self._pending_shots if frame_idx - s["frame"] <= self.SHOT_FRAME_WINDOW]
        for shot in list(self._pending_shots):
            if _point_in_bbox(ball, shot["goal_box"]):
                if shot["team"] == 0:
                    self._team0_shots += 1
                else:
                    self._team1_shots += 1
                dist_m = self.transformer.distance_meters(shot["ball"], _goal_center(shot["goal_box"]))
                angle = float(np.arctan2(box_width(shot["goal_box"]), max(dist_m, 0.1)))
                xg = _compute_xg(dist_m, angle)
                if shot["team"] == 0:
                    self._team0_xg += xg
                else:
                    self._team1_xg += xg
                self._check_save(frame_idx, shot, speed)
                self._pending_shots.remove(shot)

    def _check_save(self, frame_idx: int, shot: dict, speed: float):
        if speed < self.SAVE_VELOCITY:
            return
        defending = 1 - shot["team"]
        if self._possessor_team == defending and frame_idx - shot["frame"] <= self.GK_SAVE_FRAMES:
            if defending == 0:
                self._team0_saves += 1
            else:
                self._team1_saves += 1

    def _update_tackles(self, frame_idx: int, players: list[dict], ball: tuple[float, float] | None):
        if ball is None:
            return
        near = [p for p in players if _euclidean((p["cx"], p["cy"]), ball) <= self.TACKLE_RADIUS]
        teams = {p["team_id"] for p in near}
        if len(teams) < 2:
            return
        prior = self._possessor_track_id
        if prior is None:
            return
        tacklers = [p for p in near if p["track_id"] != prior]
        if not tacklers:
            return
        for t in tacklers:
            if t["team_id"] == self._possessor_team:
                continue
            key = f"team{t['team_id']}"
            self._tackles[key]["attempted"] += 1
            self._pending_tackles.append(
                {
                    "frame": frame_idx,
                    "tackler_team": t["team_id"],
                    "defending_team": self._possessor_team,
                }
            )

        self._pending_tackles = [
            t for t in self._pending_tackles if frame_idx - t["frame"] <= 10
        ]
        for t in list(self._pending_tackles):
            if frame_idx - t["frame"] >= 10:
                if self._possessor_team == t["tackler_team"]:
                    self._tackles[f"team{t['tackler_team']}"]["won"] += 1
                self._pending_tackles.remove(t)

    def _update_dribble_streak(self, frame_idx: int, owner: dict, players: list[dict]):
        if not self._dribble_streak:
            return
        self._dribble_streak["positions"].append((owner["cx"], owner["cy"]))
        frames = frame_idx - self._dribble_streak["start"] + 1
        if frames < self.DRIBBLE_MIN_FRAMES:
            return
        positions = self._dribble_streak["positions"]
        total_dist = sum(
            _euclidean(positions[i], positions[i + 1]) for i in range(len(positions) - 1)
        )
        avg_speed = total_dist / max(1, frames)
        teammates_near = any(
            _euclidean((p["cx"], p["cy"]), (owner["cx"], owner["cy"])) < self.DRIBBLE_TEAMMATE_RADIUS
            for p in players
            if p["team_id"] == owner["team_id"] and p["track_id"] != owner["track_id"]
        )
        if avg_speed >= self.DRIBBLE_MIN_SPEED and not teammates_near:
            team = self._dribble_streak["team"]
            if team == 0:
                self._team0_dribbles += 1
            else:
                self._team1_dribbles += 1
            if frames > self._top_dribble_frames:
                self._top_dribble_frames = frames
                self._top_dribbler_track_id = owner["track_id"]
            self._dribble_streak = None

    def _check_dribble_end(self, frame_idx: int, _success: bool):
        self._dribble_streak = None

    def _update_set_pieces(self, ball: tuple[float, float] | None):
        if ball is None:
            return
        inside = cv2.pointPolygonTest(self.field_polygon, ball, False) >= 0
        if self._ball_was_in_field and not inside:
            x, y = ball
            poly = self.field_polygon
            min_x, max_x = poly[:, 0].min(), poly[:, 0].max()
            min_y, max_y = poly[:, 1].min(), poly[:, 1].max()
            if x <= min_x or x >= max_x:
                self._throw_ins += 1
            elif y <= min_y:
                if self._last_touch_team == 0:
                    self._corners += 1
                else:
                    self._goal_kicks += 1
            elif y >= max_y:
                if self._last_touch_team == 1:
                    self._corners += 1
                else:
                    self._goal_kicks += 1
        self._ball_was_in_field = inside

    def _update_pressing(self, players: list[dict]):
        if self._possessor_track_id is None or self._possessor_team is None:
            return
        carrier = next((p for p in players if p["track_id"] == self._possessor_track_id), None)
        if carrier is None:
            return
        cpos = (carrier["cx"], carrier["cy"])
        for team in (0, 1):
            if team == self._possessor_team:
                continue
            opp = [p for p in players if p["team_id"] == team]
            if not opp:
                continue
            avg_dist = float(np.mean([_euclidean((p["cx"], p["cy"]), cpos) for p in opp]))
            if team == 0:
                self._team0_press_values.append(avg_dist)
            else:
                self._team1_press_values.append(avg_dist)

    def _update_formation(self, frame_idx: int, players: list[dict]):
        if frame_idx % self.FORMATION_INTERVAL != 0:
            return
        snapshot = {"frame_idx": frame_idx}
        for team in (0, 1):
            outfield = [
                p for p in players
                if p["team_id"] == team and not p.get("is_goalkeeper")
            ]
            snapshot[f"team{team}_formation"] = _formation_string(
                outfield, self.attacking_direction
            )
        self._formation_history.append(snapshot)

    def _update_sprints(self, track_id: int, center: tuple[float, float]):
        speed = self.speed_estimator.speeds.get(track_id, 0.0)
        state = self._sprint_state.setdefault(
            track_id, {"frames": 0, "distance": 0.0, "last": center}
        )
        if speed >= self.SPRINT_SPEED_MPS:
            state["frames"] += 1
            state["distance"] += self.transformer.distance_meters(state["last"], center)
        else:
            if state["frames"] >= self.SPRINT_MIN_FRAMES:
                stats = self._sprint_stats.setdefault(
                    track_id, {"sprint_count": 0, "sprint_distance_m": 0.0}
                )
                stats["sprint_count"] += 1
                stats["sprint_distance_m"] += state["distance"]
            state["frames"] = 0
            state["distance"] = 0.0
        state["last"] = center

    def update(
        self,
        frame_idx: int,
        detections: list[dict],
        ball: tuple[float, float] | None,
        ball_observed: bool = True,
    ):
        if self.halftime_frame is not None and frame_idx == self.halftime_frame:
            self.set_halftime(frame_idx)

        if ball is not None:
            self._ball_history.append((frame_idx, ball))

        self._current_players_cache = self._players(detections)
        players = self._current_players_cache

        for p in players:
            tid = p["track_id"]
            center = (p["cx"], p["cy"])
            self._heatmap_positions.setdefault(tid, []).append(center)
            self.speed_estimator.update(tid, center)
            self._update_sprints(tid, center)

        self._update_possession(frame_idx, players, ball)
        speed = self._ball_speed() if ball is not None else 0.0
        speed_mps = 0.0
        vx_sign = 0.0
        if ball is not None and len(self._ball_history) >= 2:
            _, p0 = self._ball_history[-2]
            _, p1 = self._ball_history[-1]
            speed_mps = self.pitch.speed_mps(p0, p1, frame_gap=1)
            vx_sign = p1[0] - p0[0]

        self.pass_detector.update(frame_idx, ball, speed, players, ball_observed=ball_observed)
        self.goal_detector.update(frame_idx, ball, speed, self._last_touch_team)
        self.shot_detector.update(
            frame_idx,
            ball,
            speed_mps,
            vx_sign,
            self._possessor_track_id,
            self._possessor_team,
            ball_observed=ball_observed,
        )
        self._update_ball_events(frame_idx, ball)
        self._update_tackles(frame_idx, players, ball)
        self._update_set_pieces(ball)
        self._update_pressing(players)
        self._update_formation(frame_idx, players)

        total = sum(self._possession_frames.values()) or 1
        ball_m = self.pitch.to_meters(ball) if ball is not None else (None, None)
        possessor_m = None
        if self._possessor_track_id is not None:
            carrier = next((p for p in players if p["track_id"] == self._possessor_track_id), None)
            if carrier is not None:
                possessor_m = self.pitch.to_meters(self.pitch.foot_px(carrier["bbox"]))

        self._frame_log.append(
            {
                "frame_idx": frame_idx,
                "team0_possession_pct": self._possession_frames["team0"] / total * 100,
                "team1_possession_pct": self._possession_frames["team1"] / total * 100,
                "loose_pct": self._possession_frames["loose"] / total * 100,
                "team0_passes": self.pass_detector.counts()["team0_passes"],
                "team1_passes": self.pass_detector.counts()["team1_passes"],
                "possessor_track_id": self._possessor_track_id,
                "ball_x_m": ball_m[0] if ball is not None else None,
                "ball_y_m": ball_m[1] if ball is not None else None,
                "possessor_x_m": possessor_m[0] if possessor_m else None,
                "possessor_y_m": possessor_m[1] if possessor_m else None,
            }
        )

    def _possession_pct(self) -> dict:
        total = sum(self._possession_frames.values()) or 1
        result = {
            "team0_pct": self._possession_frames["team0"] / total * 100,
            "team1_pct": self._possession_frames["team1"] / total * 100,
            "loose_pct": self._possession_frames["loose"] / total * 100,
        }
        # Single-team clips: unattributed loose ball -> sole possessing team
        if result["team1_pct"] == 0.0 and result["loose_pct"] > 0.0:
            result["team0_pct"] += result["loose_pct"]
            result["loose_pct"] = 0.0
        elif result["team0_pct"] == 0.0 and result["loose_pct"] > 0.0:
            result["team1_pct"] += result["loose_pct"]
            result["loose_pct"] = 0.0
        return result

    def get_heatmap(self, track_id: int) -> np.ndarray:
        positions = self._heatmap_positions.get(track_id, [])
        h_bins = max(1, self.frame_height // self.HEATMAP_BIN)
        w_bins = max(1, self.frame_width // self.HEATMAP_BIN)
        heatmap = np.zeros((h_bins, w_bins), dtype=np.float32)
        for x, y in positions:
            bx = min(w_bins - 1, int(x // self.HEATMAP_BIN))
            by = min(h_bins - 1, int(y // self.HEATMAP_BIN))
            heatmap[by, bx] += 1
        return heatmap

    def get_formation_history(self) -> list:
        return list(self._formation_history)

    def get_stats(self) -> dict:
        press0 = float(np.mean(self._team0_press_values)) if self._team0_press_values else 0.0
        press1 = float(np.mean(self._team1_press_values)) if self._team1_press_values else 0.0
        heatmaps = {
            tid: self.get_heatmap(tid).tolist()
            for tid in self._heatmap_positions
        }
        pass_counts = self.pass_detector.counts()
        goal_counts = self.goal_detector.counts()
        shot_counts = self.shot_detector.counts()

        pass_events = []
        for e in self.pass_detector.events:
            dist_m = 0.0
            speed_mps = 0.0
            if e.release_ball_px and e.receive_ball_px:
                dist_m = self.pitch.distance_m(e.release_ball_px, e.receive_ball_px)
                gap = max(1, e.frame - e.release_frame)
                speed_mps = dist_m * self.fps / gap
            pass_events.append(
                {
                    "frame": e.frame,
                    "release_frame": e.release_frame,
                    "team_id": e.team_id,
                    "from_track_id": e.from_track_id,
                    "to_track_id": e.to_track_id,
                    "ball_speed_peak_px": e.ball_speed_peak,
                    "distance_m": round(dist_m, 2),
                    "speed_mps": round(speed_mps, 2),
                    "release_ball_m": list(self.pitch.to_meters(e.release_ball_px))
                    if e.release_ball_px
                    else None,
                    "receive_ball_m": list(self.pitch.to_meters(e.receive_ball_px))
                    if e.receive_ball_px
                    else None,
                }
            )

        return {
            "possession": self._possession_pct(),
            "passes": pass_counts,
            "pass_events": pass_events,
            "goals": goal_counts,
            "goal_events": [
                {
                    "frame": e.frame,
                    "scoring_team": e.scoring_team,
                    "defended_goal": e.defended_goal,
                    "ball_speed": e.ball_speed,
                }
                for e in self.goal_detector.events
            ],
            "interceptions": {
                "team0_interceptions": self._team0_interceptions,
                "team1_interceptions": self._team1_interceptions,
            },
            "shots": {
                "team0_shots": shot_counts["team0_shots"],
                "team1_shots": shot_counts["team1_shots"],
            },
            "shots_on_target": {
                "team0": shot_counts["team0_shots_on_target"],
                "team1": shot_counts["team1_shots_on_target"],
            },
            "shot_events": [
                {
                    "frame": e.frame,
                    "team_id": e.team_id,
                    "shooter_track_id": e.shooter_track_id,
                    "ball_speed_mps": round(e.ball_speed_mps, 2),
                    "distance_to_goal_m": round(e.distance_to_goal_m, 2),
                    "on_target": e.on_target,
                    "xg": round(e.xg, 3),
                    "ball_m": [round(e.ball_x_m, 2), round(e.ball_y_m, 2)],
                }
                for e in self.shot_detector.events
            ],
            "heatmaps": heatmaps,
            "distance_covered": self.speed_estimator.get_distances(),
            "formations": self.get_formation_history(),
            "tackles": {
                "team0_tackles_attempted": self._tackles["team0"]["attempted"],
                "team0_tackles_won": self._tackles["team0"]["won"],
                "team1_tackles_attempted": self._tackles["team1"]["attempted"],
                "team1_tackles_won": self._tackles["team1"]["won"],
            },
            "dribbles": {
                "team0_dribbles": self._team0_dribbles,
                "team1_dribbles": self._team1_dribbles,
                "top_dribbler_track_id": self._top_dribbler_track_id,
            },
            "set_pieces": {
                "throw_ins": self._throw_ins,
                "corners": self._corners,
                "goal_kicks": self._goal_kicks,
            },
            "pressing": {"team0_press_index": press0, "team1_press_index": press1},
            "offsides": {"team0_offsides": self._team0_offsides, "team1_offsides": self._team1_offsides},
            "sprints": self._sprint_stats,
            "saves": {"team0_saves": self._team0_saves, "team1_saves": self._team1_saves},
            "xg": {"team0_xG": self._team0_xg, "team1_xG": self._team1_xg},
        }

    def export_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.get_stats(), f, indent=2)

    def export_csv(self, path: str):
        if not self._frame_log:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._frame_log[0].keys())
            writer.writeheader()
            writer.writerows(self._frame_log)


def box_width(box: list[float]) -> float:
    return abs(box[2] - box[0])


def default_homography(frame_width: int, frame_height: int) -> np.ndarray:
    """Identity-ish mapping: scale pixels to ~105m x 68m pitch."""
    src = np.float32([
        [0, 0],
        [frame_width, 0],
        [frame_width, frame_height],
        [0, frame_height],
    ])
    dst = np.float32([[0, 0], [105, 0], [105, 68], [0, 68]])
    return cv2.getPerspectiveTransform(src, dst)


def default_field_polygon(frame_width: int, frame_height: int) -> list:
    margin = 50
    return [
        (margin, margin),
        (frame_width - margin, margin),
        (frame_width - margin, frame_height - margin),
        (margin, frame_height - margin),
    ]


def default_goal_boxes(frame_width: int, frame_height: int) -> dict:
    gw = frame_width * 0.08
    gh = frame_height * 0.25
    cy = frame_height / 2
    return {
        "team0": [0, cy - gh / 2, gw, cy + gh / 2],
        "team1": [frame_width - gw, cy - gh / 2, frame_width, cy + gh / 2],
    }


def run_pipeline_on_video(
    video_path: str,
    model_path: str = "models/best.pt",
    max_frames: int | None = None,
    halftime_frame: int | None = None,
) -> StatEngine:
    from analysis.camera_movement import CameraMovementEstimator
    from trackers import Tracker
    from utils import read_video

    frames = read_video(video_path)
    if max_frames:
        frames = frames[:max_frames]
    h, w = frames[0].shape[:2]

    tracker = Tracker(model_path)
    color_assigner = TeamColorAssigner()
    ball_interpolator = BallInterpolator(fps=25)
    camera_estimator = CameraMovementEstimator()

    engine = StatEngine(
        goal_boxes=default_goal_boxes(w, h),
        homography=default_homography(w, h),
        field_polygon=default_field_polygon(w, h),
        attacking_direction="left_to_right",
        fps=25,
        frame_width=w,
        frame_height=h,
        halftime_frame=halftime_frame,
    )

    frame_records = tracker.get_object_tracks(frames)
    for record in frame_records:
        idx = record["frame_idx"]
        frame = record["frame"]
        camera_estimator.estimate(frame)
        detections = color_assigner.assign_teams(frame, record["detections"])
        ball_obs = ball_interpolator.update(idx, record["ball"])
        engine.update(idx, detections, ball_obs.position, ball_observed=ball_obs.observed)
        if idx > 0 and idx % StatEngine.FORMATION_INTERVAL == 0:
            stats = engine.get_stats()
            print_dashboard(stats, idx)

    return engine


def print_dashboard(stats: dict, frame_idx: int):
    print(f"\n=== Stats @ frame {frame_idx} ===")
    print(f"Possession: {stats['possession']}")
    print(f"Passes: {stats['passes']}")
    print(f"Shots: {stats['shots']}  xG: {stats['xg']}")
    print(f"Interceptions: {stats['interceptions']}")
    print("=" * 40)


def load_detections_json(path: str) -> tuple[list[dict], dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    meta = {k: data[k] for k in ("fps", "frame_width", "frame_height") if k in data}
    return data["frames"], meta


def main():
    parser = argparse.ArgumentParser(description="Football statistics engine")
    parser.add_argument("--input", help="Path to detections JSON")
    parser.add_argument("--homography", help="Path to homography .npy file")
    parser.add_argument("--output", help="Output stats JSON path")
    parser.add_argument("--csv", help="Output per-frame CSV path")
    parser.add_argument("--live", action="store_true", help="Run live on video")
    parser.add_argument("--source", default="input_videos/elclasico.mp4")
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--halftime-frame", type=int, default=None)
    args = parser.parse_args()

    if args.live or (not args.input):
        engine = run_pipeline_on_video(
            args.source,
            model_path=args.model,
            max_frames=args.max_frames,
            halftime_frame=args.halftime_frame,
        )
    else:
        frames_data, meta = load_detections_json(args.input)
        w = meta.get("frame_width", 1920)
        h = meta.get("frame_height", 1080)
        homography = (
            np.load(args.homography)
            if args.homography
            else default_homography(w, h)
        )
        engine = StatEngine(
            goal_boxes=default_goal_boxes(w, h),
            homography=homography,
            field_polygon=default_field_polygon(w, h),
            attacking_direction="left_to_right",
            fps=meta.get("fps", 25),
            frame_width=w,
            frame_height=h,
            halftime_frame=args.halftime_frame,
        )
        ball_interpolator = BallInterpolator(fps=meta.get("fps", 25))
        for record in frames_data:
            idx = record["frame_idx"]
            ball_obs = ball_interpolator.update(
                idx, tuple(record["ball"]) if record.get("ball") else None
            )
            engine.update(idx, record["detections"], ball_obs.position, ball_observed=ball_obs.observed)

    if args.output:
        engine.export_json(args.output)
        print(f"Wrote {args.output}")
    if args.csv:
        engine.export_csv(args.csv)
        print(f"Wrote {args.csv}")
    if not args.output:
        print(json.dumps(engine.get_stats(), indent=2))


if __name__ == "__main__":
    main()
