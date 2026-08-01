"""Detect goals when the ball crosses the goal mouth derived from goalpost boxes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _point_in_box(point: tuple[float, float], box: list[float]) -> bool:
    x, y = point
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


@dataclass
class GoalEvent:
    frame: int
    scoring_team: int
    defended_goal: str
    ball_speed: float


@dataclass
class GoalDetector:
    goal_boxes: dict[str, list[float]]
    goal_lines: dict[str, dict] | None = None
    min_cross_speed: float = 12.0
    cooldown_frames: int = 60

    events: list[GoalEvent] = field(default_factory=list)
    _prev_ball: tuple[float, float] | None = None
    _inside: dict[str, bool] = field(default_factory=dict)
    _last_goal_frame: int = -999

    def __post_init__(self) -> None:
        for side in self.goal_boxes:
            self._inside[side] = False

    def _crossed_mouth(
        self,
        prev: tuple[float, float],
        curr: tuple[float, float],
        side: str,
    ) -> bool:
        if self.goal_lines and side in self.goal_lines:
            line = self.goal_lines[side]
            mouth_x = line["mouth_x"]
            y1, y2 = line["y1"], line["y2"]
            if not (min(y1, y2) <= curr[1] <= max(y1, y2)):
                return False
            if side == "team0":
                return prev[0] > mouth_x >= curr[0]
            return prev[0] < mouth_x <= curr[0]

        box = self.goal_boxes[side]
        was_out = not _point_in_box(prev, box)
        now_in = _point_in_box(curr, box)
        return was_out and now_in

    def update(
        self,
        frame_idx: int,
        ball: tuple[float, float] | None,
        ball_speed: float,
        last_touch_team: int | None,
    ) -> GoalEvent | None:
        if ball is None:
            self._prev_ball = None
            return None

        if self._prev_ball is None:
            self._prev_ball = ball
            for side, box in self.goal_boxes.items():
                self._inside[side] = _point_in_box(ball, box)
            return None

        prev = self._prev_ball
        self._prev_ball = ball

        if ball_speed < self.min_cross_speed:
            return None
        if frame_idx - self._last_goal_frame < self.cooldown_frames:
            return None

        for defended_goal in self.goal_boxes:
            scoring_team = 1 if defended_goal == "team0" else 0
            if not self._crossed_mouth(prev, ball, defended_goal):
                continue
            event = GoalEvent(
                frame=frame_idx,
                scoring_team=scoring_team,
                defended_goal=defended_goal,
                ball_speed=ball_speed,
            )
            self.events.append(event)
            self._last_goal_frame = frame_idx
            self._inside[defended_goal] = True
            return event

        for side, box in self.goal_boxes.items():
            self._inside[side] = _point_in_box(ball, box)
        return None

    def counts(self) -> dict[str, int]:
        return {
            "team0_goals": sum(1 for e in self.events if e.scoring_team == 0),
            "team1_goals": sum(1 for e in self.events if e.scoring_team == 1),
        }
