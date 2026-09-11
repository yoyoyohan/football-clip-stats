"""Detect goals when the ball crosses the goal mouth derived from goalpost boxes."""

from __future__ import annotations

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
    # Box-based crosses are noisier; keep a higher bar. Line crosses can be softer.
    min_cross_speed: float = 8.0
    unambiguous_min_cross_speed: float = 5.0
    cooldown_frames: int = 60
    # Optional one-frame hold after an unambiguous line cross before committing.
    post_cross_confirm_frames: int = 0

    events: list[GoalEvent] = field(default_factory=list)
    _prev_ball: tuple[float, float] | None = None
    _inside: dict[str, bool] = field(default_factory=dict)
    _last_goal_frame: int = -999
    _pending_cross: dict | None = None

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

    def _speed_needed(self, side: str) -> float:
        if self.goal_lines and side in self.goal_lines:
            return self.unambiguous_min_cross_speed
        return self.min_cross_speed

    def _emit(
        self,
        frame_idx: int,
        scoring_team: int,
        defended_goal: str,
        ball_speed: float,
    ) -> GoalEvent:
        event = GoalEvent(
            frame=frame_idx,
            scoring_team=scoring_team,
            defended_goal=defended_goal,
            ball_speed=ball_speed,
        )
        self.events.append(event)
        self._last_goal_frame = frame_idx
        self._inside[defended_goal] = True
        self._pending_cross = None
        return event

    def update(
        self,
        frame_idx: int,
        ball: tuple[float, float] | None,
        ball_speed: float,
        last_touch_team: int | None,
    ) -> GoalEvent | None:
        if ball is None:
            self._prev_ball = None
            self._pending_cross = None
            return None

        if self._prev_ball is None:
            self._prev_ball = ball
            for side, box in self.goal_boxes.items():
                self._inside[side] = _point_in_box(ball, box)
            return None

        prev = self._prev_ball
        self._prev_ball = ball

        if frame_idx - self._last_goal_frame < self.cooldown_frames:
            self._pending_cross = None
            return None

        # Short post-cross confirmation for unambiguous line crosses.
        if self._pending_cross is not None:
            pending = self._pending_cross
            side = pending["defended_goal"]
            if self.goal_lines and side in self.goal_lines:
                line = self.goal_lines[side]
                y1, y2 = line["y1"], line["y2"]
                still_in_mouth_y = min(y1, y2) <= ball[1] <= max(y1, y2)
                crossed_deeper = (
                    (side == "team0" and ball[0] <= line["mouth_x"])
                    or (side == "team1" and ball[0] >= line["mouth_x"])
                )
                if still_in_mouth_y and crossed_deeper:
                    pending["confirm"] += 1
                    if pending["confirm"] >= self.post_cross_confirm_frames:
                        return self._emit(
                            pending["frame"],
                            pending["scoring_team"],
                            side,
                            pending["ball_speed"],
                        )
                    return None
            self._pending_cross = None

        for defended_goal in self.goal_boxes:
            scoring_team = 1 if defended_goal == "team0" else 0
            if not self._crossed_mouth(prev, ball, defended_goal):
                continue
            if ball_speed < self._speed_needed(defended_goal):
                continue

            using_lines = bool(self.goal_lines and defended_goal in self.goal_lines)
            if using_lines and self.post_cross_confirm_frames > 0:
                self._pending_cross = {
                    "frame": frame_idx,
                    "scoring_team": scoring_team,
                    "defended_goal": defended_goal,
                    "ball_speed": ball_speed,
                    "confirm": 0,
                }
                return None

            return self._emit(frame_idx, scoring_team, defended_goal, ball_speed)

        for side, box in self.goal_boxes.items():
            self._inside[side] = _point_in_box(ball, box)
        return None

    def counts(self) -> dict[str, int]:
        return {
            "team0_goals": sum(1 for e in self.events if e.scoring_team == 0),
            "team1_goals": sum(1 for e in self.events if e.scoring_team == 1),
        }
