"""Evaluation overlay: ground ovals at players' feet, ball boxes, HUD.

Player bounding boxes stay internal for tracking. Rendering draws a
flattened ellipse on the pitch around the foot point.
"""

from __future__ import annotations

import cv2
import numpy as np

from utils.detection_utils import normalize_ball


TEAM_BGR = {
    0: (40, 180, 255),
    1: (255, 140, 40),
}
UNKNOWN_BGR = (200, 200, 200)
REFEREE_BGR = (0, 220, 255)
BALL_BGR = (60, 255, 80)
GOALPOST_BGR = (255, 0, 255)
POSSESSOR_RING = (255, 255, 255)


def player_foot_point(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, float(y2))


def player_body_center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def player_circle_radius(bbox: list[float]) -> int:
    """Horizontal radius of the ground oval (kept for older call sites)."""
    return player_ground_axes(bbox)[0]


def player_ground_axes(bbox: list[float]) -> tuple[int, int]:
    """(rx, ry) for a pitch oval centered on the player's feet."""
    width = max(1.0, bbox[2] - bbox[0])
    rx = int(max(12, min(48, round(width * 0.9))))
    ry = int(max(5, min(18, round(rx * 0.38))))
    return rx, ry


def format_confidence(confidence: float | None) -> str | None:
    if confidence is None:
        return None
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    if value > 1.5:
        value = value / 100.0
    return f"{value:.2f}"


def player_label(detection: dict) -> str:
    track_id = detection.get("track_id")
    conf = format_confidence(detection.get("confidence"))
    parts: list[str] = []
    if track_id is not None:
        parts.append(str(track_id))
    if conf is not None:
        parts.append(conf)
    return "  ".join(parts)


def _label_dy(foot_points: list[tuple[int, float, float]]) -> dict[int, int]:
    """Stagger labels when two foot-points sit on top of each other."""
    offsets: dict[int, int] = {}
    ordered = sorted(foot_points, key=lambda item: (item[1], item[2]))
    for i, (track_id, cx, by) in enumerate(ordered):
        dy = -16
        if i > 0:
            _pid, px, py = ordered[i - 1]
            if abs(cx - px) < 36 and abs(by - py) < 28:
                dy = 20
        offsets[track_id] = dy
    return offsets


def _team_color(detection: dict) -> tuple[int, int, int]:
    name = detection.get("class_name")
    if name == "referee":
        return REFEREE_BGR
    team = detection.get("team_id")
    return TEAM_BGR.get(team, UNKNOWN_BGR)


def draw_player_circle(
    frame: np.ndarray,
    detection: dict,
    *,
    possessor: bool = False,
    label_dy: int = -16,
) -> None:
    bbox = detection.get("bbox")
    if not bbox or len(bbox) < 4:
        return
    cx, by = player_foot_point(bbox)
    center = (int(round(cx)), int(round(by)))
    rx, ry = player_ground_axes(bbox)
    color = _team_color(detection)
    thickness = 3 if possessor else 2
    cv2.ellipse(frame, center, (rx, ry), 0, 0, 360, color, thickness, lineType=cv2.LINE_AA)
    if possessor:
        cv2.ellipse(
            frame,
            center,
            (rx + 4, ry + 3),
            0,
            0,
            360,
            POSSESSOR_RING,
            2,
            lineType=cv2.LINE_AA,
        )

    text = player_label(detection)
    if not text:
        return
    tx = center[0] - 16
    ty = max(16, center[1] - ry + label_dy)
    cv2.putText(
        frame,
        text,
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def _dashed_rect(
    frame: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash: int = 7,
) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    for x in range(x1, max(x1 + 1, x2), dash * 2):
        cv2.line(frame, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(frame, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, max(y1 + 1, y2), dash * 2):
        cv2.line(frame, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(frame, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def draw_ball_box(frame: np.ndarray, ball, *, interpolated: bool = False) -> None:
    if ball is None:
        return
    bbox = None
    confidence = None
    if isinstance(ball, dict):
        bbox = ball.get("bbox")
        confidence = ball.get("confidence")
        if bbox is None and ball.get("position"):
            x, y = ball["position"]
            bbox = [x - 8, y - 8, x + 8, y + 8]
    else:
        pos, confidence, _area = normalize_ball(ball)
        if pos is None:
            return
        bbox = [pos[0] - 8, pos[1] - 8, pos[0] + 8, pos[1] + 8]
    if not bbox:
        return
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    if interpolated:
        _dashed_rect(frame, (x1, y1), (x2, y2), BALL_BGR, 2)
        label = "ball track"
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), BALL_BGR, 2)
        conf = format_confidence(confidence)
        label = f"ball {conf}" if conf else "ball"
    cv2.putText(
        frame,
        label,
        (x1, max(16, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        BALL_BGR,
        1,
        cv2.LINE_AA,
    )


def draw_goalpost(frame: np.ndarray, detection: dict) -> None:
    bbox = detection.get("bbox")
    if not bbox or len(bbox) < 4:
        return
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), GOALPOST_BGR, 2)
    conf = format_confidence(detection.get("confidence"))
    label = f"goalpost {conf}" if conf else "goalpost"
    cv2.putText(
        frame,
        label,
        (x1, max(16, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        GOALPOST_BGR,
        1,
        cv2.LINE_AA,
    )


def draw_hud(
    frame: np.ndarray,
    stats: dict | None,
    *,
    possessor_track_id: int | None = None,
    possessor_team: int | None = None,
    team_names: dict | None = None,
) -> None:
    names = team_names or {"team0": "Team A", "team1": "Team B"}
    poss = (stats or {}).get("possession") or {}
    passes = (stats or {}).get("passes") or {}
    shots = (stats or {}).get("shots") or {}
    goals = (stats or {}).get("goals") or {}
    lines = [
        "VIDEO ANALYSIS",
        f"{names.get('team0', 'Team A')}: poss {poss.get('team0_pct', 0):.0f}%  "
        f"P {passes.get('team0_passes', 0)}  S {shots.get('team0_shots', 0)}  "
        f"G {goals.get('team0_goals', 0)}",
        f"{names.get('team1', 'Team B')}: poss {poss.get('team1_pct', 0):.0f}%  "
        f"P {passes.get('team1_passes', 0)}  S {shots.get('team1_shots', 0)}  "
        f"G {goals.get('team1_goals', 0)}",
        f"Loose {poss.get('loose_pct', 0):.0f}%   possessor {possessor_track_id} "
        f"(team{possessor_team})",
    ]
    overlay = frame.copy()
    cv2.rectangle(overlay, (6, 4), (620, 92), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    y = 22
    for text in lines:
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 20


def annotate_frame(
    frame: np.ndarray,
    detections: list[dict],
    ball=None,
    *,
    possessor_track_id: int | None = None,
    possessor_team: int | None = None,
    stats: dict | None = None,
    team_names: dict | None = None,
    draw_hud_panel: bool = True,
    ball_observed: bool = True,
    ball_interpolated: bool = False,
) -> np.ndarray:
    """Draw ground ovals + ball/goalpost boxes. Does not draw player rectangles."""
    out = frame.copy()
    player_like = {"player", "goalkeeper", "player_team1", "player_team_2"}
    feet: list[tuple[int, float, float]] = []
    for det in detections:
        if det.get("class_name") not in player_like:
            continue
        tid = det.get("track_id")
        if tid is None:
            continue
        cx, by = player_foot_point(det["bbox"])
        feet.append((int(tid), cx, by))
    offsets = _label_dy(feet)

    for det in detections:
        name = det.get("class_name")
        if name in player_like or name == "referee":
            tid = det.get("track_id")
            draw_player_circle(
                out,
                det,
                possessor=tid is not None and tid == possessor_track_id,
                label_dy=offsets.get(tid, -16) if tid is not None else -16,
            )
        elif name == "goalpost":
            draw_goalpost(out, det)

    if ball_observed or ball_interpolated:
        draw_ball_box(out, ball, interpolated=ball_interpolated and not ball_observed)

    if draw_hud_panel:
        draw_hud(
            out,
            stats,
            possessor_track_id=possessor_track_id,
            possessor_team=possessor_team,
            team_names=team_names,
        )
    return out
