"""
Pitch calibration for broadcast clips where the full field is NOT visible.

Usage:
    # Scrub to a frame, then calibrate (default = one-goal mode):
    python calibrate_pitch.py --source input_videos/moroccomatch.mp4

    # Wider shot — 4+ landmarks:
    python calibrate_pitch.py --source input_videos/moroccomatch.mp4 --mode landmarks

    python calibrate_pitch.py --source input_videos/moroccomatch.mp4 --mode auto
"""

from __future__ import annotations

import argparse
import select
import sys
import textwrap

import cv2
import numpy as np

from trackers import Tracker
from utils.calibration import (
    LANDMARK_CATALOG,
    LANDMARK_HELP,
    LANDMARK_SHORTCUTS,
    auto_calibration_from_goalposts,
    build_calibration,
    build_calibration_from_landmarks,
    build_calibration_from_one_goal,
    default_calibration_path,
    save_calibration,
)
from utils.goal_regions import extract_overlay

points_px: list[tuple[float, float]] = []
points_m: list[tuple[float, float]] = []
active_landmark: str | None = None
frame_copy = None

MENU_SECTIONS: list[tuple[str, list[str]]] = [
    (
        "GOAL POSTS (click ground at bottom of post)",
        ["1", "2", "3", "4"],
    ),
    (
        "SPOTS",
        ["5", "6", "7"],
    ),
    (
        "CORNER FLAGS (only if visible)",
        ["8", "9", "0", "-"],
    ),
    (
        "PENALTY-BOX FRONT LINE (18-yard line corners)",
        ["q", "w", "e", "r"],
    ),
]


# ---------------------------------------------------------------------------
# Video scrubbing
# ---------------------------------------------------------------------------
class VideoScrubber:
    """Open a video and jump to any frame (trackbar + keys)."""

    def __init__(self, path: str, start_frame: int = 0):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise SystemExit(f"Cannot open video: {path}")
        self.total = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.frame_idx = int(np.clip(start_frame, 0, self.total - 1))
        self.frame: np.ndarray | None = None
        self.seek(self.frame_idx)

    def seek(self, idx: int) -> bool:
        idx = int(np.clip(idx, 0, self.total - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            return False
        self.frame_idx = idx
        self.frame = frame
        return True

    def step(self, delta: int) -> bool:
        return self.seek(self.frame_idx + delta)

    def release(self) -> None:
        self.cap.release()

    def time_label(self) -> str:
        t = self.frame_idx / self.fps if self.fps > 0 else 0.0
        return f"frame {self.frame_idx}/{self.total - 1}  ({t:.1f}s)"


def _poll_terminal_key() -> str | None:
    """Read a single key from the terminal (macOS OpenCV often misses window keys)."""
    if not sys.stdin.isatty():
        return None
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if not ready:
        return None
    ch = sys.stdin.read(1)
    if ch in ("\n", "\r"):
        return "enter"
    return ch.lower()


def _read_ui_key(wait_ms: int = 30) -> str | None:
    """Prefer terminal input; fall back to the OpenCV window (click it first on Mac)."""
    term = _poll_terminal_key()
    if term:
        return term
    key = cv2.waitKeyEx(wait_ms)
    if key == -1:
        return None
    if key in (27,):
        return "esc"
    # Arrow keys (platform-dependent codes from waitKeyEx)
    if key in (81, 2, 2424832, 63234):
        return "left"
    if key in (83, 3, 2555904, 63235):
        return "right"
    if key in (82, 0, 2490368, 63232):
        return "up"
    if key in (84, 1, 2621440, 63233):
        return "down"
    code = key & 0xFF
    if 32 <= code < 127:
        return chr(code).lower()
    return None


def _apply_scrub_key(scrubber: VideoScrubber, ch: str | None) -> bool:
    """Move scrubber if ch is a scrub key. Returns True if frame changed."""
    if not ch:
        return False
    if ch in ("a", "left", ","):
        return scrubber.step(-1)
    if ch in ("d", "right", "."):
        return scrubber.step(1)
    if ch in ("[",):
        return scrubber.step(-10)
    if ch in ("]",):
        return scrubber.step(10)
    if ch in ("{",):
        return scrubber.step(-30)
    if ch in ("}",):
        return scrubber.step(30)
    return False


def pick_frame(scrubber: VideoScrubber) -> int | None:
    """
    Interactive frame picker. Drag the slider or use keys, then press Enter/Space.
    Returns chosen frame index, or None if quit.
    """
    window = "Pick a frame — drag slider or a/d, then SPACE/ENTER"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    trackbar_busy = {"v": False}

    def on_trackbar(pos: int) -> None:
        if trackbar_busy["v"]:
            return
        scrubber.seek(pos)

    cv2.createTrackbar("frame", window, scrubber.frame_idx, max(0, scrubber.total - 1), on_trackbar)

    print("\n" + "=" * 60)
    print("PICK A FRAME")
    print("=" * 60)
    print("  Drag the 'frame' slider under the video")
    print("  a / ←  previous frame     d / →  next frame")
    print("  [ ]    jump ±10 frames    { }    jump ±30")
    print("  SPACE or ENTER  →  use this frame for calibration")
    print("  x or ESC        →  quit")
    print("=" * 60 + "\n")

    while True:
        assert scrubber.frame is not None
        display = scrubber.frame.copy()
        h, w = display.shape[:2]
        overlay = display.copy()
        cv2.rectangle(overlay, (10, 10), (min(720, w - 10), 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, display, 0.35, 0, display)
        lines = [
            "SCROLL TO A FRAME WITH A GOAL OR CLEAR PITCH LINES",
            scrubber.time_label(),
            "Slider / a d [ ] { }   then SPACE or ENTER to calibrate",
            "x = quit",
        ]
        for i, line in enumerate(lines):
            cv2.putText(display, line, (20, 36 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(window, display)

        # Keep trackbar in sync if we stepped with keys
        cur = cv2.getTrackbarPos("frame", window)
        if cur != scrubber.frame_idx:
            trackbar_busy["v"] = True
            cv2.setTrackbarPos("frame", window, scrubber.frame_idx)
            trackbar_busy["v"] = False

        ch = _read_ui_key(30)
        if ch in ("x", "esc", "q"):
            print("Quit without calibrating.")
            cv2.destroyWindow(window)
            return None
        if ch in ("enter", " ", "c"):
            print(f"Using {scrubber.time_label()}")
            cv2.destroyWindow(window)
            return scrubber.frame_idx
        if _apply_scrub_key(scrubber, ch):
            trackbar_busy["v"] = True
            cv2.setTrackbarPos("frame", window, scrubber.frame_idx)
            trackbar_busy["v"] = False


def _shortcut_map() -> dict[str, str]:
    return {key: name for key, name in LANDMARK_SHORTCUTS}


def _help_text(name: str) -> str:
    return LANDMARK_HELP.get(name, name.replace("_", " "))


def _wrap(text: str, width: int = 52) -> list[str]:
    return textwrap.wrap(text, width=width) or [text]


def _draw_help(frame: np.ndarray, frame_label: str = "") -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    panel_w = min(560, w - 20)
    panel_h = min(420, h - 20)

    overlay = out.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)

    shortcuts = _shortcut_map()
    y = 32
    header = [
        "PITCH CALIBRATION",
        frame_label or "",
        "LEFT/RIGHT = sides of your TV screen",
        "a/d/[ ] change frame (clears clicks)   c=save  b=reset  x=quit",
        "Pick key -> click mark -> repeat (need 4+)",
        "",
    ]
    for line in header:
        if not line:
            continue
        cv2.putText(out, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 18

    for section_title, keys in MENU_SECTIONS:
        cv2.putText(out, section_title, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 220, 255), 1)
        y += 16
        for key in keys:
            name = shortcuts[key]
            desc = _help_text(name)
            for i, line in enumerate(_wrap(f"[{key}] {desc}", width=62)):
                prefix = "     " if i else "  "
                cv2.putText(
                    out, prefix + line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1
                )
                y += 15
        y += 4

    if active_landmark:
        cv2.putText(out, ">>> CLICK NOW:", (20, min(h - 50, y + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        for i, line in enumerate(_wrap(_help_text(active_landmark), width=60)):
            cv2.putText(
                out, line, (20, min(h - 28, y + 32 + i * 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1
            )

    cv2.putText(
        out,
        f"Points placed: {len(points_px)} / 4 minimum",
        (20, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0) if len(points_px) >= 4 else (0, 200, 255),
        1,
    )

    for i, (px, py) in enumerate(points_px):
        cv2.circle(out, (int(px), int(py)), 8, (0, 255, 255), -1)
        cv2.putText(out, str(i + 1), (int(px) + 10, int(py) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return out


def _print_terminal_help() -> None:
    shortcuts = _shortcut_map()
    print("\n" + "=" * 60)
    print("HOW TO READ LEFT / RIGHT / NEAR / FAR")
    print("=" * 60)
    print("  LEFT side of screen  = goal on the LEFT of your TV")
    print("  RIGHT side of screen = goal on the RIGHT of your TV")
    print("  BOTTOM of screen     = touchline NEAR the camera")
    print("  TOP of screen        = touchline FAR from the camera")
    print("=" * 60)
    print("\nPress a KEY, then CLICK that point on the frozen video frame:\n")
    for section_title, keys in MENU_SECTIONS:
        print(f"--- {section_title} ---")
        for key in keys:
            name = shortcuts[key]
            print(f"  [{key}]  {_help_text(name)}")
        print()
    print("Scrub: a/d ±1 frame, [ ] ±10   Controls: c=save  b=reset  x=quit\n")


def _on_mouse_landmarks(event, x, y, _flags, param):
    global points_px, points_m, active_landmark, frame_copy
    if event != cv2.EVENT_LBUTTONDOWN or active_landmark is None:
        return
    mx, my = LANDMARK_CATALOG[active_landmark]
    points_px.append((float(x), float(y)))
    points_m.append((mx, my))
    print(f"  ✓ Point {len(points_px)}: {_help_text(active_landmark)}  @ pixel ({x}, {y})")
    active_landmark = None
    frame_copy = _draw_help(param["frame"], param.get("label", ""))


def _save_calibration(
    data: dict,
    output_path: str,
    frame: np.ndarray,
    fps: float | None = None,
) -> None:
    if fps is not None:
        data["fps"] = float(fps)
    save_calibration(output_path, data)
    print(f"\nSaved calibration to {output_path}")

    preview = frame.copy()
    poly = np.array(data["field_polygon"], dtype=np.int32)
    cv2.polylines(preview, [poly], True, (0, 255, 0), 2)
    for side, box in data["goal_boxes"].items():
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(preview, side, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.imshow("Calibration preview — green=pitch outline, blue=goals", preview)
    cv2.waitKey(0)


def run_goal_mode(scrubber: VideoScrubber, output_path: str, video_source: str) -> None:
    """Tight broadcast shots where you only see ONE goal (2 posts, maybe penalty spot)."""
    posts: list[tuple[float, float]] = []
    depth_px: tuple[float, float] | None = None
    goal_side: str | None = None
    depth_type = "penalty_spot"
    step = "side"
    display = scrubber.frame.copy() if scrubber.frame is not None else None
    window = "Calibrate — one goal visible"

    def clear_clicks() -> None:
        nonlocal posts, depth_px, goal_side, step
        posts.clear()
        depth_px = None
        goal_side = None
        step = "side"

    def redraw(msg: str) -> None:
        nonlocal display
        assert scrubber.frame is not None
        display = scrubber.frame.copy()
        overlay = display.copy()
        cv2.rectangle(overlay, (10, 10), (720, 240), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
        lines = [
            "ONE-GOAL CALIBRATION",
            scrubber.time_label() + "   a/d/[ ] change frame (resets clicks)",
            "l = goal on LEFT of TV    r = goal on RIGHT of TV",
            "Click post 1, then post 2.  Then press c to SAVE (2 posts is enough).",
            "Optional: click penalty spot / 18-yard line for better accuracy",
            "Keys work in THIS TERMINAL or video window",
            "c = save   b = reset clicks   x = quit",
            "",
            msg,
        ]
        for i, line in enumerate(lines):
            color = (0, 255, 255) if i == len(lines) - 1 else (255, 255, 255)
            cv2.putText(display, line, (20, 36 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        for i, (px, py) in enumerate(posts):
            cv2.circle(display, (int(px), int(py)), 10, (0, 255, 255), -1)
            cv2.putText(display, f"post{i+1}", (int(px) + 12, int(py) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if depth_px:
            cv2.circle(display, (int(depth_px[0]), int(depth_px[1])), 10, (0, 200, 255), -1)
            cv2.putText(display, "depth", (int(depth_px[0]) + 12, int(depth_px[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        cv2.imshow(window, display)

    def save_now() -> bool:
        if not goal_side or len(posts) != 2:
            print("Need goal side (l/r) and 2 post clicks before saving.")
            return False
        assert scrubber.frame is not None
        try:
            data = build_calibration_from_one_goal(
                posts[0], posts[1], goal_side,
                depth_px=depth_px, depth_type=depth_type,
            )
        except ValueError as e:
            print(f"Calibration failed: {e}")
            return False
        h, w = scrubber.frame.shape[:2]
        data["video"] = video_source
        data["frame_idx"] = scrubber.frame_idx
        data["frame_size"] = {"width": w, "height": h}
        data["attacking_direction"] = "left_to_right"
        est = " (depth estimated from goal width)" if data.get("depth_estimated") else ""
        print(f"Saved single-goal calibration{est} from {scrubber.time_label()}.")
        _save_calibration(data, output_path, scrubber.frame, fps=scrubber.fps)
        return True

    def handle_key(ch: str | None) -> bool:
        nonlocal step, depth_px, goal_side, depth_type
        if not ch:
            return False
        if ch in ("x", "esc", "q"):
            print("Quit without saving.")
            return True
        if _apply_scrub_key(scrubber, ch):
            clear_clicks()
            redraw(f"Moved to {scrubber.time_label()}. Press l/r then click posts.")
            print(f"Frame → {scrubber.time_label()} (clicks cleared)")
            return False
        if ch == "l":
            goal_side = "left"
            step = "post1"
            redraw("LEFT goal — click the LEFT post (ground at bottom).")
            print("Goal on LEFT of screen. Click first post...")
        elif ch == "r":
            goal_side = "right"
            step = "post1"
            redraw("RIGHT goal — click the LEFT post (ground at bottom).")
            print("Goal on RIGHT of screen. Click first post...")
        elif ch == "p":
            depth_type = "penalty_spot"
            print("Depth landmark: penalty spot (~11m from goal)")
        elif ch == "y":
            depth_type = "penalty_box_front_center"
            print("Depth landmark: front of 18-yard box")
        elif ch in ("s", "enter") and len(posts) == 2 and step in ("depth", "save"):
            depth_px = None
            step = "save"
            redraw("No 3rd point — depth will be estimated. Press c to save.")
            print("Skipping 3rd click — estimating depth from goal width. Press c to save.")
        elif ch == "b":
            clear_clicks()
            redraw("Reset. Press l or r for which goal is visible.")
            print("Reset.")
        elif ch == "c":
            if save_now():
                return True
        return False

    def on_mouse(event, x, y, _flags, _param):
        nonlocal step, depth_px
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if step == "post1":
            posts.clear()
            depth_px = None
            posts.append((float(x), float(y)))
            step = "post2"
            redraw("Post 1 set. Click the OTHER post of the same goal.")
        elif step == "post2" and len(posts) == 1:
            posts.append((float(x), float(y)))
            step = "depth"
            redraw("Both posts set — press c in the TERMINAL to save (or click a 3rd point).")
            print("\n>>> Both posts set.")
            print(">>> Press c here in the terminal to SAVE (no 3rd click needed).")
            print(">>> Or click penalty spot / 18-yard line for better accuracy.\n")
        elif step == "depth" and len(posts) == 2:
            depth_px = (float(x), float(y))
            step = "save"
            redraw("Depth set. Press c to save.")
            print("Depth point set. Press c to save.")

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    def on_trackbar(pos: int) -> None:
        if pos == scrubber.frame_idx:
            return
        scrubber.seek(pos)
        clear_clicks()
        redraw(f"Moved to {scrubber.time_label()}. Press l/r then click posts.")

    cv2.createTrackbar("frame", window, scrubber.frame_idx, max(0, scrubber.total - 1), on_trackbar)
    redraw("Press l or r — which side of the TV is the goal you see?")

    print("\n" + "=" * 60)
    print("ONE-GOAL MODE")
    print("=" * 60)
    print("  Slider / a d [ ]  — change frame anytime (clears clicks)")
    print("  1. Press l or r")
    print("  2. Click both goal posts")
    print("  3. Press c to SAVE")
    print("=" * 60 + "\n")

    while True:
        assert display is not None
        cv2.imshow(window, display)
        if handle_key(_read_ui_key(30)):
            break

    cv2.destroyAllWindows()


def run_landmarks_mode(scrubber: VideoScrubber, output_path: str, video_source: str) -> None:
    global points_px, points_m, active_landmark, frame_copy
    points_px.clear()
    points_m.clear()
    active_landmark = None

    shortcuts = _shortcut_map()
    window = "Calibrate pitch — press key then click"
    mouse_param = {"frame": scrubber.frame, "label": scrubber.time_label()}

    def refresh() -> None:
        mouse_param["frame"] = scrubber.frame
        mouse_param["label"] = scrubber.time_label()
        frame_copy_local = _draw_help(scrubber.frame, scrubber.time_label())
        globals()["frame_copy"] = frame_copy_local

    def clear_and_refresh() -> None:
        points_px.clear()
        points_m.clear()
        globals()["active_landmark"] = None
        refresh()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, _on_mouse_landmarks, mouse_param)

    def on_trackbar(pos: int) -> None:
        if pos == scrubber.frame_idx:
            return
        scrubber.seek(pos)
        clear_and_refresh()
        print(f"Frame → {scrubber.time_label()} (clicks cleared)")

    cv2.createTrackbar("frame", window, scrubber.frame_idx, max(0, scrubber.total - 1), on_trackbar)
    refresh()
    _print_terminal_help()

    while True:
        if frame_copy is not None:
            cv2.imshow(window, frame_copy)
        ch = _read_ui_key(30)
        if ch in ("x", "esc"):
            print("Quit without saving.")
            break
        if _apply_scrub_key(scrubber, ch):
            clear_and_refresh()
            cv2.setTrackbarPos("frame", window, scrubber.frame_idx)
            print(f"Frame → {scrubber.time_label()} (clicks cleared)")
            continue
        if ch == "b":
            clear_and_refresh()
            print("Reset all points.")
        if ch and ch in shortcuts:
            globals()["active_landmark"] = shortcuts[ch]
            refresh()
            print(f"\n>>> Selected [{ch}]: {_help_text(active_landmark)}")
            print("    Click that exact spot on the video now...")
        if ch == "c":
            if len(points_px) < 4:
                print(f"Need at least 4 points (you have {len(points_px)}).")
                continue
            assert scrubber.frame is not None
            h, w = scrubber.frame.shape[:2]
            data = build_calibration_from_landmarks(points_px, points_m)
            data["video"] = video_source
            data["frame_idx"] = scrubber.frame_idx
            data["frame_size"] = {"width": w, "height": h}
            data["method"] = "landmarks"
            data["attacking_direction"] = "left_to_right"
            _save_calibration(data, output_path, scrubber.frame, fps=scrubber.fps)
            break

    cv2.destroyAllWindows()


def run_corners_mode(scrubber: VideoScrubber, output_path: str, video_source: str) -> None:
    points: list[tuple[int, int]] = []
    labels = [
        "BOTTOM-LEFT corner (near-left)",
        "BOTTOM-RIGHT corner (near-right)",
        "TOP-RIGHT corner (far-right)",
        "TOP-LEFT corner (far-left)",
    ]
    window = "Calibrate — 4 corners (full pitch only)"

    def redraw():
        assert scrubber.frame is not None
        out = scrubber.frame.copy()
        for i, (x, y) in enumerate(points):
            cv2.circle(out, (x, y), 8, (0, 255, 255), -1)
            cv2.putText(out, str(i + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        hint = labels[len(points)] if len(points) < 4 else "Press c to save"
        cv2.putText(out, scrubber.time_label(), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(out, hint + "   a/d change frame", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.imshow(window, out)

    def on_mouse(event, x, y, _f, _p):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            print(f"  ✓ {labels[len(points) - 1]} @ ({x}, {y})")
            redraw()

    def on_trackbar(pos: int) -> None:
        if pos == scrubber.frame_idx:
            return
        scrubber.seek(pos)
        points.clear()
        redraw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    cv2.createTrackbar("frame", window, scrubber.frame_idx, max(0, scrubber.total - 1), on_trackbar)
    redraw()
    while True:
        ch = _read_ui_key(30)
        if ch in ("x", "esc"):
            break
        if _apply_scrub_key(scrubber, ch):
            points.clear()
            cv2.setTrackbarPos("frame", window, scrubber.frame_idx)
            redraw()
            continue
        if ch == "b":
            points.clear()
            redraw()
        if ch == "c" and len(points) == 4:
            assert scrubber.frame is not None
            h, w = scrubber.frame.shape[:2]
            data = build_calibration([(float(x), float(y)) for x, y in points])
            data["video"] = video_source
            data["frame_idx"] = scrubber.frame_idx
            data["frame_size"] = {"width": w, "height": h}
            data["method"] = "corners"
            _save_calibration(data, output_path, scrubber.frame, fps=scrubber.fps)
            break
    cv2.destroyAllWindows()


def run_auto_mode(scrubber: VideoScrubber, output_path: str, model_path: str, video_source: str) -> None:
    assert scrubber.frame is not None
    print(f"Running YOLO on {scrubber.time_label()} to find goalposts...")
    tracker = Tracker(model_path)
    records = tracker.get_object_tracks([scrubber.frame])
    overlay = extract_overlay(records[0]["detections"])
    goalposts = overlay.get("goalposts", [])
    h, w = scrubber.frame.shape[:2]

    if len(goalposts) < 2:
        raise SystemExit(
            "Auto calibration failed: need at least 2 goalpost detections.\n"
            "Try: python calibrate_pitch.py --mode goal\n"
            "  (scrub to a frame that shows a goal)"
        )

    data = auto_calibration_from_goalposts(goalposts, w, h)
    if data is None:
        raise SystemExit(
            "Auto calibration failed: could not match goalposts on BOTH goals.\n"
            "Use: python calibrate_pitch.py --mode goal"
        )

    data["video"] = video_source
    data["frame_idx"] = scrubber.frame_idx
    data["attacking_direction"] = "left_to_right"
    print(f"Auto calibration OK from {len(goalposts)} goalpost detection(s).")

    preview = scrubber.frame.copy()
    for g in goalposts:
        x1, y1, x2, y2 = map(int, g["bbox"])
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Detected goalposts", preview)
    cv2.waitKey(1500)
    _save_calibration(data, output_path, scrubber.frame, fps=scrubber.fps)


def main():
    parser = argparse.ArgumentParser(description="Calibrate pitch (partial views supported)")
    parser.add_argument("--source", default="input_videos/elclasico.mp4")
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Starting frame for the scrubber (default: 0). You can change it in the UI.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--mode",
        choices=["goal", "landmarks", "corners", "auto"],
        default="goal",
        help="goal=one visible goal (2 posts, default), landmarks=4+ marks, corners=full pitch, auto=YOLO goalposts",
    )
    parser.add_argument(
        "--skip-picker",
        action="store_true",
        help="Skip the frame-picker screen and go straight to calibration at --frame",
    )
    parser.add_argument("--model", default="models/best.pt")
    args = parser.parse_args()

    scrubber = VideoScrubber(args.source, args.frame)
    output_path = args.output or str(default_calibration_path(args.source))

    try:
        if not args.skip_picker and args.mode != "auto":
            chosen = pick_frame(scrubber)
            if chosen is None:
                return
        elif args.mode == "auto" and not args.skip_picker:
            chosen = pick_frame(scrubber)
            if chosen is None:
                return

        if args.mode == "goal":
            run_goal_mode(scrubber, output_path, args.source)
        elif args.mode == "landmarks":
            run_landmarks_mode(scrubber, output_path, args.source)
        elif args.mode == "corners":
            run_corners_mode(scrubber, output_path, args.source)
        else:
            run_auto_mode(scrubber, output_path, args.model, args.source)
    finally:
        scrubber.release()


if __name__ == "__main__":
    main()
