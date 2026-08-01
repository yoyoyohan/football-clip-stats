#!/usr/bin/env python3
"""
Full match-clip stats pipeline (run locally — preferred over Kaggle for iteration).

Video in → YOLO (best.pt) → ByteTrack → team colors → homography pitch coords → stats out.

Usage:
  # 1. Calibrate pitch once per camera angle (interactive)
  python calibrate_pitch.py --source input_videos/elclasico.mp4 --frame 50

  # 2. Run full pipeline
  python run_clip.py --source input_videos/elclasico.mp4

  # Force re-detect (after retraining best.pt)
  python run_clip.py --source input_videos/moroccomatch.mp4 --no-cache
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

# Re-use main pipeline logic
from main import _video_fps
from analysis import BallInterpolator, TeamColorAssigner
from stats_engine import StatEngine, default_field_polygon, default_goal_boxes, default_homography
from trackers import Tracker
from utils import read_video
from utils.calibration import (
    auto_calibration_from_goalposts,
    default_calibration_path,
    load_calibration,
    save_calibration,
)
from utils.goal_regions import goal_mouth_lines_from_boxes, refine_goal_boxes_from_frames


def export_pitch_tracks(
    path: Path,
    frame_records: list[dict],
    video_frames: list,
    assigner: TeamColorAssigner,
    engine: StatEngine,
    ball_interpolator: BallInterpolator,
):
    """Per-frame player + ball positions in pitch meters (105 x 68)."""
    rows = []
    for record in frame_records:
        idx = record["frame_idx"]
        frame = video_frames[idx]
        dets = assigner.assign_teams(frame, record["detections"])
        obs = ball_interpolator.update(idx, record.get("ball"))
        ball_px = obs.position
        ball_m = engine.pitch.to_meters(ball_px) if ball_px else (None, None)

        for d in dets:
            if d.get("class_name") not in {"player", "goalkeeper"}:
                continue
            if d.get("track_id") is None:
                continue
            pm = engine.pitch.bbox_to_meters(d["bbox"])
            rows.append(
                {
                    "frame_idx": idx,
                    "track_id": d["track_id"],
                    "team_id": d.get("team_id"),
                    "class_name": d.get("class_name"),
                    "x_m": round(pm["x_m"], 2),
                    "y_m": round(pm["y_m"], 2),
                    "foot_px_x": round(pm["foot_px"][0], 1),
                    "foot_px_y": round(pm["foot_px"][1], 1),
                }
            )
        if ball_px:
            rows.append(
                {
                    "frame_idx": idx,
                    "track_id": "ball",
                    "team_id": None,
                    "class_name": "ball",
                    "x_m": round(ball_m[0], 2),
                    "y_m": round(ball_m[1], 2),
                    "foot_px_x": round(ball_px[0], 1),
                    "foot_px_y": round(ball_px[1], 1),
                }
            )

    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(stats: dict, fps: float) -> None:
    print("\n" + "=" * 50)
    print("MATCH CLIP STATS")
    print("=" * 50)
    print(f"Possession:     {stats['possession']}")
    print(f"Passes:         {stats['passes']}")
    print(f"Shots:          {stats['shots']}")
    print(f"On target:      {stats.get('shots_on_target', {})}")
    print(f"Goals:          {stats['goals']}")
    print(f"xG:             {stats.get('xg', {})}")

    print(f"\nPass events ({len(stats['pass_events'])}):")
    for i, e in enumerate(stats["pass_events"], 1):
        t = e["frame"] / fps
        print(
            f"  #{i} {t:.2f}s  {e['from_track_id']}→{e['to_track_id']}  "
            f"{e.get('distance_m', '?')}m @ {e.get('speed_mps', '?')} m/s"
        )

    print(f"\nShot events ({len(stats.get('shot_events', []))}):")
    for i, e in enumerate(stats.get("shot_events", []), 1):
        ot = "ON TARGET" if e["on_target"] else "off target"
        print(
            f"  #{i} {e['frame']/fps:.2f}s  team{e['team_id']}  "
            f"{e['distance_to_goal_m']}m  {e['ball_speed_mps']} m/s  {ot}"
        )

    if stats.get("goal_events"):
        print(f"\nGoals:")
        for e in stats["goal_events"]:
            print(f"  frame {e['frame']}  team{e['scoring_team']} scores")


def main():
    parser = argparse.ArgumentParser(description="Run full stats pipeline on a video clip")
    parser.add_argument("--source", required=True, help="Path to video clip")
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--no-cache", action="store_true", help="Re-run YOLO+tracking")
    parser.add_argument("--team0-name", default="team0")
    parser.add_argument("--team1-name", default="team1")
    args = parser.parse_args()

    video_path = Path(args.source)
    if not video_path.exists():
        sys.exit(f"Video not found: {video_path}")

    stem = video_path.stem
    out_dir = Path("output_videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    track_cache = out_dir / f"{stem}_frame_records.pkl"

    video_frames = read_video(str(video_path))
    h, w = video_frames[0].shape[:2]
    fps = _video_fps(str(video_path))

    if args.no_cache and track_cache.exists():
        track_cache.unlink()

    if track_cache.exists():
        print(f"Loading cached tracks: {track_cache}")
        with track_cache.open("rb") as f:
            frame_records = pickle.load(f)
    else:
        print(f"Detecting with {args.model} ({len(video_frames)} frames)...")
        tracker = Tracker(args.model)
        raw = tracker.get_object_tracks(video_frames)
        frame_records = [
            {
                "frame_idx": r["frame_idx"],
                "detections": r["detections"],
                "ball": r["ball"],
                "overlay": r.get("overlay", {}),
            }
            for r in raw
        ]
        with track_cache.open("wb") as f:
            pickle.dump(frame_records, f)
        print(f"Cached: {track_cache}")

    assigner = TeamColorAssigner()
    for rec in frame_records:
        assigner.collect(video_frames[rec["frame_idx"]], rec["detections"])
    assigner.fit_teams()
    print(f"Team colors assigned to {len(assigner._track_teams)} tracks")

    cal_path = default_calibration_path(str(video_path))
    cal = load_calibration(cal_path, w, h)
    if cal:
        print(f"Pitch calibration: {cal_path}")
        homography = cal["homography"]
        goal_boxes = cal["goal_boxes"]
        field_polygon = cal["field_polygon"]
        attacking_direction = cal["attacking_direction"]
        # Prefer real video FPS; only override if calibration explicitly stored fps.
        if cal.get("fps"):
            fps = float(cal["fps"])
    else:
        print(f"No calibration at {cal_path}")
        auto = None
        for rec in frame_records:
            posts = rec.get("overlay", {}).get("goalposts", [])
            if len(posts) >= 2:
                auto = auto_calibration_from_goalposts(posts, w, h)
                if auto:
                    auto["video"] = str(video_path)
                    auto["fps"] = fps
                    auto["attacking_direction"] = "left_to_right"
                    save_calibration(cal_path, auto)
                    print(f"Auto-calibrated from goalposts → saved {cal_path}")
                    break
        if auto:
            cal = load_calibration(cal_path, w, h)
        if cal:
            homography = cal["homography"]
            goal_boxes = cal["goal_boxes"]
            field_polygon = cal["field_polygon"]
            attacking_direction = cal["attacking_direction"]
        else:
            print("  → Calibrate manually:")
            print(f"     python calibrate_pitch.py --source {video_path} --mode goal")
            print(f"     python calibrate_pitch.py --source {video_path} --mode landmarks")
            print("  → Using default homography (less accurate pitch meters)")
            homography = default_homography(w, h)
            goal_boxes = default_goal_boxes(w, h)
            field_polygon = default_field_polygon(w, h)
            attacking_direction = "left_to_right"

    print(f"Video FPS: {fps:.2f}")

    goalpost_boxes = refine_goal_boxes_from_frames(frame_records, w)
    if goalpost_boxes:
        goal_boxes = goalpost_boxes
    goal_lines = goal_mouth_lines_from_boxes(goal_boxes)

    engine = StatEngine(
        goal_boxes=goal_boxes,
        goal_lines=goal_lines,
        homography=homography,
        field_polygon=field_polygon,
        attacking_direction=attacking_direction,
        fps=fps,
        frame_width=w,
        frame_height=h,
    )

    bi = BallInterpolator(fps=fps)
    for rec in frame_records:
        idx = rec["frame_idx"]
        dets = assigner.assign_teams(video_frames[idx], rec["detections"])
        obs = bi.update(idx, rec.get("ball"))
        engine.update(idx, dets, obs.position, ball_observed=obs.observed)

    stats = engine.get_stats()
    stats["video"] = str(video_path)
    stats["fps"] = fps
    stats["frame_size"] = {"width": w, "height": h}
    stats["calibration"] = str(cal_path) if cal else None
    stats["team_names"] = {"team0": args.team0_name, "team1": args.team1_name}
    stats["pitch_meters"] = {"length": 105.0, "width": 68.0}

    stats_path = out_dir / f"{stem}_stats.json"
    csv_path = out_dir / f"{stem}_stats_per_frame.csv"
    tracks_path = out_dir / f"{stem}_pitch_tracks.csv"

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    engine.export_csv(str(csv_path))
    export_pitch_tracks(tracks_path, frame_records, video_frames, assigner, engine, bi)

    print(f"\nWrote:\n  {stats_path}\n  {csv_path}\n  {tracks_path}")
    print_summary(stats, fps)


if __name__ == "__main__":
    main()
