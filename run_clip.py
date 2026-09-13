#!/usr/bin/env python3
"""
Full match-clip stats pipeline (run locally — preferred over Kaggle for iteration).

Video in → YOLO (best.pt) → ByteTrack → team colors → homography pitch coords → stats out.

Usage:
  # Drop a clip and run — pitch auto-calibrates from goalpost detections
  python run_clip.py --source input_videos/elclasico.mp4

  # Force re-detect + refresh auto calibration
  python run_clip.py --source input_videos/moroccomatch.mp4 --no-cache

  # Force re-run auto calibration even if a manual JSON exists
  python run_clip.py --source input_videos/elclasico.mp4 --force-auto-cal

  # Interactive fallback (only if auto-cal fails or you want higher accuracy)
  python calibrate_pitch.py --source input_videos/elclasico.mp4
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
    default_calibration_path,
    is_auto_calibration,
    is_manual_calibration,
    load_calibration,
    robust_auto_calibration_from_frames,
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
    parser.add_argument(
        "--force-auto-cal",
        action="store_true",
        help="Recompute automatic pitch calibration even if a manual JSON exists",
    )
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
    method = cal.get("method") if cal else None

    # Keep manual/landmark/corner JSON unless --force-auto-cal.
    # Refresh prior auto calibrations when --no-cache or --force-auto-cal.
    should_auto = False
    if args.force_auto_cal:
        should_auto = True
        print("--force-auto-cal: recomputing automatic pitch calibration")
    elif cal is None:
        should_auto = True
        print(f"No calibration at {cal_path} — running automatic multi-frame calibration")
    elif is_auto_calibration(method) and args.no_cache:
        should_auto = True
        print(f"Refreshing auto calibration ({method}) because --no-cache was set")
    elif is_manual_calibration(method):
        print(f"Pitch calibration (manual/{method or 'legacy'}): {cal_path}")
    else:
        print(f"Pitch calibration: {cal_path} (method={method})")

    if should_auto:
        auto = robust_auto_calibration_from_frames(frame_records, w, h)
        if auto:
            auto["video"] = str(video_path)
            auto["fps"] = fps
            auto["attacking_direction"] = "left_to_right"
            save_calibration(cal_path, auto)
            q = auto.get("quality")
            q_txt = f"{q:.2f}" if isinstance(q, (int, float)) else "?"
            print(
                f"Auto-calibrated ({auto.get('method')}, quality={q_txt}) → saved {cal_path}"
            )
            if auto.get("quality_notes"):
                print(f"  notes: {', '.join(auto['quality_notes'])}")
            cal = load_calibration(cal_path, w, h)
        else:
            print("Automatic calibration failed (insufficient / unstable goalposts)")
            if cal and is_manual_calibration(method) and not args.force_auto_cal:
                print(f"  → Keeping existing calibration: {cal_path}")
            elif cal and is_auto_calibration(method) and not args.force_auto_cal:
                print(f"  → Keeping previous auto calibration: {cal_path}")
            else:
                cal = None

    if cal:
        homography = cal["homography"]
        goal_boxes = cal["goal_boxes"]
        field_polygon = cal["field_polygon"]
        attacking_direction = cal["attacking_direction"]
        # Prefer real video FPS; only override if calibration explicitly stored fps.
        if cal.get("fps"):
            fps = float(cal["fps"])
        if not should_auto:
            # Already printed path above for keep paths; ensure load message once.
            pass
    else:
        print("  → Interactive fallback:")
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
