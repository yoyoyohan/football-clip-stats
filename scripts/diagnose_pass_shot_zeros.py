#!/usr/bin/env python3
"""Offline diagnosis for 0-pass / 0-shot clips using cached frame_records pkls.

Usage:
  python scripts/diagnose_pass_shot_zeros.py \\
      --pkl output_videos/clip_14_frame_records.pkl \\
      [--tracks output_videos/clip_14_pitch_tracks.csv] \\
      [--stats output_videos/clip_14_stats.json] \\
      [--cal calibration/clip_14.json] \\
      [--fps 30] [--width 1920] [--height 1080]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.ball_interpolator import BallInterpolator
from analysis.pass_detector import MAX_BALL_JUMP_M, MAX_BALL_SPEED_MPS, PassDetector
from analysis.pitch_coordinates import PitchCoordinateMapper
from stats_engine import default_homography
from utils.calibration import load_calibration
from utils.detection_utils import normalize_ball


def _load_pitch(cal_path: str | None, w: int, h: int, fps: float) -> PitchCoordinateMapper:
    if cal_path:
        cal = load_calibration(cal_path, w, h)
        if cal:
            return PitchCoordinateMapper(cal["homography"], fps=fps)
    return PitchCoordinateMapper(default_homography(w, h), fps=fps)


def diagnose(pkl: Path, tracks: Path | None, stats: Path | None, cal: str | None,
             fps: float, w: int, h: int) -> dict:
    with pkl.open("rb") as f:
        records = pickle.load(f)

    raw_ball = 0
    for rec in records:
        pos, _, _ = normalize_ball(rec.get("ball"))
        if pos is not None:
            raw_ball += 1

    bi = BallInterpolator(fps=fps, frame_width=w, frame_height=h)
    smoothed = bi.smooth_frame_records(records)
    smooth_pos = sum(1 for o in smoothed if o.position is not None)
    smooth_obs = sum(1 for o in smoothed if o.observed)

    # Team ids on players in a sample of frames
    team_counts = Counter()
    player_frames = 0
    for rec in records[:: max(1, len(records) // 50)]:
        for d in rec.get("detections") or []:
            if d.get("class_name") in {"player", "goalkeeper"}:
                player_frames += 1
                team_counts[d.get("team_id")] += 1

    pitch = _load_pitch(cal, w, h, fps)

    # Ball jump stats in meters on smoothed observed pairs
    jumps_m = []
    speeds = []
    teleport_speed = 0
    teleport_jump = 0
    prev = None
    prev_i = None
    for i, obs in enumerate(smoothed):
        if not obs.observed or obs.position is None:
            continue
        if prev is not None and prev_i is not None:
            gap = max(1, i - prev_i)
            dist = pitch.distance_m(prev, obs.position)
            mps = pitch.speed_mps(prev, obs.position, frame_gap=gap)
            jumps_m.append(dist)
            speeds.append(mps)
            if mps > MAX_BALL_SPEED_MPS:
                teleport_speed += 1
            if dist > MAX_BALL_JUMP_M * gap:
                teleport_jump += 1
        prev, prev_i = obs.position, i

    # Re-run PassDetector only (no team assigner — uses team_id from pkl if present)
    det = PassDetector(fps=fps, frame_width=w, pitch=pitch)
    for rec, obs in zip(records, smoothed):
        players = [
            d for d in (rec.get("detections") or [])
            if d.get("team_id") is not None
            and d.get("class_name") not in {"ball", "referee", "goalpost", "linesman"}
        ]
        det.update(rec["frame_idx"], obs.position, 0.0, players, ball_observed=obs.observed)

    out = {
        "frames": len(records),
        "raw_ball_frames": raw_ball,
        "raw_ball_pct": round(100.0 * raw_ball / max(1, len(records)), 1),
        "smooth_ball_frames": smooth_pos,
        "smooth_observed_frames": smooth_obs,
        "smooth_ball_pct": round(100.0 * smooth_pos / max(1, len(records)), 1),
        "team_id_counts_sample": {str(k): v for k, v in team_counts.items()},
        "ball_jump_m_p50": round(float(np.median(jumps_m)), 3) if jumps_m else None,
        "ball_jump_m_p95": round(float(np.percentile(jumps_m, 95)), 3) if jumps_m else None,
        "ball_speed_mps_p50": round(float(np.median(speeds)), 2) if speeds else None,
        "ball_speed_mps_p95": round(float(np.percentile(speeds, 95)), 2) if speeds else None,
        "segments_over_42mps": teleport_speed,
        "segments_over_8m_jump": teleport_jump,
        "pass_detector_counts": det.counts(),
        "pass_events": len(det.events),
        "gates": {
            "MAX_BALL_SPEED_MPS": MAX_BALL_SPEED_MPS,
            "MAX_BALL_JUMP_M": MAX_BALL_JUMP_M,
            "note": "Pre-fix code rejected segments with mps>42; wide-FOV H maps ~28px/f above that.",
        },
    }

    if stats and stats.exists():
        with stats.open() as f:
            s = json.load(f)
        out["cached_stats_passes"] = s.get("passes")
        out["cached_stats_shots"] = s.get("shots")
        out["cached_possession"] = s.get("possession")
        out["cached_calibration"] = s.get("calibration")

    if tracks and tracks.exists():
        import csv
        with tracks.open() as f:
            rows = list(csv.DictReader(f))
        ball_rows = [r for r in rows if r.get("class_name") == "ball"]
        out["pitch_track_ball_rows"] = len(ball_rows)
        out["pitch_track_player_rows"] = len(rows) - len(ball_rows)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--tracks")
    ap.add_argument("--stats")
    ap.add_argument("--cal")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    args = ap.parse_args()
    result = diagnose(
        Path(args.pkl),
        Path(args.tracks) if args.tracks else None,
        Path(args.stats) if args.stats else None,
        args.cal,
        args.fps,
        args.width,
        args.height,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
