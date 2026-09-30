#!/usr/bin/env python3
"""Produce an annotated evaluation video + text report for a clip.

Reuses the existing YOLO → ByteTrack → teams → StatEngine pipeline.
Does not retrain. Cached tracks are reused unless --no-cache is set.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis import BallInterpolator, TeamColorAssigner
from analysis.overlay import annotate_frame
from main import _video_fps
from run_clip import export_pitch_tracks
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
from utils.detection_utils import normalize_ball
from utils.goal_regions import goal_mouth_lines_from_boxes, refine_goal_boxes_from_frames


def _mean_confidence(frame_records: list[dict], class_name: str) -> float | None:
    values = []
    for rec in frame_records:
        for det in rec.get("detections", []):
            if det.get("class_name") != class_name:
                continue
            conf = det.get("confidence")
            if conf is not None:
                values.append(float(conf))
    if not values:
        return None
    return sum(values) / len(values)


def _ball_frame_stats(frame_records: list[dict]) -> dict:
    detected = 0
    confs = []
    for rec in frame_records:
        ball = rec.get("ball")
        pos, _conf, _area = normalize_ball(ball)
        if pos is None:
            continue
        detected += 1
        if isinstance(ball, dict) and ball.get("confidence") is not None:
            confs.append(float(ball["confidence"]))
    n = max(1, len(frame_records))
    return {
        "frames_with_ball": detected,
        "frame_recall": detected / n,
        "mean_confidence": (sum(confs) / len(confs)) if confs else None,
    }


def _player_frame_stats(frame_records: list[dict]) -> dict:
    counts = []
    confs = []
    for rec in frame_records:
        players = [
            d
            for d in rec.get("detections", [])
            if d.get("class_name") in {"player", "goalkeeper"}
        ]
        counts.append(len(players))
        for d in players:
            if d.get("confidence") is not None:
                confs.append(float(d["confidence"]))
    return {
        "mean_players_per_frame": (sum(counts) / len(counts)) if counts else 0.0,
        "mean_confidence": (sum(confs) / len(confs)) if confs else None,
    }


def compare_ground_truth(pred: dict, gt: dict) -> dict:
    out: dict = {}
    poss_gt = gt.get("possession") or {}
    poss_pr = pred.get("possession") or {}
    if poss_gt.get("team0_pct") is not None:
        out["possession_team0_error_pp"] = abs(
            float(poss_pr.get("team0_pct", 0)) - float(poss_gt["team0_pct"])
        )
    if poss_gt.get("team1_pct") is not None:
        out["possession_team1_error_pp"] = abs(
            float(poss_pr.get("team1_pct", 0)) - float(poss_gt["team1_pct"])
        )

    def _count_err(section: str, key0: str, key1: str, gt0: str, gt1: str):
        g = gt.get(section) or {}
        p = pred.get(section) or {}
        if g.get(gt0) is not None:
            out[f"{section}_team0_pred"] = p.get(key0, 0)
            out[f"{section}_team0_gt"] = g[gt0]
            out[f"{section}_team0_abs_err"] = abs(int(p.get(key0, 0)) - int(g[gt0]))
        if g.get(gt1) is not None:
            out[f"{section}_team1_pred"] = p.get(key1, 0)
            out[f"{section}_team1_gt"] = g[gt1]
            out[f"{section}_team1_abs_err"] = abs(int(p.get(key1, 0)) - int(g[gt1]))

    _count_err("passes", "team0_passes", "team1_passes", "team0", "team1")
    _count_err("shots", "team0_shots", "team1_shots", "team0", "team1")
    _count_err("goals", "team0_goals", "team1_goals", "team0", "team1")
    return out


def format_report(
    stats: dict,
    frame_records: list[dict],
    *,
    video: str,
    model: str,
    gt_cmp: dict | None = None,
) -> str:
    names = stats.get("team_names") or {"team0": "Team A", "team1": "Team B"}
    poss = stats.get("possession") or {}
    passes = stats.get("passes") or {}
    shots = stats.get("shots") or {}
    goals = stats.get("goals") or {}
    players = _player_frame_stats(frame_records)
    ball = _ball_frame_stats(frame_records)
    pconf = players["mean_confidence"]
    bconf = ball["mean_confidence"]
    lines = [
        "VIDEO ANALYSIS",
        f"Video: {video}",
        f"Model: {model}",
        "",
        "Players detected:",
        f"Average players / frame: {players['mean_players_per_frame']:.2f}",
        f"Average player confidence: {pconf:.3f}" if pconf is not None else "Average player confidence: n/a (re-detect with --no-cache)",
        "",
        "Ball detection:",
        f"Frames with a ball: {ball['frames_with_ball']} / {len(frame_records)}",
        f"Frame recall: {ball['frame_recall']:.3f}",
        f"Average ball confidence: {bconf:.3f}" if bconf is not None else "Average ball confidence: n/a",
        "Precision: n/a (needs boxed ground truth)",
        "Recall: n/a (needs boxed ground truth; frame recall above is a proxy)",
        "",
        "Possession:",
        f"{names.get('team0', 'Team A')}: {poss.get('team0_pct', 0):.1f}%",
        f"{names.get('team1', 'Team B')}: {poss.get('team1_pct', 0):.1f}%",
        f"Loose / unknown: {poss.get('loose_pct', 0):.1f}%",
        "",
        "Passes:",
        f"{names.get('team0', 'Team A')}: {passes.get('team0_passes', 0)}",
        f"{names.get('team1', 'Team B')}: {passes.get('team1_passes', 0)}",
        "",
        "Shots:",
        f"{names.get('team0', 'Team A')}: {shots.get('team0_shots', 0)}",
        f"{names.get('team1', 'Team B')}: {shots.get('team1_shots', 0)}",
        "",
        "Goals:",
        f"{names.get('team0', 'Team A')}: {goals.get('team0_goals', 0)}",
        f"{names.get('team1', 'Team B')}: {goals.get('team1_goals', 0)}",
    ]
    if gt_cmp:
        lines += ["", "Ground-truth comparison:"]
        for key, value in gt_cmp.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.2f}")
            else:
                lines.append(f"  {key}: {value}")
    return "\n".join(lines) + "\n"


def analyze_clip(args):
    video_path = Path(args.source)
    stem = video_path.stem
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    track_cache = Path(args.cache) if getattr(args, "cache", None) else out_dir / f"{stem}_frame_records.pkl"

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
        tracker = Tracker(
            args.model,
            fps=fps,
            device=getattr(args, "device", None),
            ball_fallback=not getattr(args, "no_ball_fallback", False),
        )
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

    assigner = TeamColorAssigner()
    for rec in frame_records:
        assigner.collect(video_frames[rec["frame_idx"]], rec["detections"])
    assigner.fit_teams()

    cal_path = default_calibration_path(str(video_path))
    cal = load_calibration(cal_path, w, h)
    method = cal.get("method") if cal else None
    should_auto = cal is None or (is_auto_calibration(method) and args.no_cache)
    if should_auto:
        auto = robust_auto_calibration_from_frames(frame_records, w, h)
        if auto:
            auto["video"] = str(video_path)
            auto["fps"] = fps
            auto["attacking_direction"] = "left_to_right"
            save_calibration(cal_path, auto)
            cal = load_calibration(cal_path, w, h)
        elif not (cal and is_manual_calibration(method)):
            cal = None

    if cal:
        homography = cal["homography"]
        goal_boxes = cal["goal_boxes"]
        field_polygon = cal["field_polygon"]
        attacking_direction = cal["attacking_direction"]
        if cal.get("fps"):
            fps = float(cal["fps"])
    else:
        homography = default_homography(w, h)
        goal_boxes = default_goal_boxes(w, h)
        field_polygon = default_field_polygon(w, h)
        attacking_direction = "left_to_right"

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
    bi = BallInterpolator(fps=fps, frame_width=w, frame_height=h)
    smoothed_ball = bi.smooth_frame_records(frame_records)
    assigned = []
    for rec, obs in zip(frame_records, smoothed_ball):
        dets = assigner.assign_teams(video_frames[rec["frame_idx"]], rec["detections"])
        assigned.append(dets)
        engine.update(rec["frame_idx"], dets, obs.position, ball_observed=obs.observed)

    stats = engine.get_stats()
    stats["video"] = str(video_path)
    stats["fps"] = fps
    stats["frame_size"] = {"width": w, "height": h}
    stats["calibration"] = str(cal_path) if cal else None
    stats["team_names"] = {"team0": args.team0_name, "team1": args.team1_name}
    stats["model"] = args.model
    stats["player_cv"] = _player_frame_stats(frame_records)
    stats["ball_cv"] = _ball_frame_stats(frame_records)
    stats["goalpost_mean_confidence"] = _mean_confidence(frame_records, "goalpost")
    return stats, frame_records, video_frames, assigner, engine, assigned, smoothed_ball


def write_video(
    path: Path,
    video_frames: list,
    assigned: list[list[dict]],
    frame_records: list[dict],
    engine: StatEngine,
    stats: dict,
    fps: float,
    smoothed_ball: list | None = None,
) -> None:
    h, w = video_frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    log = {row["frame_idx"]: row for row in engine._frame_log}
    if smoothed_ball is None:
        smoothed_ball = [None] * len(frame_records)
    for rec, frame, dets, obs in zip(frame_records, video_frames, assigned, smoothed_ball):
        row = log.get(rec["frame_idx"], {})
        raw = rec.get("ball")
        observed = bool(getattr(obs, "observed", False) and raw)
        interpolated = bool(obs is not None and getattr(obs, "position", None) and not getattr(obs, "observed", False))
        if observed:
            ball = raw
        elif obs is not None and obs.position:
            ball = {"position": obs.position, "confidence": obs.confidence}
        else:
            ball = None
        possessor_id = row.get("possessor_track_id")
        possessor_team = None
        if possessor_id is not None:
            for det in dets:
                if det.get("track_id") == possessor_id:
                    possessor_team = det.get("team_id")
                    break
        annotated = annotate_frame(
            frame,
            dets,
            ball,
            possessor_track_id=possessor_id,
            possessor_team=possessor_team,
            stats=stats,
            team_names=stats.get("team_names"),
            ball_observed=observed,
            ball_interpolated=interpolated,
        )
        writer.write(annotated)
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a soccer clip and write video + report")
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--out-dir", default="output_videos")
    parser.add_argument("--gt", default=None, help="Optional ground-truth JSON")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--cache", default=None, help="Override frame_records pickle path")
    parser.add_argument("--device", default=None, help="YOLO device, e.g. mps / cpu / 0")
    parser.add_argument("--no-ball-fallback", action="store_true")
    parser.add_argument("--team0-name", default="Team A")
    parser.add_argument("--team1-name", default="Team B")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    video_path = Path(args.source)
    if not video_path.exists():
        sys.exit(f"Video not found: {video_path}")

    stats, frame_records, video_frames, assigner, engine, assigned, smoothed_ball = analyze_clip(args)
    stem = video_path.stem
    out_dir = Path(args.out_dir)
    gt_cmp = None
    if args.gt:
        gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
        gt_cmp = compare_ground_truth(stats, gt)
        stats["ground_truth_comparison"] = gt_cmp

    report = format_report(stats, frame_records, video=str(video_path), model=args.model, gt_cmp=gt_cmp)
    report_path = out_dir / f"{stem}_eval_report.txt"
    stats_path = out_dir / f"{stem}_stats.json"
    csv_path = out_dir / f"{stem}_stats_per_frame.csv"
    tracks_path = out_dir / f"{stem}_pitch_tracks.csv"
    video_out = out_dir / f"{stem}_eval.mp4"

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    engine.export_csv(str(csv_path))
    export_pitch_tracks(
        tracks_path,
        frame_records,
        video_frames,
        assigner,
        engine,
        BallInterpolator(fps=stats["fps"], frame_width=stats["frame_size"]["width"], frame_height=stats["frame_size"]["height"]),
    )
    report_path.write_text(report, encoding="utf-8")
    if not args.no_video:
        write_video(
            video_out,
            video_frames,
            assigned,
            frame_records,
            engine,
            stats,
            stats["fps"],
            smoothed_ball=smoothed_ball,
        )
        print(f"Wrote {video_out}")
    print(report)
    print(f"Wrote {report_path}")
    print(f"Wrote {stats_path}")


if __name__ == "__main__":
    main()
