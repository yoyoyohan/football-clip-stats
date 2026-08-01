import argparse
import json
import pickle
from pathlib import Path

import cv2

from analysis import BallInterpolator, HybridEventClassifier, TeamColorAssigner
from stats_engine import StatEngine, default_field_polygon, default_goal_boxes, default_homography
from trackers import Tracker
from utils import read_video, save_video
from utils.calibration import default_calibration_path, load_calibration
from utils.goal_regions import (
    goal_mouth_lines_from_boxes,
    refine_goal_boxes_from_frames,
)


def _video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return float(fps)


def main():
    parser = argparse.ArgumentParser(description="Run stats pipeline on a match clip")
    parser.add_argument("--source", default="input_videos/elclasico.mp4")
    parser.add_argument(
        "--hybrid-events",
        action="store_true",
        help="Use trained ML classifiers in models/event_classifiers/ to confirm events",
    )
    args = parser.parse_args()

    video_path = args.source
    video_stem = Path(video_path).stem
    track_cache = Path(f"output_videos/{video_stem}_frame_records.pkl")

    video_frames = read_video(video_path)
    h, w = video_frames[0].shape[:2]
    fps = _video_fps(video_path)

    if track_cache.exists():
        print(f"Loading cached tracks from {track_cache}")
        with track_cache.open("rb") as f:
            frame_records = pickle.load(f)
    else:
        print(f"Running YOLO + tracking on {video_path} ({len(video_frames)} frames)...")
        tracker = Tracker("models/best.pt")
        frame_records = tracker.get_object_tracks(video_frames)
        track_cache.parent.mkdir(parents=True, exist_ok=True)
        slim = []
        for record in frame_records:
            slim.append(
                {
                    "frame_idx": record["frame_idx"],
                    "detections": record["detections"],
                    "ball": record["ball"],
                    "overlay": record.get("overlay", {}),
                }
            )
        with track_cache.open("wb") as f:
            pickle.dump(slim, f)
        print(f"Cached tracks to {track_cache}")

    # --- 1. Jersey KMeans team classification ---
    color_assigner = TeamColorAssigner(barcelona_team_id=0)
    for record in frame_records:
        color_assigner.collect(video_frames[record["frame_idx"]], record["detections"])
    color_assigner.fit_teams()
    print("Teams assigned:", len(color_assigner._track_teams), "tracks")

    ball_interpolator = BallInterpolator(fps=fps)

    cal_path = default_calibration_path(video_path)
    cal = load_calibration(cal_path, w, h)
    if cal:
        print(f"Using calibration: {cal_path}")
        homography = cal["homography"]
        goal_boxes = cal["goal_boxes"]
        field_polygon = cal["field_polygon"]
        attacking_direction = cal["attacking_direction"]
        if cal.get("fps"):
            fps = float(cal["fps"])
    else:
        print(f"No calibration at {cal_path} — using defaults")
        homography = default_homography(w, h)
        goal_boxes = default_goal_boxes(w, h)
        field_polygon = default_field_polygon(w, h)
        attacking_direction = "left_to_right"

    goalpost_boxes = refine_goal_boxes_from_frames(frame_records, w)
    if goalpost_boxes:
        print("Goal regions from detected goalposts")
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

    if args.hybrid_events:
        classifier = HybridEventClassifier()
        engine.pass_detector.event_classifier = classifier
        if classifier.has_pass_model:
            print("Hybrid pass classifier loaded")
        else:
            print("No pass classifier at models/event_classifiers/ — geometry only")

    for record in frame_records:
        idx = record["frame_idx"]
        frame = video_frames[idx]
        detections = color_assigner.assign_teams(frame, record["detections"])
        ball_obs = ball_interpolator.update(idx, record["ball"])
        engine.update(idx, detections, ball_obs.position, ball_observed=ball_obs.observed)

    stats = engine.get_stats()
    stats["team_names"] = {"team0": "barcelona", "team1": "real_madrid"}
    stats["goal_lines"] = goal_lines

    stats_path = Path(f"output_videos/{video_stem}_stats.json")
    csv_path = Path(f"output_videos/{video_stem}_stats_per_frame.csv")
    out_video = Path(f"output_videos/{video_stem}_annotated.mp4")

    Path("output_videos").mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    engine.export_csv(str(csv_path))
    save_video(video_frames, str(out_video))

    print(f"\nWrote {stats_path}")
    print("\n=== Results ===")
    print("Possession:", stats["possession"])
    print("Passes:", stats["passes"])
    print("Pass events:", len(stats["pass_events"]))
    for i, e in enumerate(stats["pass_events"], 1):
        t = e["frame"] / fps
        print(
            f"  #{i} recv {t:.2f}s (frame {e['frame']})  "
            f"{e['from_track_id']}->{e['to_track_id']}  "
            f"release {e['release_frame']/fps:.2f}s"
        )
    print("Goals:", stats["goals"])
    print("Goal events:", stats["goal_events"])
    print("Shots:", stats.get("shots", {}))
    print("Shots on target:", stats.get("shots_on_target", {}))


if __name__ == "__main__":
    main()
