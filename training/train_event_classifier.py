#!/usr/bin/env python3
"""Train hybrid event classifiers from labeled candidate CSVs.

Label format (pass_labels.csv):
  frame,release_frame,from_track_id,to_track_id,team_id,label
  58,24,6,3,0,1
  131,52,14,9,0,0   # negative example

Generate candidates + features first:
  python training/export_event_candidates.py --source input_videos/elclasico.mp4

Then train:
  python training/train_event_classifier.py --labels training/labels/pass_labels.csv --event pass
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from analysis.event_features import PassCandidateFeatures


def load_labeled_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"No rows in {path}")

    feature_rows = []
    labels = []
    for row in rows:
        if "features_json" in row:
            import json

            feats = json.loads(row["features_json"])
            feature_rows.append(feats)
        else:
            f = PassCandidateFeatures(
                frame=int(row["frame"]),
                release_frame=int(row.get("release_frame", row["frame"])),
                fps=float(row.get("fps", 30)),
                frame_width=float(row.get("frame_width", 1280)),
                from_track_id=int(row.get("from_track_id", 0)),
                to_track_id=int(row.get("to_track_id", 0)),
                team_id=int(row.get("team_id", 0)),
                ball_speed_peak=float(row.get("ball_speed_peak", 0)),
                ball_travel_px=float(row.get("ball_travel_px", 0)),
                passer_receiver_dist_px=float(row.get("passer_receiver_dist_px", 0)),
                control_streak=int(row.get("control_streak", 0)),
                possession_dwell_frames=int(row.get("possession_dwell_frames", 0)),
                flight_frames=int(row.get("flight_frames", 0)),
                ball_observed_fraction=float(row.get("ball_observed_fraction", 0)),
                players_near_ball=int(row.get("players_near_ball", 0)),
                recv_mode=row.get("recv_mode", "control"),
                geometry_score=float(row.get("geometry_score", 1)),
            )
            feature_rows.append(f.to_vector())
        labels.append(int(row["label"]))
    return np.array(feature_rows, dtype=np.float32), np.array(labels, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser(description="Train hybrid event classifier")
    parser.add_argument("--labels", required=True, help="CSV with features + label column")
    parser.add_argument("--event", default="pass", choices=["pass", "goal", "shot"])
    parser.add_argument("--out", default="models/event_classifiers")
    args = parser.parse_args()

    x, y = load_labeled_csv(Path(args.labels))
    if len(np.unique(y)) < 2:
        raise SystemExit("Need both positive (1) and negative (0) labels to train.")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(random_state=42)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    print(classification_report(y_test, preds))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import joblib

        out_path = out_dir / f"{args.event}_classifier.joblib"
        joblib.dump(model, out_path)
        print(f"Wrote {out_path}")
    except ImportError:
        raise SystemExit("Install joblib: pip install joblib")


if __name__ == "__main__":
    main()
