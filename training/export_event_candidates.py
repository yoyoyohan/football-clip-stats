#!/usr/bin/env python3
"""Export pass/goal candidate features for labeling and ML training.

Uses relaxed geometry to propose high-recall candidates, then writes CSV rows
you can label (1=real event, 0=false positive).
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

from analysis import BallInterpolator, TeamColorAssigner
from analysis.event_features import PassCandidateFeatures, count_players_near
from analysis.pass_detector import PassDetector
from utils import read_video


class CandidateLoggingPassDetector(PassDetector):
  """PassDetector that logs every geometry-qualified candidate before final filters."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.candidates: list[PassCandidateFeatures] = []
    self._candidate_context_frames = 0
    self._candidate_observed = 0

  def _log_candidate(
    self,
    frame_idx: int,
    ball: tuple[float, float],
    receiver: dict,
    recv_dist: float,
    mode: str,
    players: list[dict],
    *,
    accepted: bool,
  ) -> None:
    if self._passer_at_release is None or self._passer_team is None:
      return
    recv_pos = ((receiver["bbox"][0] + receiver["bbox"][2]) / 2, receiver["bbox"][3])
    passer_pos = self._passer_pos or recv_pos
    travel = 0.0
    if self._ball_at_release is not None:
      travel = ((ball[0] - self._ball_at_release[0]) ** 2 + (ball[1] - self._ball_at_release[1]) ** 2) ** 0.5
    pr_dist = ((recv_pos[0] - passer_pos[0]) ** 2 + (recv_pos[1] - passer_pos[1]) ** 2) ** 0.5
    feat = PassCandidateFeatures(
      frame=frame_idx,
      release_frame=self._release_frame,
      fps=self.fps,
      frame_width=self.frame_width,
      from_track_id=self._passer_at_release,
      to_track_id=receiver["track_id"],
      team_id=receiver["team_id"],
      ball_speed_peak=self._peak_speed,
      ball_travel_px=travel,
      passer_receiver_dist_px=pr_dist,
      control_streak=self._control_streak,
      flight_frames=max(0, frame_idx - self._release_frame),
      ball_observed_fraction=(
        self._flight_observations / max(1, frame_idx - self._release_frame)
      ),
      players_near_ball=count_players_near(players, ball, self._px(self.nearby_radius)),
      recv_mode=mode,
      geometry_score=1.0 if accepted else 0.6,
    )
    self.candidates.append(feat)

  def _try_pass_to(self, frame_idx, ball, receiver, recv_dist, mode, ball_observed):
    # Run parent logic
    event = super()._try_pass_to(frame_idx, ball, receiver, recv_dist, mode, ball_observed)
    self._log_candidate(
      frame_idx,
      ball,
      receiver,
      recv_dist,
      mode,
      [],
      accepted=event is not None,
    )
    return event


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--source", default="input_videos/elclasico.mp4")
  parser.add_argument("--out", default="training/labels/pass_candidates.csv")
  args = parser.parse_args()

  video_path = args.source
  stem = Path(video_path).stem
  cache = Path(f"output_videos/{stem}_frame_records.pkl")
  frames = read_video(video_path)
  h, w = frames[0].shape[:2]

  import cv2

  cap = cv2.VideoCapture(video_path)
  fps = float(cap.get(cv2.CAP_PROP_FPS) or 30)
  cap.release()

  with cache.open("rb") as f:
    records = pickle.load(f)

  assigner = TeamColorAssigner()
  for rec in records:
    assigner.collect(frames[rec["frame_idx"]], rec["detections"])
  assigner.fit_teams()

  det = CandidateLoggingPassDetector(fps=fps, frame_width=w)
  bi = BallInterpolator(fps=fps)

  for rec in records:
    idx = rec["frame_idx"]
    dets = assigner.assign_teams(frames[idx], rec["detections"])
    players = [
      d
      for d in dets
      if d.get("team_id") is not None and d.get("class_name") not in {"ball", "referee", "goalpost"}
    ]
    obs = bi.update(idx, rec.get("ball"))
    if obs.position is None:
      continue
    det.update(idx, obs.position, 0.0, players, ball_observed=obs.observed)

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(
      [
        "frame",
        "release_frame",
        "from_track_id",
        "to_track_id",
        "team_id",
        "ball_speed_peak",
        "ball_travel_px",
        "passer_receiver_dist_px",
        "control_streak",
        "flight_frames",
        "ball_observed_fraction",
        "players_near_ball",
        "recv_mode",
        "geometry_score",
        "features_json",
        "label",
      ]
    )
    for c in det.candidates:
      writer.writerow(
        [
          c.frame,
          c.release_frame,
          c.from_track_id,
          c.to_track_id,
          c.team_id,
          c.ball_speed_peak,
          c.ball_travel_px,
          c.passer_receiver_dist_px,
          c.control_streak,
          c.flight_frames,
          c.ball_observed_fraction,
          c.players_near_ball,
          c.recv_mode,
          c.geometry_score,
          json.dumps(c.to_vector()),
          "",
        ]
      )
  print(f"Wrote {len(det.candidates)} candidates to {out}")
  print("Fill the label column (1=pass, 0=not) then run training/train_event_classifier.py")


if __name__ == "__main__":
  main()
