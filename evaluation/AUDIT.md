# System audit (no retrain)

Inspected the existing detection → tracking → team → possession → event pipeline
before changing visualization and evaluation. Working event logic was left in place.

## Pipeline

```
VIDEO
  → YOLO (models/best.pt)          # 5 classes, conf=0.1
  → ByteTrack                      # players / GK / referee / ball
  → TeamColorAssigner              # torso HSV/LAB KMeans
  → Homography (calibration JSON)  # 105 × 68 m
  → StatEngine
       ├── possession (closest foot + temporal dwell)
       ├── PassDetector (release → flight → same-team receive)
       ├── ShotDetector (speed + attacking third + goal direction)
       └── GoalDetector (goal-line / goal-box cross)
  → stats JSON + evaluation video
```

The detector and the event engine are already separate. Do not couple them.

## Detector

| Item | Current |
| --- | --- |
| Architecture | Ultralytics YOLO (Kaggle run named `kaggle_finetune_v9c`; README still says YOLOv8) |
| Active weights | `models/best.pt` (49 MB, 16 Sep) |
| Previous weights | `models/best.pt.bak_pre_kaggle_finetune` (147 MB, same date stamp as the swap) |
| Older archives | `newbest/best.zip` (13 Aug), `newbest/best (1).zip` (14 Aug), repo-root `best.pt` (147 MB, 15 Aug) |
| Classes | `ball`, `player`, `goalkeeper`, `referee`, `goalpost` |
| Inference conf | 0.1 (tracker + `yolo_inference.py`) |
| Train image size | 960 (overnight Kaggle) / 640 (docs) |
| Batch | 8 (falls to 4 on OOM) |
| Epochs / patience | 50 / 20 |
| LR | `lr0=0.001` from `pretrained_best.pt` |
| Dataset | `training/datasets/soccernet5` (Roboflow-style remap) |

**Newest ≠ best.** The Sep 16 49 MB file is the active runtime checkpoint. There is
no held-out mAP table that proves it beats the 147 MB pre-finetune weights.
Do not delete the `.bak` file. Compare both on the same clip before the next train.

## Dataset (local `soccernet5`)

| Split | Images | Empty labels | Ball | Player | GK | Referee | Goalpost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 4811 | 426 | 3020 | 83004 | 2745 | 9049 | **263** |
| valid | 580 | 118 | 383 | 8619 | 276 | 927 | **73** |
| test | 370 | 60 | 265 | 6049 | 211 | 671 | **16** |

Problems found:

- **Goalposts are rare.** Test has 16 boxes. That is the main reason goal
  structure detection is weak — not the architecture.
- **Players dominate.** Any mAP is mostly a player-box score.
- **Split leakage.** 253 sequence keys appear in more than one of train/valid/test
  (same clip, different Roboflow hashes). Validation is optimistic.
- **Empty frames** are common (no objects). Fine if they are true negatives;
  they do not teach goalmouths.
- Near-duplicate consecutive frames from the same sequence inflate size.

Do **not** retrain on this split as-is. Rebuild a match/sequence split first.

## Tracking / teams / events

| Component | Status | Keep? |
| --- | --- | --- |
| ByteTrack | Works; was constructed at default 30 fps | Yes. Now receives real video fps + ~1s lost-track buffer. |
| Team KMeans | Torso crop, dark-kit features | Yes. Not whole-box grass clustering. |
| Possession | Was closest-player every frame; idle already counted as loose | Dwell of 0.12 s added so A↔B flicker does not flip the team. |
| Passes | Flight + same-team receive, fps-scaled | Keep. El Clásico demo: 4 team0 vs hand-count 4. |
| Shots | Speed + attacking third + path-to-goal | Keep. Homography errors still create false meters. |
| Goals | Line/box cross, multi-frame | Keep. Needs better goalposts + calibration, not a new detector class. |
| Ball interpolator | Already separates observed vs smoothed | Keep. Overlay draws a box only on observed detections. |

## Visualization (before this change)

`yolo_inference.py` used `result.plot()` — rectangular boxes for every class.
Player confidence was not stored on detection dicts, so any on-screen score
would have been invented or missing.

## What was left alone

Pass, shot, goal, and team-assignment algorithms were not rewritten.
No architecture change. No overnight retrain.

## Held-out test footage

- Sanity: `input_videos/elclasico.mp4` + `evaluation/gt/elclasico.json`
- Generalization candidates: `Leve_singlecam_clips/clip_*.mp4` (18 s, mixed phases)

Do not retune against the 10-second demo after every change.
