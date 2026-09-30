# Evaluation report

This is the college-demo snapshot after the audit. It is written to be
defensible, not flattering.

## A. Dataset

**Current:** `training/datasets/soccernet5`, 5 classes
(`ball`, `player`, `goalkeeper`, `referee`, `goalpost`).
Roboflow-style frames. 4811 / 580 / 370 images.

**Problems**

- Goalpost boxes: 263 / 73 / 16. That is not enough for a reliable structure class.
- 253 sequence keys leak across train/valid/test. Reported mAP is not a
  generalization number.
- Player boxes are ~85% of all labels. A high mAP can hide a dead ball/goal head.
- Empty label files: 426 train, 118 valid, 60 test.

**Changes made:** none to the images. Added `scripts/audit_dataset.py` so the
counts can be regenerated. **Do not start the next train until a
sequence-level split exists.**

**Why a new split would be better:** entire clips stay together; the test set
is footage the model has never seen; goalmouth / ball-in-motion frames can be
up-weighted instead of cloning easy midfield frames.

Ideal next dataset (structure, not a rewrite of the loader):

- Split by match / video / sequence
- Keep current 5 classes
- Add hard goalpost views (both ends, partial posts, players in front)
- Add ball-in-flight, ball-at-feet, ball-in-goal-mouth
- Cap near-duplicate consecutive frames

## B. Training

**No new training run in this pass.** That was intentional.

| Field | Last overnight attempt |
| --- | --- |
| Base | fine-tune `pretrained_best.pt` (YOLOv9c kernel name) |
| Dataset | soccernet5 as uploaded to Kaggle |
| Image size | 960 |
| Batch | 8 |
| Epochs / patience | 50 / 20 |
| Optimizer / lr | Ultralytics default SGD, `lr0=0.001` |
| Checkpoint used at runtime | `models/best.pt` (49 MB, 16 Sep) |
| Previous checkpoint | `models/best.pt.bak_pre_kaggle_finetune` (147 MB) |

**Best checkpoint is unknown** until both files are scored on the same
held-out clip. If the Sep 16 model looks worse on ball/goalposts, restore the
`.bak` file as the baseline instead of training again.

Proposed experiments (only after the split is fixed):

1. **A — baseline:** current architecture + cleaned sequence split.
2. **B — augmentation:** same weights, mosaic/scale that actually help small balls
   and distant posts.
3. **C:** only if A/B show a specific failure (e.g. still no goalposts).

## C. Detection evaluation

Official precision / recall / mAP50 / mAP50-95 were **not** recomputed here.
The local valid set leaks train sequences, so a fresh `yolo val` on it would
be an optimistic number, not a trustworthy one.

Proxy from the pipeline (needs `--no-cache` so detections carry confidence):

| Class | What we can say now |
| --- | --- |
| Player | Dominant class; detections are usable for tracking |
| Ball | Present often enough for pass logic on the short demo; still the fragile class |
| Goalpost | Structurally under-represented; do not trust single-frame posts |

Run after a sequence split:

```text
yolo val model=models/best.pt data=training/datasets/soccernet5/data.yaml
```

and the same command on `.bak_pre_kaggle_finetune`. Put both rows in the
checkpoint table below.

| Model | Player | Ball | Goal | Notes |
| --- | --- | --- | --- | --- |
| `best.pt.bak_pre_kaggle_finetune` | not re-scored | not re-scored | not re-scored | Keep as fallback |
| `models/best.pt` (16 Sep) | not re-scored | not re-scored | not re-scored | Active; unproven vs bak |

## D. Soccer analytics evaluation

Same held-out **event** set: `evaluation/gt/elclasico.json`.

| Statistic | Ground truth | This re-run (`evaluate_clip.py` on Aug 15 cache) | Notes |
| --- | --- | --- | --- |
| Passes team0 / team1 | 4 / 0 | 0 / 1 | Does **not** reproduce the older 4-pass JSON. Team labels are brightness-based (can swap A/B). PassDetector is independent of the new dwell. |
| Shots | 0 / 0 | 0 / 0 | No shot in this clip |
| Goals | 0 / 0 | 0 / 0 | |
| Possession | not yet hand-labeled | 8.2% / 60.2% / 31.6% loose | Better than the stale 100% team0 file. Still not a verified % |
| Players / frame | — | 8.32 | Cache has no detector confidence |
| Ball frame recall | — | 197 / 342 (0.58) | Proxy only |
| Tracking | not scored (IDSW) | ByteTrack | Circles follow the track foot-point |

An older `elclasico_stats.json` on this branch reported 4 team0 passes. That file is **not** the current engine output. Trust the script, not a leftover JSON. Do not retune pass gates to recover 4 on this 10 s clip.

**Reliable today (on short, similar broadcast clips):** player presence, pass
*counts* when the ball is visible, team color on clearly different kits.

**Not reliable yet:** possession percentages on long/idle video, shots, goals,
goalpost-driven goal lines, pitch meters when homography collapses (ball
pinned near x=105 m on the demo).

## E. Video demo

```bash
python scripts/evaluate_clip.py \
  --source input_videos/elclasico.mp4 \
  --gt evaluation/gt/elclasico.json \
  --model models/best.pt
```

Writes:

- `output_videos/elclasico_eval.mp4` — player circles + detector confidence,
  ball box + confidence, goalposts, IDs, teams, possession HUD
- `output_videos/elclasico_eval_report.txt`

`yolo_inference.py` uses the same overlay (no `result.plot()` boxes on players).

## F. Recommendation

1. **What was wrong.** The last train used a leaky frame split and almost no
   goalposts. mAP could rise while soccer stats stayed weak. Possession was
   closest-player every frame. Players were drawn as boxes. Confidence was
   dropped on the detection dict. ByteTrack ignored real fps. The 10 s demo
   was treated like a final exam.
2. **What changed.** Audit + dataset counts. Detector confidence kept.
   Player foot-circles + ball boxes. Possession dwell (0.12 s). ByteTrack
   fps. Evaluation script + frozen GT template. No retrain.
3. **Is the new model better?** We did not ship a new model. The Sep 16
   checkpoint is not certified better than the backup.
4. **Reliable stats:** pass *count* on the short demo; player/team labels
   when kits differ.
5. **Still weak:** goalposts, goals, shots, long-clip possession, meters
   under a bad homography.
6. **Next bottleneck:** dataset split + goalpost/ball coverage. Then score
   `best.pt` vs `.bak` on the same Leve clips. Only then train experiment A.

Do not optimize for a prettier mAP on `soccernet5/valid`.
