# Stats Pipeline — Run Locally

**Use this project locally, not Kaggle**, for analyzing clips. Kaggle is only for retraining `best.pt`.

## Architecture

```
Video clip
  → YOLO (models/best.pt) + ByteTrack     # detect players, ball, goalposts
  → TeamColorAssigner (jersey KMeans)     # team 0 / team 1
  → Homography (calibration/*.json)       # pixel → pitch meters (105×68)
  → StatEngine
       ├── possession (who's closest to ball)
       ├── passes (geometry + speed/distance in meters)
       ├── shots + shots on target (pitch position + ball speed m/s)
       └── goals (goal-line crossing)
```

**Pitch coordinates** use a homography (standard 2D broadcast analytics). Full 3D reconstruction is not needed for passes, shots, or possession zones.

## Step-by-step

### 1. Put your clip in `input_videos/`
```bash
cp myclip.mp4 input_videos/myclip.mp4
```

### 2. Calibrate the pitch (once per video)
**Follow [CALIBRATION.md](CALIBRATION.md)** — scrub the video, pick a frame, click landmarks.

Short version:
```bash
python calibrate_pitch.py --source input_videos/myclip.mp4
# scrub → SPACE → click goal posts (or use --mode landmarks) → press c
```
Saves `calibration/myclip.json`.

### 3. Run the full pipeline
```bash
python run_clip.py --source input_videos/myclip.mp4 --no-cache --team0-name Morocco --team1-name France
```

### 4. Outputs in `output_videos/`

| File | Contents |
|------|----------|
| `{clip}_stats.json` | Passes, shots, SOT, goals, possession, events with **meters & m/s** |
| `{clip}_stats_per_frame.csv` | Per-frame possession %, ball `x_m`/`y_m` |
| `{clip}_pitch_tracks.csv` | Every player + ball position in **pitch meters** per frame |
| `{clip}_frame_records.pkl` | Cached YOLO tracks (delete to re-detect) |

## Stats included

- **Possession** — % by team (jersey color clustering)
- **Passes** — count + per-event `distance_m`, `speed_mps`, pitch coordinates
- **Shots** — ball toward goal above speed threshold in attacking third
- **Shots on target** — ball position inside goal mouth (7.32m wide)
- **Goals** — ball crosses goal line

## When to use Kaggle

| Task | Where |
|------|-------|
| Analyze a clip | **Here** — `run_clip.py` |
| Retrain YOLO (better ball detection) | Kaggle — `training/kaggle_soccernet_train.ipynb` |
| Train event classifier (optional) | **Here** — `training/train_event_classifier.py` |

## Optional: hybrid AI event filter

```bash
python training/export_event_candidates.py --source input_videos/elclasico.mp4
# label CSV, then:
python training/train_event_classifier.py --labels training/labels/pass_labels.csv --event pass
python main.py --source input_videos/moroccomatch.mp4 --hybrid-events
```
