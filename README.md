# Football Clip Stats Pipeline

End-to-end computer vision pipeline that turns a short broadcast football clip into match stats:

**possession · passes · shots · shots on target · goals**

Built with **YOLOv8 + ByteTrack**, jersey color clustering, pitch homography, and geometry-based event detection.

> **Note for reviewers:** Sample match videos and model weights are not in this repo (size + copyright). See setup below.

---

## Architecture

```
Video clip
  → YOLO (models/best.pt) + ByteTrack     # players, ball, goalposts
  → TeamColorAssigner (jersey KMeans)     # team 0 / team 1
  → Homography (calibration/*.json)       # pixels → pitch meters (105×68)
  → StatEngine
       ├── possession
       ├── passes (distance_m, speed_mps)
       ├── shots + shots on target
       └── goals
```

## Quick start

```bash
# 1. Install
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Add your detector weights
#    Place trained YOLO weights at: models/best.pt
#    (classes: ball, player, goalkeeper, referee, goalpost)

# 3. Add a short clip
mkdir -p input_videos
# copy your .mp4 into input_videos/

# 4. Calibrate pitch (scrub video → click landmarks → save)
python calibrate_pitch.py --source input_videos/yourclip.mp4
# Full guide: CALIBRATION.md

# 5. Run stats
python run_clip.py --source input_videos/yourclip.mp4 --no-cache
```

Outputs land in `output_videos/`:

| File | Contents |
|------|----------|
| `{clip}_stats.json` | Passes, shots, SOT, goals, possession |
| `{clip}_stats_per_frame.csv` | Possession + ball pitch coords |
| `{clip}_pitch_tracks.csv` | Player/ball positions in meters |

## Project layout

```
run_clip.py           # main CLI (detect → stats)
calibrate_pitch.py    # interactive pitch calibration
stats_engine.py       # possession / pass / shot / goal logic
analysis/             # detectors, interpolator, team colors, pitch coords
trackers/             # YOLO + ByteTrack wrapper
utils/                # calibration, video I/O, goal regions
training/             # Kaggle train notebooks + SoccerNet converter
tests/                # unit tests
CALIBRATION.md        # step-by-step calibration guide
PIPELINE.md           # pipeline overview
```

## Calibration

Broadcast clips rarely show the full pitch. Calibration supports:

- **`--mode goal`** — one goal visible (2 posts; default)
- **`--mode landmarks`** — 4+ pitch marks when no goal is in frame
- **`--mode corners`** — full pitch visible

See **[CALIBRATION.md](CALIBRATION.md)** for the full user flow.

## Current limitations (honest)

- Accuracy depends heavily on **ball detection** quality
- Tight midfield shots (no goal) weaken shot/goal geometry
- Jersey clustering can skew possession on similar kits
- Best on **short clips** (10–30s), not full matches yet

## Tests

```bash
pytest -q
```

## License

MIT — see [LICENSE](LICENSE).
