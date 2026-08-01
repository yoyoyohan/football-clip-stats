# Models

YOLO weights are **not** stored in this repo (too large for GitHub).

## Expected layout

```
models/best.pt          # main detector: ball, player, goalkeeper, referee, goalpost
```

Place your trained `best.pt` here (or at the repo root — `run_clip.py` defaults to `models/best.pt`).

## Training

See `training/KAGGLE_SOCCERNET_TRAIN.md` and the Kaggle notebooks under `training/`.
