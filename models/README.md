# Models

YOLO weights are **not** stored in git (too large for GitHub).

## Expected layout

```
models/best.pt                         # active detector (5 classes)
models/best.pt.bak_pre_kaggle_finetune # previous checkpoint — keep
```

Classes: `ball`, `player`, `goalkeeper`, `referee`, `goalpost`.

`run_clip.py` and `scripts/evaluate_clip.py` default to `models/best.pt`.

The newest file is not automatically the best file. Compare both checkpoints
on the same held-out clip before replacing the active weights.

## Training

See `training/KAGGLE_SOCCERNET_TRAIN.md` and `evaluation/REPORT.md`.
Do not start another overnight run until the dataset split is sequence-level.
