# Kaggle: Retrain YOLO with Goalpost Detection

Use **`training/kaggle_goalpost_train.ipynb`** — this fixes the bug where goalposts were never included in training data.

## Why retrain?

SoccerNet stores goalposts as **line** annotations, not bounding boxes. The old converter skipped them, so your model's `goalpost` class was never learned.

## Kaggle setup (one time)

1. Go to [kaggle.com](https://kaggle.com) → **New Notebook**
2. **Settings** → Accelerator: **GPU T4 x2**
3. **Settings** → Internet: **On**
4. **Add Input** → **Models** → upload `models/best.pt` from your Mac
5. **File** → **Upload notebook** → choose `training/kaggle_goalpost_train.ipynb`
6. Click **Run All**

## What each cell does

| Cell | Action | What to check |
|------|--------|---------------|
| 1 | `pip install` | No errors |
| 2 | Download 20 SoccerNet games | ~2 GB, finishes in ~10 min |
| 3 | Write fixed converter | Prints `Wrote /kaggle/working/soccernet_to_yolo.py` |
| 4 | Convert to YOLO + **verify goalpost labels** | Must print `OK: XXXX goalpost labels found` — if 0, **stop** |
| 5 | Train 50 epochs | ~45–90 min on GPU |
| 6 | Validate + download link | `goalpost` mAP50 should be > 0 |

## After Kaggle finishes

```bash
# 1. Download best.pt from Kaggle Output tab (cell 6 link)

# 2. Replace your local model
cp ~/Downloads/best.pt ~/Desktop/statsapp/models/best.pt

# 3. Test detection on video
cd ~/Desktop/statsapp
python yolo_inference.py
# Watch: output_videos/yolo_annotated.mp4 — should show "goalpost" YOLO boxes

# 4. Run full stats
python main.py
```

## Optional: pitch calibration (for stats accuracy)

Goalpost **detection** comes from the new model. Goal **regions for shots** can still use calibration:

```bash
python calibrate_pitch.py --source input_videos/elclasico.mp4 --frame 50
# Click 4 pitch corners, press 'c' to save
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ZERO goalpost labels` in cell 4 | Re-run cell 3 (converter) then cell 4 |
| `No best.pt in /kaggle/input` | Add your model under Add Input → Models |
| Out of disk | Lower `MAX_GAMES` to 15 in cell 2 |
| goalpost mAP50 still 0 after train | Increase `MAX_GAMES` to 40+ and re-run |
