# Ground-truth evaluation set

Keep this set **fixed**. Do not swap clips after every experiment.

`team0` = Team A (brighter kit). `team1` = Team B (darker kit).
Use numbers only (`100`, not `100%`). Leave a field `null` until you have watched that clip.

## El Clásico demo

`elclasico.json` is a short sanity check, not the generalization test.

## Leve clips

One file per clip in this folder: `clip_01.json` … `clip_25.json`.
Videos live at `Leve_singlecam_clips/clip_XX.mp4`.

Edit the nulls after you watch:

- `possession.team0_pct` / `team1_pct`
- `passes.team0` / `team1`
- `shots.team0` / `team1`
- `goals.team0` / `team1`

If one team completed a passing sequence with no interception, that team can be 100% possession.

## Run

```bash
python scripts/evaluate_clip.py \
  --source input_videos/elclasico.mp4 \
  --gt evaluation/gt/elclasico.json

python scripts/evaluate_clip.py \
  --source Leve_singlecam_clips/clip_01.mp4 \
  --gt evaluation/gt/clip_01.json
```
