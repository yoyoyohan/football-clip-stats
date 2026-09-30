# Evaluation

Frozen labels for whether a change actually helps. Do not swap clips after every experiment.

- Ground truth JSON: [`gt/`](gt/)
- Dataset/training notes: [`AUDIT.md`](AUDIT.md), [`REPORT.md`](REPORT.md)

```bash
python scripts/evaluate_clip.py \
  --source input_videos/yourclip.mp4 \
  --gt evaluation/gt/elclasico.json
```

`team0` = brighter kit, `team1` = darker kit. Use `100`, not `100%`.
