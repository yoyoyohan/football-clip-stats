#!/usr/bin/env python3
"""Overnight: re-detect clips 11/13/15 (+ El Clásico) on both checkpoints.

Writes reports under output_videos/overnight/{best,bak}/ and a comparison JSON.
Does not overwrite models/best.pt.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIPS = [
    ("Leve_singlecam_clips/clip_11.mp4", "evaluation/gt/clip_11.json"),
    ("Leve_singlecam_clips/clip_13.mp4", "evaluation/gt/clip_13.json"),
    ("Leve_singlecam_clips/clip_15.mp4", "evaluation/gt/clip_15.json"),
    ("input_videos/elclasico.mp4", "evaluation/gt/elclasico.json"),
]
MODELS = {
    "best": "models/best.pt",
    "bak": "models/best.pt.bak_pre_kaggle_finetune",
}


def pick_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def main() -> None:
    device = pick_device()
    summary: dict = {"device": device, "runs": []}
    for model_name, weights in MODELS.items():
        w = ROOT / weights
        if not w.exists():
            print(f"SKIP missing {w}", flush=True)
            continue
        out_dir = ROOT / "output_videos" / "overnight" / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for source, gt in CLIPS:
            src = ROOT / source
            if not src.exists():
                print(f"SKIP missing {src}", flush=True)
                continue
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "evaluate_clip.py"),
                "--source",
                str(src),
                "--model",
                str(w),
                "--gt",
                str(ROOT / gt),
                "--out-dir",
                str(out_dir),
                "--no-cache",
                "--device",
                device,
            ]
            print("RUN", " ".join(cmd), flush=True)
            proc = subprocess.run(cmd, cwd=str(ROOT))
            stem = src.stem
            report = out_dir / f"{stem}_eval_report.txt"
            stats_path = out_dir / f"{stem}_stats.json"
            row = {
                "model": model_name,
                "weights": weights,
                "clip": stem,
                "returncode": proc.returncode,
            }
            if stats_path.exists():
                stats = json.loads(stats_path.read_text())
                row["possession"] = stats.get("possession")
                row["passes"] = stats.get("passes")
                row["shots"] = stats.get("shots")
                row["goals"] = stats.get("goals")
                row["ball_cv"] = stats.get("ball_cv")
                row["gt"] = stats.get("ground_truth_comparison")
            if report.exists():
                print(report.read_text(), flush=True)
            summary["runs"].append(row)
    out = ROOT / "output_videos" / "overnight" / "comparison.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("WROTE", out, flush=True)


if __name__ == "__main__":
    main()
