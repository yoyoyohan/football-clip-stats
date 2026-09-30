#!/usr/bin/env python3
"""Fine-tune models/best.pt (YOLOv9c, 5 classes) on training/datasets/finetune800.

This is NOT a YOLO26 rewrite. Same architecture, lower LR, few epochs.

Local (slow / no CUDA on this Mac):
  python training/finetune_best.py --data training/datasets/finetune800/data.yaml --weights models/best.pt

Kaggle (preferred):
  upload finetune800 + best.pt, then run the same flags with --kaggle
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="training/datasets/finetune800/data.yaml")
    parser.add_argument("--weights", default="models/best.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr0", type=float, default=1e-4)
    parser.add_argument("--project", default="runs/finetune800")
    parser.add_argument("--name", default="bestpt_800")
    parser.add_argument("--kaggle", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.kaggle:
        args.project = "/kaggle/working/runs"
        out_copy = Path("/kaggle/working/best.pt")
    else:
        out_copy = Path("models/best_finetune800.pt")

    data = Path(args.data)
    weights = Path(args.weights)
    if not data.exists():
        raise SystemExit(
            f"Missing {data}. Run:\n"
            "  python training/prepare_finetune800.py --src /path/to/your/800-image-export"
        )
    if not weights.exists():
        raise SystemExit(f"Missing {weights}")

    model = YOLO(str(weights))
    print("Starting from", weights, "classes", model.names)
    try:
        model.train(
            data=str(data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            lr0=args.lr0,
            lrf=0.01,
            optimizer="AdamW",
            cos_lr=True,
            close_mosaic=8,
            amp=True,
            device=args.device,
            project=args.project,
            name=args.name,
            exist_ok=True,
            seed=42,
            plots=True,
            save=True,
        )
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        print("OOM — retry batch 4")
        model.train(
            data=str(data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=4,
            patience=args.patience,
            lr0=args.lr0,
            lrf=0.01,
            optimizer="AdamW",
            cos_lr=True,
            close_mosaic=8,
            amp=True,
            device=args.device,
            project=args.project,
            name=args.name,
            exist_ok=True,
            seed=42,
            plots=True,
            save=True,
        )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    if best.exists():
        out_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, out_copy)
        print("Saved", out_copy)
        print("Do NOT overwrite models/best.pt until clips 11/13/15 improve.")
    else:
        raise SystemExit("Training finished but best.pt was not written")


if __name__ == "__main__":
    main()
