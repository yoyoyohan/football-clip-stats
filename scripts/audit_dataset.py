#!/usr/bin/env python3
"""Summarize the local YOLO dataset: class counts and split leakage."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

NAMES = {0: "ball", 1: "player", 2: "goalkeeper", 3: "referee", 4: "goalpost"}


def _sequence_key(stem: str) -> str:
    for token in ("_jpg.rf.", "_png.rf.", "_jpg.rf", "_png.rf"):
        if token in stem:
            return stem.split(token)[0]
    return stem.rsplit(".", 1)[0]


def audit(root: Path) -> dict:
    report: dict = {"root": str(root), "splits": {}, "leak_keys": 0, "examples": []}
    stems: dict[str, list[str]] = defaultdict(list)
    for split in ("train", "valid", "test"):
        labels = list((root / split / "labels").glob("*.txt"))
        images = list((root / split / "images").glob("*"))
        counts = Counter()
        empty = 0
        for path in labels:
            lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
            if not lines:
                empty += 1
            for line in lines:
                counts[int(line.split()[0])] += 1
            stems[_sequence_key(path.stem)].append(split)
        report["splits"][split] = {
            "images": len(images),
            "labels": len(labels),
            "empty_labels": empty,
            "boxes": {NAMES.get(i, str(i)): counts[i] for i in range(5)},
            "total_boxes": int(sum(counts.values())),
        }
    leaks = {key: splits for key, splits in stems.items() if len(set(splits)) > 1}
    report["unique_sequence_keys"] = len(stems)
    report["leak_keys"] = len(leaks)
    report["examples"] = [(key, splits) for key, splits in list(leaks.items())[:8]]
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="training/datasets/soccernet5")
    args = parser.parse_args()
    report = audit(Path(args.root))
    print(f"Dataset: {report['root']}")
    for split, info in report["splits"].items():
        print(f"\n{split}: {info['images']} images, {info['empty_labels']} empty labels")
        for name, count in info["boxes"].items():
            print(f"  {name:12s} {count}")
        print(f"  total boxes  {info['total_boxes']}")
    print(f"\nUnique sequence keys: {report['unique_sequence_keys']}")
    print(f"Keys leaking across splits: {report['leak_keys']}")
    for key, splits in report["examples"]:
        print(f"  {key}: {splits}")


if __name__ == "__main__":
    main()
