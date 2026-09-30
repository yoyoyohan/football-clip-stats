#!/usr/bin/env python3
"""Turn a YOLO/Roboflow export into a 5-class fine-tune set for models/best.pt.

Expected class order (must match best.pt):
  0 ball, 1 player, 2 goalkeeper, 3 referee, 4 goalpost

Usage:
  python training/prepare_finetune800.py --src /path/to/your/export
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

TARGET_NAMES = ["ball", "player", "goalkeeper", "referee", "goalpost"]
NAME_ALIASES = {
    "ball": "ball",
    "soccer ball": "ball",
    "football": "ball",
    "player": "player",
    "person": "player",
    "goalkeeper": "goalkeeper",
    "gk": "goalkeeper",
    "keeper": "goalkeeper",
    "goalie": "goalkeeper",
    "referee": "referee",
    "ref": "referee",
    "official": "referee",
    "goalpost": "goalpost",
    "goal post": "goalpost",
    "goal": "goalpost",
    "post": "goalpost",
    "goalposts": "goalpost",
}


def _read_names(yaml_path: Path) -> list[str]:
    text = yaml_path.read_text(encoding="utf-8")
    names: list[str] = []
    in_names = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("names:"):
            rest = line[6:].strip()
            if rest.startswith("["):
                inner = rest.strip("[]")
                names = [p.strip().strip("'\"") for p in inner.split(",") if p.strip()]
                return names
            in_names = True
            continue
        if in_names:
            if not line or line.startswith("#") or ":" in line and not line[:1].isdigit() and not line.startswith("-"):
                if names:
                    break
            if line[:1].isdigit() and ":" in line:
                names.append(line.split(":", 1)[1].strip().strip("'\""))
            elif line.startswith("-"):
                names.append(line[1:].strip().strip("'\""))
    return names


def _sequence_key(stem: str) -> str:
    for token in ("_jpg.rf.", "_png.rf.", ".rf."):
        if token in stem:
            return stem.split(token)[0]
    parts = stem.replace("-", "_").split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return stem


def _iter_pairs(root: Path) -> list[tuple[Path, Path]]:
    images = []
    for split in ("train", "valid", "val", "test", ""):
        img_dir = root / split / "images" if split else root / "images"
        lbl_dir = root / split / "labels" if split else root / "labels"
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            lbl = lbl_dir / f"{img.stem}.txt"
            images.append((img, lbl))
    if images:
        return images
    # flat folder of images + labels
    for img in sorted(root.rglob("*")):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if "labels" in img.parts:
            continue
        lbl = img.with_suffix(".txt")
        alt = root / "labels" / f"{img.stem}.txt"
        images.append((img, lbl if lbl.exists() else alt))
    return images


def _remap_label(src: Path, dst: Path, id_map: dict[int, int]) -> Counter:
    counts: Counter = Counter()
    lines_out: list[str] = []
    if src.exists():
        for line in src.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            old = int(parts[0])
            if old not in id_map:
                continue
            new = id_map[old]
            parts[0] = str(new)
            lines_out.append(" ".join(parts))
            counts[TARGET_NAMES[new]] += 1
    dst.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")
    return counts


def _copy_replay(soccernet5: Path, dest_img: Path, dest_lbl: Path, per_class: int, rng: random.Random) -> Counter:
    wanted = {"goalpost": 4, "ball": 0, "goalkeeper": 2, "referee": 3, "player": 1}
    buckets: dict[str, list[Path]] = {k: [] for k in wanted}
    for lbl in (soccernet5 / "train" / "labels").glob("*.txt"):
        ids = {int(line.split()[0]) for line in lbl.read_text().splitlines() if line.strip()}
        for name, cid in wanted.items():
            if cid in ids:
                buckets[name].append(lbl)
    copied: Counter = Counter()
    used: set[str] = set()
    dest_img.mkdir(parents=True, exist_ok=True)
    dest_lbl.mkdir(parents=True, exist_ok=True)
    for name in ("goalpost", "ball", "goalkeeper", "referee"):
        pool = [p for p in buckets[name] if p.stem not in used]
        rng.shuffle(pool)
        take = pool[:per_class]
        for lbl in take:
            img = None
            for ext in (".jpg", ".jpeg", ".png"):
                cand = soccernet5 / "train" / "images" / f"{lbl.stem}{ext}"
                if cand.exists():
                    img = cand
                    break
            if img is None:
                continue
            shutil.copy2(img, dest_img / img.name)
            shutil.copy2(lbl, dest_lbl / lbl.name)
            used.add(lbl.stem)
            copied[name] += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Folder with images/labels or train/valid YOLO export")
    parser.add_argument("--out", default="training/datasets/finetune800")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replay", type=int, default=40, help="Replay images per rare class from soccernet5")
    parser.add_argument("--soccernet5", default="training/datasets/soccernet5")
    args = parser.parse_args()

    src = Path(args.src).expanduser().resolve()
    out = Path(args.out)
    if not src.exists():
        raise SystemExit(
            f"Dataset not found: {src}\n"
            "Put your 800-image YOLO/Roboflow export there, then rerun."
        )

    yaml_candidates = list(src.rglob("data.yaml"))[:5]
    names = _read_names(yaml_candidates[0]) if yaml_candidates else []
    if not names:
        names = TARGET_NAMES
        print("No data.yaml names found — assuming already 0=ball,1=player,2=goalkeeper,3=referee,4=goalpost")
    else:
        print("Source classes:", names)

    id_map: dict[int, int] = {}
    unknown = []
    for i, name in enumerate(names):
        canon = NAME_ALIASES.get(name.strip().lower())
        if canon is None:
            unknown.append(name)
            continue
        id_map[i] = TARGET_NAMES.index(canon)
    if unknown:
        print("Unmapped classes (dropped):", unknown)
    print("id_map", id_map)

    pairs = _iter_pairs(src)
    if not pairs:
        # maybe src itself is the images folder
        pairs = _iter_pairs(src.parent) if (src.parent / "labels").exists() else []
    if not pairs:
        raise SystemExit(f"No image/label pairs under {src}")

    groups: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for img, lbl in pairs:
        groups[_sequence_key(img.stem)].append((img, lbl))
    keys = sorted(groups)
    rng = random.Random(args.seed)
    rng.shuffle(keys)
    n_val = max(1, int(round(len(keys) * args.val_frac)))
    val_keys = set(keys[:n_val])
    print(f"Sequences: {len(keys)}  val sequences: {n_val}  images: {len(pairs)}")

    if out.exists():
        shutil.rmtree(out)
    counts = {"train": Counter(), "valid": Counter()}
    n_img = {"train": 0, "valid": 0}
    for key, items in groups.items():
        split = "valid" if key in val_keys else "train"
        img_dir = out / split / "images"
        lbl_dir = out / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img, lbl in items:
            shutil.copy2(img, img_dir / img.name)
            c = _remap_label(lbl, lbl_dir / f"{img.stem}.txt", id_map)
            counts[split].update(c)
            n_img[split] += 1

    replay_root = Path(args.soccernet5)
    if args.replay and (replay_root / "train" / "labels").exists():
        extra = _copy_replay(replay_root, out / "train" / "images", out / "train" / "labels", args.replay, rng)
        print("Replay from soccernet5 (anti-forgetting):", dict(extra))
        counts["train"].update(extra)

    yaml_path = out / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out.resolve()}",
                "train: train/images",
                "val: valid/images",
                "nc: 5",
                "names:",
                *[f"  {i}: {n}" for i, n in enumerate(TARGET_NAMES)],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {yaml_path}")
    for split in ("train", "valid"):
        print(f"{split}: {n_img[split]} images  boxes={dict(counts[split])}")
    print("Next: upload this folder + models/best.pt to Kaggle, then run training/finetune_best.py")


if __name__ == "__main__":
    main()
