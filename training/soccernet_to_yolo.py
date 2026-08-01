"""
Convert SoccerNet-v3 JSON annotations to Ultralytics YOLO format.

SoccerNet class reference:
https://github.com/SoccerNet/SoccerNet/blob/master/SoccerNet/Evaluation/utils.py
"""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path

from tqdm import tqdm

YOLO_NAMES = ["ball", "player", "goalkeeper", "referee", "goalpost"]

# SoccerNet-v3 bbox class string -> YOLO class index
SN_BBOX_CLASS_TO_YOLO: dict[str, int] = {
    "Ball": 0,
    "Player team left": 1,
    "Player team right": 1,
    "Player team unknown 1": 1,
    "Player team unknown 2": 1,
    "Goalkeeper team left": 2,
    "Goalkeeper team right": 2,
    "Goalkeeper team unknown": 2,
    "Main referee": 3,
    "Side referee": 3,
}

# Goal parts are line annotations in SoccerNet-v3, not bboxes.
SN_GOAL_LINE_CLASS_TO_YOLO: dict[str, int] = {
    "Goal left post left ": 4,
    "Goal left post right": 4,
    "Goal left crossbar": 4,
    "Goal right post left": 4,
    "Goal right post right": 4,
    "Goal right crossbar": 4,
}


def _xyxy_to_yolo_line(
    cls_id: int, x1: float, y1: float, x2: float, y2: float, image_meta: dict
) -> str | None:
    w_img = float(image_meta["width"])
    h_img = float(image_meta["height"])
    x_c = ((x1 + x2) / 2.0) / w_img
    y_c = ((y1 + y2) / 2.0) / h_img
    bw = abs(x2 - x1) / w_img
    bh = abs(y2 - y1) / h_img
    x_c = min(1.0, max(0.0, x_c))
    y_c = min(1.0, max(0.0, y_c))
    bw = min(1.0, max(0.0, bw))
    bh = min(1.0, max(0.0, bh))
    if bw <= 0 or bh <= 0:
        return None
    return f"{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}"


def bbox_to_yolo_line(bbox: dict, image_meta: dict) -> str | None:
    sn_class = bbox.get("class")
    if sn_class not in SN_BBOX_CLASS_TO_YOLO:
        return None
    cls_id = SN_BBOX_CLASS_TO_YOLO[sn_class]
    x1 = float(bbox["points"]["x1"])
    y1 = float(bbox["points"]["y1"])
    x2 = float(bbox["points"]["x2"])
    y2 = float(bbox["points"]["y2"])
    return _xyxy_to_yolo_line(cls_id, x1, y1, x2, y2, image_meta)


def line_to_yolo_line(line: dict, image_meta: dict, padding: float = 12.0) -> str | None:
    sn_class = line.get("class")
    if sn_class not in SN_GOAL_LINE_CLASS_TO_YOLO:
        return None
    points = line.get("points") or []
    if len(points) < 4:
        return None
    xs = [float(points[i]) for i in range(0, len(points), 2)]
    ys = [float(points[i]) for i in range(1, len(points), 2)]
    if not xs or not ys:
        return None
    cls_id = SN_GOAL_LINE_CLASS_TO_YOLO[sn_class]
    x1 = min(xs) - padding
    y1 = min(ys) - padding
    x2 = max(xs) + padding
    y2 = max(ys) + padding
    return _xyxy_to_yolo_line(cls_id, x1, y1, x2, y2, image_meta)


def _extract_image(zip_path: Path, image_name: str, dest_path: Path) -> bool:
    if dest_path.exists():
        return True
    if not zip_path.exists():
        return False
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        if image_name not in zf.namelist():
            return False
        with zf.open(image_name) as src, open(dest_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return True


def convert_game(
    soccernet_root: Path,
    game_rel_path: str,
    images_out: Path,
    labels_out: Path,
    stem_prefix: str,
) -> int:
    game_dir = soccernet_root / game_rel_path
    labels_path = game_dir / "Labels-v3.json"
    if not labels_path.exists():
        return 0

    metadata = json.loads(labels_path.read_text(encoding="utf-8"))
    url_local = metadata["GameMetadata"]["UrlLocal"]
    zip_path = soccernet_root / url_local / "Frames-v3.zip"
    count = 0

    for action_name in metadata["GameMetadata"]["list_actions"]:
        img_names = [action_name] + metadata["actions"][action_name]["linked_replays"]
        for i, img_name in enumerate(img_names):
            img_type = "actions" if i == 0 else "replays"
            ann = metadata[img_type][img_name]
            yolo_lines = []
            for bbox in ann.get("bboxes", []):
                line = bbox_to_yolo_line(bbox, ann["imageMetadata"])
                if line:
                    yolo_lines.append(line)
            for goal_line in ann.get("lines", []):
                line = line_to_yolo_line(goal_line, ann["imageMetadata"])
                if line:
                    yolo_lines.append(line)
            lines = yolo_lines
            if not lines:
                continue

            safe_stem = f"{stem_prefix}_{img_name.replace('/', '_').replace('.png', '')}"
            image_out = images_out / f"{safe_stem}.png"
            label_out = labels_out / f"{safe_stem}.txt"

            if not _extract_image(zip_path, img_name, image_out):
                continue

            label_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
            count += 1
    return count


def write_data_yaml(output_dir: Path) -> Path:
    yaml_path = output_dir / "data.yaml"
    train = output_dir / "images" / "train"
    val = output_dir / "images" / "val"
    test = output_dir / "images" / "test"

    # If only train was converted (common on Kaggle), reuse train for val.
    val_path = "images/val" if val.exists() and any(val.glob("*")) else "images/train"
    test_path = "images/test" if test.exists() and any(test.glob("*")) else val_path

    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve()}",
                "train: images/train",
                f"val: {val_path}",
                f"test: {test_path}",
                f"nc: {len(YOLO_NAMES)}",
                f"names: {YOLO_NAMES}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def convert_soccernet_v3(
    soccernet_root: str | Path,
    output_dir: str | Path,
    splits: list[str] | None = None,
    max_games_per_split: int | None = None,
) -> Path:
    from SoccerNet.utils import getListGames

    soccernet_root = Path(soccernet_root)
    output_dir = Path(output_dir)
    splits = splits or ["train", "valid", "test"]

    split_map = {"train": "train", "valid": "val", "test": "test"}
    total = 0

    for split in splits:
        yolo_split = split_map.get(split, split)
        images_out = output_dir / "images" / yolo_split
        labels_out = output_dir / "labels" / yolo_split
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        games = getListGames(split, task="frames")
        if max_games_per_split is not None:
            games = games[:max_games_per_split]

        for game in tqdm(games, desc=f"Converting {split}"):
            prefix = game.replace("/", "_").replace(" ", "_")
            total += convert_game(
                soccernet_root, game, images_out, labels_out, prefix
            )

    yaml_path = write_data_yaml(output_dir)
    print(f"Converted {total} images -> {output_dir}")
    return yaml_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SoccerNet-v3 to YOLO converter")
    parser.add_argument("--soccernet-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "valid"])
    parser.add_argument("--max-games-per-split", type=int, default=None)
    args = parser.parse_args()

    convert_soccernet_v3(
        args.soccernet_root,
        args.output_dir,
        splits=args.splits,
        max_games_per_split=args.max_games_per_split,
    )
