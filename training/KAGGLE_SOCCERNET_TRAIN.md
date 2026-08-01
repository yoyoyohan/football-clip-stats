# Kaggle: Train YOLO on SoccerNet-v3

Copy each section into a **separate cell** in a Kaggle notebook with **GPU** enabled.

Register at [soccer-net.org](https://www.soccer-net.org) first — you need a download password.

---

## Cell 1 — Install dependencies

```python
!pip install -q ultralytics SoccerNet tqdm
```

---

## Cell 2 — SoccerNet password (Kaggle Secrets)

1. Kaggle → **Add-ons** → **Secrets**
2. Create secret: `SOCCERNET_PASSWORD` = your SoccerNet password

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["SOCCERNET_PASSWORD"] = UserSecretsClient().get_secret("SOCCERNET_PASSWORD")
print("Password loaded")
```

Or paste directly (not recommended):

```python
import os
os.environ["SOCCERNET_PASSWORD"] = "YOUR_PASSWORD_HERE"
```

---

## Cell 3 — Download SoccerNet-v3 (use small subset on Kaggle)

Full dataset ≈ 60GB. Start with **train only + 30 games** (~few GB).

```python
from SoccerNet.Downloader import SoccerNetDownloader

SOCCERNET_DIR = "/kaggle/working/SoccerNet"
os.makedirs(SOCCERNET_DIR, exist_ok=True)

downloader = SoccerNetDownloader(LocalDirectory=SOCCERNET_DIR)
downloader.password = os.environ["SOCCERNET_PASSWORD"]

# Train split only — add "valid" when you have more disk space
downloader.downloadGames(
    files=["Labels-v3.json", "Frames-v3.zip"],
    split=["train"],
    task="frames",
)
print("Download complete")
```

---

## Cell 4 — Converter (SoccerNet JSON → YOLO)

Upload `training/soccernet_to_yolo.py` from this repo as a Kaggle dataset, **or** paste the file contents here.

```python
# If you uploaded the script as a Kaggle dataset:
# !cp /kaggle/input/statsapp-scripts/soccernet_to_yolo.py /kaggle/working/

from pathlib import Path
import sys
sys.path.append("/kaggle/working")

from soccernet_to_yolo import convert_soccernet_v3

YOLO_DIR = "/kaggle/working/soccernet_yolo"
yaml_path = convert_soccernet_v3(
    soccernet_root=SOCCERNET_DIR,
    output_dir=YOLO_DIR,
    splits=["train", "valid"],      # use ["train"] only if disk is tight
    max_games_per_split=30,         # increase to 100+ for better model; None = all
)
print("data.yaml:", yaml_path)
```

Classes exported: `ball`, `player`, `goalkeeper`, `referee`, `goalpost`

---

## Cell 5 — (Optional) Upload your existing weights

Add `best.pt` from this project as a Kaggle dataset input.

```python
import shutil
from pathlib import Path

WEIGHTS_IN = Path("/kaggle/input/statsapp-weights/best.pt")  # change path
WEIGHTS_OUT = Path("/kaggle/working/best.pt")

if WEIGHTS_IN.exists():
    shutil.copy(WEIGHTS_IN, WEIGHTS_OUT)
    MODEL = str(WEIGHTS_OUT)
    print("Fine-tuning from", MODEL)
else:
    MODEL = "yolov8m.pt"
    print("Training from", MODEL)
```

---

## Cell 6 — Train YOLO

```python
from ultralytics import YOLO

model = YOLO(MODEL)

results = model.train(
    data=str(yaml_path),
    epochs=50,
    imgsz=640,
    batch=16,
    patience=10,
    project="/kaggle/working/runs",
    name="soccernet_v3",
    device=0,
    # Fine-tune settings when starting from best.pt:
    lr0=1e-4,
    lrf=0.01,
    mosaic=1.0,
    mixup=0.1,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
)

print("Best weights:", results.save_dir / "weights" / "best.pt")
```

---

## Cell 7 — Validate + download

```python
from IPython.display import FileLink

best = Path("/kaggle/working/runs/soccernet_v3/weights/best.pt")
metrics = YOLO(str(best)).val(data=str(yaml_path))
print(metrics)

# Kaggle notebook output tab → download best.pt
# Replace models/best.pt locally, then:
#   python yolo_inference.py
#   python main.py
FileLink(str(best))
```

---

## Tips

| Setting | Kaggle trial | Serious training |
|---|---|---|
| `max_games_per_split` | 20–30 | `None` (all games) |
| `splits` | `["train"]` | `["train", "valid"]` |
| `epochs` | 30–50 | 80–100 |
| `MODEL` | `yolov8m.pt` | your `best.pt` |
| Disk | ~20GB used | need ~60GB+ |

## After download

```bash
cp best.pt ~/Desktop/statsapp/models/best.pt
cd ~/Desktop/statsapp
python yolo_inference.py
python main.py
```
