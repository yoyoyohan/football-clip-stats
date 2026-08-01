# Calibration Guide — Follow This Pipeline

Do this **once per video** (or once per camera angle).  
Calibration maps the TV image → a real pitch (105×68 meters) so passes, shots, and goals use real distances.

---

## Quick path (most users)

```bash
# 1. Put your clip here
#    input_videos/myclip.mp4

# 2. Calibrate (opens video — scrub, then click)
python calibrate_pitch.py --source input_videos/myclip.mp4

# 3. Run stats
python run_clip.py --source input_videos/myclip.mp4 --no-cache
```

Done. Calibration is saved as `calibration/myclip.json`.

---

## Step-by-step

### Step 1 — Open the calibrator

```bash
cd /path/to/statsapp
python calibrate_pitch.py --source input_videos/YOURCLIP.mp4
```

A window opens with your video and a **frame** slider.

### Step 2 — Pick a good frame

| Control | What it does |
|---------|----------------|
| Drag **frame** slider | Scrub the whole clip |
| `a` / `d` | Previous / next frame |
| `[` / `]` | Jump ±10 frames |
| `{` / `}` | Jump ±30 frames |
| **SPACE** or **ENTER** | Lock this frame and continue |
| `x` | Quit |

**What to look for** (best → still OK):

1. **Best:** A goal clearly visible (both posts)
2. **Good:** Penalty box, center circle, or clear pitch lines
3. **OK:** Any wide-ish shot of the field  
4. **Worst:** Extreme close-up of one player only → skip calibration (see Step 5)

### Step 3 — Choose how to click (auto-picked for you)

Default mode is **`--mode goal`** (one goal visible).  
If that doesn’t fit what you see, quit (`x`) and restart with the right mode:

| What you see on screen | Command |
|------------------------|---------|
| **One goal** (2 posts) — tight TV shot | `python calibrate_pitch.py --source input_videos/YOURCLIP.mp4` |
| **No goal**, but center / box / corners | `python calibrate_pitch.py --source input_videos/YOURCLIP.mp4 --mode landmarks` |
| **Full pitch** (all 4 corners) | `python calibrate_pitch.py --source input_videos/YOURCLIP.mp4 --mode corners` |
| **Nothing useful** (no goal, no lines) | Skip calibration → go to Step 5 |

---

### Step 4A — One-goal mode (default)

After you lock a frame:

1. Press **`l`** if the goal is on the **LEFT** of the TV, or **`r`** if on the **RIGHT**  
   *(Type in the terminal if keys don’t work in the video window — common on Mac.)*
2. Click the ground at **post 1**, then **post 2**
3. Optional: click the **penalty spot** or **18-yard line** for better accuracy
4. Press **`c`** in the terminal to **save**

Still on the wrong frame? Use the **frame** slider or `a`/`d` — clicks reset when you move.

**You only need 2 clicks** (both posts). A 3rd click is optional.

---

### Step 4B — Landmarks mode (no goal in frame)

```bash
python calibrate_pitch.py --source input_videos/YOURCLIP.mp4 --mode landmarks
```

1. Scrub → **SPACE** to lock a frame with pitch marks
2. Press a **number key**, then **click** that mark on the video
3. Repeat until you have **at least 4** points
4. Press **`c`** to save

| Key | What to click |
|-----|----------------|
| `1` `2` | Left goal posts (if visible) |
| `3` `4` | Right goal posts (if visible) |
| `5` | Center spot (kickoff dot) |
| `6` `7` | Penalty spots |
| `8` `9` | Bottom corner flags (near camera) |
| `0` `-` | Top corner flags (far from camera) |
| `q` `w` `e` `r` | Penalty-box front corners (18-yard line) |

**LEFT / RIGHT** = left / right side of **your TV**.  
**Bottom** = near camera · **Top** = far from camera.

Tip: for a midfield-only shot, try `5` (center) + whatever box/line corners you can see.

---

### Step 4C — Full-pitch corners mode

```bash
python calibrate_pitch.py --source input_videos/YOURCLIP.mp4 --mode corners
```

Click in order: bottom-left → bottom-right → top-right → top-left → press **`c`**.

---

### Step 5 — No usable frame? Skip calibration

You can still get **passes** and **possession** without calibration:

```bash
python run_clip.py --source input_videos/YOURCLIP.mp4 --no-cache
```

Pitch meters / shots / goals will be less accurate. That’s OK for tight midfield clips (e.g. some Morocco frames).

---

### Step 6 — Run the stats pipeline

```bash
python run_clip.py --source input_videos/YOURCLIP.mp4 --no-cache --team0-name TeamA --team1-name TeamB
```

Examples:

```bash
# El Clásico
python run_clip.py --source input_videos/elclasico.mp4 --no-cache

# Morocco vs France
python run_clip.py --source input_videos/moroccomatch.mp4 --no-cache --team0-name Morocco --team1-name France
```

### Step 7 — Check outputs

| File | What it is |
|------|------------|
| `output_videos/YOURCLIP_stats.json` | Passes, shots, SOT, goals, possession |
| `output_videos/YOURCLIP_stats_per_frame.csv` | Possession + ball position per frame |
| `calibration/YOURCLIP.json` | Your pitch calibration (reuse next run) |

---

## Decision cheat sheet

```
Open calibrator
      │
      ▼
Scrub video ──► see a GOAL? ──yes──► default mode (2 posts) ──► save ──► run_clip
      │
      no
      ▼
See pitch LINES / center / box? ──yes──► --mode landmarks (4+ clicks) ──► save ──► run_clip
      │
      no
      ▼
Skip calibration ──► run_clip anyway (passes/possession still work)
```

---

## Common problems

| Problem | Fix |
|---------|-----|
| Keys do nothing | Click the **terminal**, type `l` / `r` / `c` there (Mac OpenCV quirk) |
| Only 2 marks visible | One-goal mode is fine — press `c` after 2 posts |
| No goal in whole clip | Use `--mode landmarks`, or skip calibration |
| Stats look wrong after new calibration | Add `--no-cache` when running `run_clip.py` |
| Want to redo calibration | Run `calibrate_pitch.py` again (overwrites `calibration/YOURCLIP.json`) |
