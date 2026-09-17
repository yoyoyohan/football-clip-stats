# Calibration Guide — Follow This Pipeline

Calibration maps the TV image → a real pitch (105×68 meters) so passes, shots, and goals use real distances.

**Automatic calibration is the default.** Interactive clicking is the fallback when auto-cal cannot find stable goalposts.

---

## Quick path (most users)

```bash
# 1. Put your clip here
#    input_videos/myclip.mp4

# 2. Run stats — pitch auto-calibrates from YOLO goalpost detections
python run_clip.py --source input_videos/myclip.mp4 --no-cache
```

Done. If goalposts were visible often enough, calibration is saved as `calibration/myclip.json` with `"method": "goalpost_auto_multiframe"`.

Optional: force a fresh auto-cal even if a manual JSON already exists:

```bash
python run_clip.py --source input_videos/myclip.mp4 --force-auto-cal
```

---

## How automatic calibration works

`run_clip.py` calls `robust_auto_calibration_from_frames` when:

| Situation | Behavior |
|-----------|----------|
| No `calibration/<clip>.json` | Auto-calibrate (default) |
| Existing JSON method is `goalpost_auto*` **and** `--no-cache` | Refresh auto-cal |
| Existing JSON method is manual / landmarks / corners / single_goal | **Keep** it (unless `--force-auto-cal`) |
| `--force-auto-cal` | Always recompute auto-cal |

The auto-calibrator:

1. Collects goalpost detections across frames (`overlay.goalposts` or `detections`)
2. Aggregates stable left/right post **feet** (bbox bottom-center) via median / clustering
3. Infers goal side (left / right / both)
4. Builds a homography (`build_calibration_from_one_goal` or dual-goal landmarks)
5. Quality-gates: enough posts, plausible separation, pitch sanity — returns `None` if bad

If auto-cal fails, `run_clip` falls back to the interactive tools below (or a default homography).

---

## Interactive fallback

Use this when auto-cal fails, or when you want higher accuracy on a tricky camera angle.

```bash
# Opens video — scrub, then click
python calibrate_pitch.py --source input_videos/myclip.mp4
```

Calibration is saved as `calibration/myclip.json` (manual methods are preserved by `run_clip` unless you pass `--force-auto-cal`).

---

## Step-by-step (interactive)

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
| `calibration/YOURCLIP.json` | Pitch calibration (auto or manual; reuse next run) |

---

## Decision cheat sheet

```
run_clip (default)
      │
      ▼
goalposts across frames? ──yes──► auto multi-frame cal ──► save JSON ──► stats
      │
      no / failed
      ▼
Interactive calibrate_pitch? ──yes──► click posts/landmarks ──► save ──► run_clip
      │
      no
      ▼
Skip calibration ──► run_clip anyway (passes/possession still work)
```

---

## Common problems

| Problem | Fix |
|---------|-----|
| Auto-cal failed / quality low | Run interactive `calibrate_pitch.py`, or ensure goalposts are detected (`--no-cache`) |
| Want to redo auto-cal | `python run_clip.py --source ... --force-auto-cal` (or `--no-cache` if prior method was auto) |
| Keys do nothing (interactive) | Click the **terminal**, type `l` / `r` / `c` there (Mac OpenCV quirk) |
| Only 2 marks visible | One-goal interactive mode is fine — press `c` after 2 posts |
| No goal in whole clip | Use `--mode landmarks`, or skip calibration |
| Stats look wrong after new calibration | Add `--no-cache` when running `run_clip.py` |
| Want to keep manual cal | Don’t pass `--force-auto-cal` — manual JSON is preserved |
