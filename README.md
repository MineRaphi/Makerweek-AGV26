# Makerweek-AGV26

A vision-guided navigation system for an Automated Guided Vehicle (AGV). A camera streams an overhead view of the playing field, the system detects ArUco markers and blue obstacle lines, builds a navigable grid, computes a path with A*, and drives the AGV along it.

> **Competition result:** We were eliminated in the first round. The loss came down to a single edge case that only appeared once during testing so we focused on different things. When the AGV rotated in place, its ArUco marker shifted slightly from the true rotation center, causing the pathfinder to sometimes see a new position and recompute a different path. The AGV then tried to correct for the new path, which moved the marker again, triggering the same problem again, locking it into an oscillation loop between two paths and never actually driving forward.

---

## How It Works

1. **Camera feed** — reads an MJPEG stream from an overhead camera.
2. **Marker detection** — detects ArUco markers: 4 fixed corner markers define the field boundaries, plus one marker mounted on the AGV for position/heading tracking.
3. **Perspective correction** — warps the tilted camera view into a flat top-down image using the 4 corner markers.
4. **Obstacle detection** — detects blue lines on the field via HSV color masking.
5. **Grid generation** — divides the flattened field into a grid, marking cells as free or blocked. Obstacles are inflated by a configurable radius so the AGV keeps a safe distance from lines.
6. **Pathfinding** — runs A* from the AGV's position to any free cell on the right edge, preferring routes with maximum clearance from obstacles. If the start cell is blocked, the algorithm snaps to the nearest free cell automatically.
7. **Steering** — computes the angle and distance to the next path waypoint and sends turn/drive commands to the AGV over HTTP.

---

## Project Structure

| File | Purpose |
|---|---|
| `main.py` | Main loop: ties everything together, runs every frame |
| `record.py` | Capture-only loop: records a replay without controlling the AGV; computes paths for every non-corner marker visible |
| `camera_feed.py` | Camera stream, ArUco detection, perspective warp, obstacle/grid detection |
| `path_algorithm.py` | Distance map, A* pathfinding, path drawing, steering target calculation |
| `drive.py` | AGV motor control via HTTP API (rotate, move, enable/disable) |
| `logger.py` | Writes a per-frame CSV log of all processed data (has been replaced by `replay.py`) |
| `replay.py` | Records and replays sessions: saves multiple video streams + CSV data, interactive player with TAB stream switching |
| `config.py` | All constants and configuration in one place |

---

## Requirements

```bash
pip install opencv-python opencv-contrib-python numpy requests
```

`opencv-contrib-python` is required for the `cv2.aruco` module.

---

## Configuration

All constants live in `config.py` — edit that file before running rather than changing individual source files.

Key settings:

| Constant | Description |
|---|---|
| `STREAM_URL` | MJPEG stream URL of the overhead camera |
| `AGV_BASE_URL` | HTTP API base URL for the AGV motor controller |
| `AGV_MARKER_ID` | ArUco ID of the marker mounted on the AGV |
| `ID_TOP_LEFT` / `ID_TOP_RIGHT` / `ID_BOTTOM_LEFT` / `ID_BOTTOM_RIGHT` | ArUco IDs of the 4 field corner markers |
| `COLS` / `ROWS` | Grid resolution — higher is more precise but slower to compute |
| `WARP_WIDTH` / `WARP_HEIGHT` | Output resolution of the flattened top-down view |
| `LOWER_BLUE` / `UPPER_BLUE` | HSV range for blue obstacle line detection — tune for your lighting |
| `INFLATION_RADIUS` | How many grid cells to expand obstacles by, to keep the AGV away from lines |
| `STEPS_PER_DEGREE` / `STEPS_PER_MM` / `PIXEL_PER_MM` | Motor and field calibration values |
| `TURN_WAIT` | Number of frames to wait between steering corrections |
| `LOGGING_ENABLED` / `LOG_DIR` | Toggle CSV logging and set the output folder |
| `REPLAY_ENABLED` | Toggle session recording |

---

## Running

**Normal run (vision + pathfinding + AGV control):**
```bash
python3 main.py
```

**Record only (no AGV control, paths drawn for all visible markers):**
```bash
python3 record.py
```

**Replay a recorded session:**
```bash
python3 replay.py
# or pass the session folder directly:
python3 replay.py sessions/session_20260701_142305/
```

Press `Q` to quit any window.

---

## Replay Player Controls

| Key | Action |
|---|---|
| `SPACE` | Pause / resume |
| `A` / `D` | Step back / forward one frame |
| `TAB` | Cycle between recorded video streams (raw, warped, final, mask) |
| `H` | Hide / show the data overlay panel |
| `+` / `-` | Speed up / slow down playback |
| `Q` / `ESC` | Quit |

---

## Notes

- ArUco dictionary is `DICT_4X4_50` — make sure your printed markers use this dictionary, or update `aruco_dict` in `camera_feed.py`.
- Calibration values (`STEPS_PER_MM`, `PIXEL_PER_MM`, etc.) are specific to the physical hardware and field setup and will need to be re-measured if either changes.
- The HSV blue detection range (`LOWER_BLUE` / `UPPER_BLUE`) is sensitive to lighting — use the tuner helper in `camera_feed.py` to find the right values for your environment.
- All 4 corner markers must be visible before the perspective warp and pathfinding can run. The system waits and prints a message until they appear.