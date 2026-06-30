# Makerweek-AGV2

A vision-guided navigation system for an Automated Guided Vehicle (AGV). A camera streams an overhead view of the playing field, the system detects ArUco markers and blue obstacle lines, builds a navigable grid, computes a path with A*, and drives the AGV along it.

## How It Works

1. **Camera feed** — reads an MJPEG stream from an overhead camera.
2. **Marker detection** — detects ArUco markers: 4 fixed corner markers define the field boundaries, plus one marker mounted on the AGV for position/heading tracking.
3. **Perspective correction** — warps the tilted camera view into a flat top-down image using the 4 corner markers.
4. **Obstacle detection** — detects blue lines on the field via HSV color masking.
5. **Grid generation** — divides the flattened field into a grid, marking cells as free or blocked.
6. **Pathfinding** — runs A* from the AGV's position to any free cell on the right edge, preferring routes with maximum clearance from obstacles.
7. **Steering** — computes the angle and distance to the next path waypoint and sends turn/drive commands to the AGV over HTTP.

## Project Structure

| File | Purpose |
|---|---|
| `main.py` | Main loop: ties everything together, runs every frame |
| `camera_feed.py` | Camera stream, ArUco detection, perspective warp, obstacle/grid detection |
| `path_algorithm.py` | Distance map, A* pathfinding, path drawing, steering target calculation |
| `drive.py` | AGV motor control via HTTP API (rotate, move, enable/disable) |

## Requirements

```bash
pip install opencv-python opencv-contrib-python numpy requests
```

`opencv-contrib-python` is required for the `cv2.aruco` module.

## Configuration

Update these constants before running:

**`camera_feed.py`**
- `stream_url` — MJPEG stream URL of the overhead camera
- `ID_TOP_LEFT`, `ID_TOP_RIGHT`, `ID_BOTTOM_LEFT`, `ID_BOTTOM_RIGHT` — ArUco IDs of the 4 corner markers
- `LOWER_BLUE` / `UPPER_BLUE` — HSV range for obstacle line detection (tune for your lighting)
- `WARP_WIDTH` / `WARP_HEIGHT` — output resolution of the flattened field view

**`main.py`**
- `AGV_MARKER_ID` — ArUco ID mounted on the AGV
- `COLS` / `ROWS` — grid resolution (higher = more precise, slower to compute)

**`drive.py`**
- `AGV_BASE_URL` — HTTP API base URL for the AGV
- `STEPS_PER_DEGREE`, `STEPS_PER_MM`, `PIXEL_PER_MM` — calibration values for your specific AGV and camera setup

## Running

```bash
python3 main.py
```

A window will open showing the flattened field with the grid, detected obstacles, and computed path overlaid.

## Notes

- ArUco dictionary is `DICT_4X4_50` — make sure your printed markers use this dictionary.
- Calibration values (`STEPS_PER_MM`, `PIXEL_PER_MM`, etc.) are specific to the physical hardware and field setup and will need to be re-measured if either changes.