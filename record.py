"""
record_only.py — Capture + Pathfinding-only script

Reads the camera stream, runs marker detection, perspective correction,
grid creation, and computes a path for every non-corner marker detected.
No drive control.

Run with:
    python record_only.py
"""

import camera_feed as cf
import path_algorithm as pa
import copy
from replay import Recorder
from config import *

# IDs that are field boundary markers — excluded from pathfinding
CORNER_IDS = {ID_TOP_LEFT, ID_TOP_RIGHT, ID_BOTTOM_LEFT, ID_BOTTOM_RIGHT}

# --- Start recorder ---
recorder = Recorder(fps=12)
recorder.start()

print("Recording... press Q to stop.")

while True:
    ret, frame = cf.cap.read()
    if not ret:
        break

    gray = cf.cv2.cvtColor(frame, cf.cv2.COLOR_BGR2GRAY)

    cf.detect_markers(frame, gray)
    cf.draw_marker_direction(AGV_MARKER_ID, frame)

    # Only proceed if all 4 corner markers are visible
    all_corners_visible = all(
        mid in cf.marker_centers
        for mid in [ID_TOP_LEFT, ID_TOP_RIGHT, ID_BOTTOM_LEFT, ID_BOTTOM_RIGHT]
    )

    if not all_corners_visible:
        print("Waiting for all 4 corner markers...")
        cf.cv2.imshow("Recording - Raw", frame)
        if cf.cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # --- Vision pipeline ---
    warped, M        = cf.flatten_image(frame)
    original_warped  = copy.deepcopy(warped)
    warped, mask, lines = cf.detect_blue_lines(warped)
    grid             = cf.create_grid(warped, COLS, ROWS)

    # Inflate obstacles
    kernel = cf.cv2.getStructuringElement(
        cf.cv2.MORPH_RECT,
        (2 * INFLATION_RADIUS + 1, 2 * INFLATION_RADIUS + 1)
    )
    inflated_grid = cf.cv2.dilate(grid.astype('uint8'), kernel, iterations=1)
    dist_map      = pa.compute_distance_map(inflated_grid)

    # Draw the base grid once
    display = cf.draw_grid(warped.copy(), grid, COLS, ROWS)

    # --- Compute a path for every non-corner marker currently visible ---
    non_corner_ids = [
        mid for mid in cf.marker_centers
        if mid not in CORNER_IDS
    ]

    all_paths = {}  # {marker_id: path or None}

    # Cycle through colors so each marker's path is visually distinct
    path_colors = [
        (0, 255, 0),    # green
        (0, 165, 255),  # orange
        (255, 0, 255),  # magenta
        (0, 255, 255),  # yellow
        (255, 255, 0),  # cyan
    ]

    for i, marker_id in enumerate(non_corner_ids):
        pos = cf.marker_to_grid(marker_id, M, warped.shape, COLS, ROWS)
        if pos is None:
            all_paths[marker_id] = None
            continue

        path = pa.astar(inflated_grid, pos, dist_map, clearance_weight=6.0)
        all_paths[marker_id] = path

        # Draw this marker's path in its own color
        color = path_colors[i % len(path_colors)]
        display = pa.draw_path(cf.cv2, display, path, inflated_grid, COLS, ROWS,
                               color=color)

        # Label the start of each path with the marker ID
        if pos is not None:
            h, w = display.shape[:2]
            row_edges = cf.np.linspace(0, h, ROWS + 1, dtype=int)
            col_edges = cf.np.linspace(0, w, COLS + 1, dtype=int)
            cx = int((col_edges[pos[1]] + col_edges[pos[1]+1]) / 2)
            cy = int((row_edges[pos[0]] + row_edges[pos[0]+1]) / 2)
            cf.cv2.putText(display, f"ID{marker_id}",
                (cx + 5, cy - 5),
                cf.cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    print(f"Computed paths for markers: {non_corner_ids}")

    recorder.record_frame(
        frames={
            "raw":    frame,
            "warped": original_warped,
            "final":  display,
            "mask":   mask,
        },
        lines          = lines,
        marker_centers = cf.marker_centers,
    )

    cf.cv2.imshow("Recording - Paths", display)

    if cf.cv2.waitKey(1) & 0xFF == ord('q'):
        break

recorder.stop()
cf.cap.release()
cf.cv2.destroyAllWindows()