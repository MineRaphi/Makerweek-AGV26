import camera_feed as cf
import path_algorithm as pa
import drive as d
import csv
import os
from datetime import datetime

# --- Configuration ---
AGV_MARKER_ID = 21   # ArUco ID printed on the AGV itself (used to track its position/heading)
COLS = 80            # number of grid columns to divide the playing field into
ROWS = 60            # number of grid rows to divide the playing field into

turn_counter = 0

# --- CSV log file setup ---
# Creates a new log file each run, named with the current timestamp
log_filename = f"logs/log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
log_file = open(log_filename, "w", newline="")
log_writer = csv.writer(log_file)

# Write the header row
log_writer.writerow([
    "timestamp",
    "agv_grid_row", "agv_grid_col",
    "agv_angle",
    "target_angle", "target_distance",
    "target_cell_row", "target_cell_col",
    "turn_amount",
    "action",            # "rotate" or "move" or "none"
    "action_value",      # the actual degrees turned or mm driven
    "path_length",       # number of cells in the computed path
    "blue_lines_found",  # how many blue line contours were detected
    "marker_ids_visible" # comma-separated list of detected marker IDs
])


# --- Connect to the AGV's motor controller ---
try:
    d.enable_wheels()
except:
    # If the AGV isn't connected (e.g. testing the vision pipeline only),
    # don't crash — just continue without drive control
    print("No AGV connection")

# --- Main loop: runs once per camera frame ---
while True:
    ret, frame = cf.cap.read()
    if not ret:
        break  # stream ended or dropped, exit the loop

    # Convert to grayscale — required for ArUco marker detection
    gray = cf.cv2.cvtColor(frame, cf.cv2.COLOR_BGR2GRAY)

    # 1. Detect all ArUco markers on the raw (un-warped) frame
    #    This also populates marker_centers / marker_corners in camera_feed.py
    cf.detect_markers(frame, gray)

    # 2. Draw a direction arrow on the raw frame showing which way the AGV is facing
    cf.draw_marker_direction(AGV_MARKER_ID, frame)

    # 3. Apply perspective correction so the field appears as a flat top-down view
    #    M is the transform matrix, needed later to convert marker positions into the warped space
    warped, M = cf.flatten_image(frame)

    # 4. Detect the blue obstacle lines on the warped (flat) image
    warped, mask, lines = cf.detect_blue_lines(warped)

    # 5. Convert the warped image into a grid: 0 = free cell, 1 = blocked (blue line)
    grid = cf.create_grid(warped, COLS, ROWS)

    # 6. Compute how far every free cell is from the nearest obstacle/edge
    #    Used by A* to prefer paths that stay away from walls
    dist_map = pa.compute_distance_map(grid)

    # 7. Find the AGV's current position on the grid (using its marker, mapped through M)
    agv_pos = cf.marker_to_grid(AGV_MARKER_ID, M, warped.shape, COLS, ROWS)

    if agv_pos is not None:
        start = agv_pos
    else:
        # Fallback starting position if the AGV marker isn't visible this frame
        start = (ROWS // 2, 0)

    # 8. Run A* pathfinding from the AGV's position to any cell on the right edge,
    #    preferring routes with more clearance from obstacles (clearance_weight)
    path = pa.astar(grid, start, dist_map, clearance_weight=6.0)

    # 9. Draw the grid and the computed path onto the warped image (for visualization/debugging)
    warped = cf.draw_grid(warped, grid, COLS, ROWS)
    warped = pa.draw_path(cf.cv2, warped, path, grid, COLS, ROWS)

    # --- Collect data for this frame ---
    timestamp       = datetime.now().isoformat()
    agv_angle       = None
    target_angle    = None
    target_distance = None
    target_cell     = None
    turn_amount     = None
    action          = "none"
    action_value    = None
    segment         = None

    # 10. Compute the next steering target along the path (angle + distance to aim for)
    if agv_pos is not None:
        segment = pa.get_next_segment(path, agv_pos, grid.shape, warped.shape, COLS, ROWS, lookahead=6)
        if segment is not None:
            target_angle, distance, target_cell = segment
    else:
        segment = None
        print("AGV marker not detected, skipping steering this frame")

    # 11. If we have a valid steering target, decide whether to turn or drive forward
    if segment is not None:
        target_angle, distance, target_cell = segment
        print(f"Steer to angle: {target_angle:.1f}°  distance: {distance:.1f}px")

        # Get the AGV's current heading from its marker orientation
        agv_dir = cf.get_marker_direction(AGV_MARKER_ID)

        if agv_dir is not None:
            turn_counter += 1

            current_angle = agv_dir[0]

            # How far off the AGV's heading is from where it needs to point
            turn_amount = target_angle - current_angle
            # Normalize to the range -180..180 so it always turns the shorter way
            turn_amount = (turn_amount + 180) % 360 - 180
            print(f"Direction off by: {turn_amount:.1f}°")

            if turn_counter == 3:
                turn_counter = 0
                # If heading is off by more than 5°, rotate to correct it first
                if turn_amount > 5 or turn_amount < -5:
                    d.rotate(turn_amount * 0.9)  # 0.9 = slight damping to avoid overshooting
                else:
                    # Heading is close enough — drive forward towards the target
                    d.move_mm(distance * d.PIXEL_PER_MM)

    # --- Write one row to the log ---
    log_writer.writerow([
        timestamp,
        agv_pos[0] if agv_pos else "",   # agv_grid_row
        agv_pos[1] if agv_pos else "",   # agv_grid_col
        round(agv_angle, 2) if agv_angle is not None else "",
        round(target_angle, 2) if target_angle is not None else "",
        round(target_distance, 2) if target_distance is not None else "",
        target_cell[0] if target_cell else "",  # target_cell_row
        target_cell[1] if target_cell else "",  # target_cell_col
        round(turn_amount, 2) if turn_amount is not None else "",
        action,
        action_value if action_value is not None else "",
        len(path) if path is not None else 0,
        len(lines),
        # Marker IDs currently visible, as a space-separated string
        " ".join(str(k) for k in cf.marker_centers.keys()),
    ])

    # Flush every frame so data isn't lost if the program crashes
    log_file.flush()

    # 12. Show the processed frame with grid, path, and overlays
    cf.cv2.imshow("Warped", warped)

    # Press 'q' to exit the loop
    if cf.cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cf.cap.release()
cf.cv2.destroyAllWindows()