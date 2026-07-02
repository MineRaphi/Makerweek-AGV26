import camera_feed as cf
import path_algorithm as pa
import drive as d
import logger
from replay import Recorder
from config import *

turn_counter = 0
agv_reachable = True

# --- Start logger (writes to a "logs/" subfolder) ---
if LOGGING_ENABLED:
    logger.init(log_dir=LOG_DIR)

# --- Start replay recorder ---
if REPLAY_ENABLED:
    recorder = Recorder(fps=12, resolution=(WARP_WIDTH, WARP_HEIGHT))
    recorder.start()

# --- Connect to the AGV's motor controller ---
try:
    d.enable_wheels()
except:
    # If the AGV isn't connected (e.g. testing the vision pipeline only),
    # don't crash — just continue without drive control
    print("No AGV connection")
    agv_reachable = False

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

    # 5.5 Inflate obstacles so the AGV doesn't drive too close to the line.
    kernel = cf.cv2.getStructuringElement(cf.cv2.MORPH_RECT, (2 * INFLATION_RADIUS + 1, 2 * INFLATION_RADIUS + 1))
    inflated_grid = cf.cv2.dilate(grid.astype('uint8'), kernel, iterations=1)

    # 6. Compute how far every free cell is from the nearest obstacle/edge
    #    Used by A* to prefer paths that stay away from walls
    dist_map = pa.compute_distance_map(inflated_grid)

    # 7. Find the AGV's current position on the grid (using its marker, mapped through M)
    agv_pos = cf.marker_to_grid(AGV_MARKER_ID, M, warped.shape, COLS, ROWS)

    if agv_pos is not None:
        start = agv_pos
    else:
        # Fallback starting position if the AGV marker isn't visible this frame
        start = (ROWS // 2, 0)

    # 8. Run A* pathfinding from the AGV's position to any cell on the right edge,
    #    preferring routes with more clearance from obstacles (clearance_weight)
    path = pa.astar(inflated_grid, start, dist_map, clearance_weight=6.0)

    # 9. Draw the grid and the computed path onto the warped image (for visualization/debugging)
    warped = cf.draw_grid(warped, grid, COLS, ROWS)
    warped = pa.draw_path(cf.cv2, warped, path, inflated_grid, COLS, ROWS)

    # --- Steering ---
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
        segment = pa.get_next_segment(path, agv_pos, inflated_grid.shape, warped.shape, COLS, ROWS, lookahead=6)
        if segment is not None:
            target_angle, distance, target_cell = segment
    else:
        segment = None
        print("AGV marker not detected, skipping steering this frame")
        if agv_reachable:
            d.move_mm(100)

    # 11. If we have a valid steering target, decide whether to turn or drive forward
    if segment is not None:
        target_angle, distance, target_cell = segment
        print(f"Steer to angle: {target_angle:.1f}°  distance: {distance:.1f}px")

        # Get the AGV's current heading from its marker orientation
        agv_dir = cf.get_marker_direction(AGV_MARKER_ID)

        if agv_dir is not None and agv_reachable:
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
                    d.move_mm(distance * PIXEL_PER_MM)
    elif agv_reachable:
        d.move_mm(100)


    # --- Log this frame ---
    if LOGGING_ENABLED:
        logger.write_frame(
            agv_pos        = agv_pos,
            agv_angle      = agv_angle,
            target_angle   = target_angle,
            target_distance= target_distance,
            target_cell    = target_cell,
            turn_amount    = turn_amount,
            action         = action,
            action_value   = action_value,
            path           = path,
            lines          = lines,
            marker_centers = cf.marker_centers,
        )

    # 12. Show the processed frame with grid, path, and overlays
    cf.cv2.imshow("Warped", warped)

    if REPLAY_ENABLED:
        recorder.record_frame(
            frame          = warped,
            agv_pos        = agv_pos,
            agv_angle      = agv_angle,
            target_angle   = target_angle,
            target_distance= target_distance,
            target_cell    = target_cell,
            turn_amount    = turn_amount,
            action         = action,
            action_value   = action_value,
            path           = path,
            lines          = lines,
            marker_centers = cf.marker_centers,
        )

    # Press 'q' to exit the loop
    if cf.cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
if LOGGING_ENABLED:
    logger.close()

if REPLAY_ENABLED:
    recorder.stop()

cf.cap.release()
cf.cv2.destroyAllWindows()

if agv_reachable:
    d.disable_wheels()