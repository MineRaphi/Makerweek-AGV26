import camera_feed as cf
import path_algorithm as pa
import drive as d

AGV_MARKER_ID = 21
COLS = 80
ROWS = 60

try:
    d.enable_wheels()
except:
    print("No AGV connection")

while True:
    ret, frame = cf.cap.read()
    if not ret:
        break

    gray = cf.cv2.cvtColor(frame, cf.cv2.COLOR_BGR2GRAY)

     # 1. Detect all markers on the raw frame
    cf.detect_markers(frame, gray)

    # 2. Draw direction arrow for your chosen marker (on raw frame)
    cf.draw_marker_direction(AGV_MARKER_ID, frame)

    # 3. Warp to flat view
    warped, M = cf.flatten_image(frame)

    warped, mask, lines = cf.detect_blue_lines(warped)

    grid   = cf.create_grid(warped, COLS, ROWS)
    dist_map = pa.compute_distance_map(grid)

    agv_pos = cf.marker_to_grid(AGV_MARKER_ID, M, warped.shape, COLS, ROWS)

    if agv_pos is not None:
        start = agv_pos
    else:
        start = (ROWS // 2, 0)  # fallback if marker not visible

    path = pa.astar(grid, start, dist_map, clearance_weight=6.0)

    warped = cf.draw_grid(warped, grid, COLS, ROWS)
    warped = pa.draw_path(cf.cv2, warped, path, grid, COLS, ROWS)

    if agv_pos is not None:
        segment = pa.get_next_segment(path, agv_pos, grid.shape, warped.shape, COLS, ROWS, lookahead=3)
        if segment is not None:
            target_angle, distance, target_cell = segment
            print(f"Steer to angle: {target_angle:.1f}°  distance: {distance:.1f}px")
    else:
        segment = None
        print("AGV marker not detected, skipping steering this frame")

    if segment is not None:
        target_angle, distance, target_cell = segment
        print(f"Steer to angle: {target_angle:.1f}°  distance: {distance:.1f}px")

        # Compare with current AGV heading to know how much to turn
        agv_dir = cf.get_marker_direction(AGV_MARKER_ID)
        if agv_dir is not None:
            current_angle = agv_dir[0]
            turn_amount = target_angle - current_angle
            turn_amount = (turn_amount + 180) % 360 - 180  # normalize to -180..180
            print(f"Direction off by: {turn_amount:.1f}°")

            if turn_amount > 5 or turn_amount < -5:
                d.rotate(turn_amount * 0.9)
            else:
                d.move_mm(distance * d.PIXEL_PER_MM)

    cf.cv2.imshow("Warped", warped)
    if cf.cv2.waitKey(1) & 0xFF == ord('q'):
        break

cf.cap.release()
cf.cv2.destroyAllWindows()
