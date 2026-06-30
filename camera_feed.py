import cv2
import cv2.aruco as aruco
import numpy as np

ID_BOTTOM_LEFT = 1
ID_TOP_LEFT = 2
ID_TOP_RIGHT = 3
ID_BOTTOM_RIGHT = 4

LOWER_BLUE = np.array([100, 100, 200])
UPPER_BLUE = np.array([130, 200, 255])
WARP_WIDTH = 2000
WARP_HEIGHT = 1300

# --- Setup ---
stream_url = "http://10.250.150.224:8081"
cap = cv2.VideoCapture(stream_url)
marker_centers = {}
marker_corners = {}

def detect_markers(frame, gray):
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        # Draw outlines + IDs on frame
        aruco.drawDetectedMarkers(frame, corners, ids)

        for i, marker_id in enumerate(ids.flatten()):
            c = corners[i][0]
            center_x = int(c[:, 0].mean())
            center_y = int(c[:, 1].mean())
            print(f"Marker ID: {marker_id}  center: ({center_x}, {center_y})")

            marker_centers[marker_id] = [center_x, center_y]
            marker_corners[marker_id] = corners[i][0]

def get_marker_direction(marker_id):
    if marker_id not in marker_corners:
        return None

    tl, tr, br, bl = marker_corners[marker_id]

    # Rotate which sides we use to define "forward"
    # Original: top of marker = forward
    # Swap to:  right of marker = forward  (fixes 90° offset)
    top_center    = (tl + tr) / 2
    bottom_center = (bl + br) / 2
    direction     = top_center - bottom_center

    center = (tl + tr + br + bl) / 4
    tip    = center + direction * 1.5
    angle  = np.degrees(np.arctan2(-direction[1], direction[0]))

    return angle, center.astype(int), tip.astype(int)

def draw_marker_direction(marker_id, frame):
    """Draw a direction arrow and angle label for the given marker."""
    result = get_marker_direction(marker_id)
    if result is None:
        return  # marker not visible this frame

    angle, center, tip = result

    # Draw the arrow
    cv2.arrowedLine(frame,
        tuple(center),
        tuple(tip),
        (0, 255, 0), 2, tipLength=0.3)

    # Draw the angle label next to the marker
    label_pos = (center[0] + 10, center[1] - 10)
    cv2.putText(frame, f"ID{marker_id}: {angle:.1f}deg",
        label_pos,
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    print(f"Marker {marker_id} pointing at {angle:.1f}°")

def flatten_image(frame):
    # After detecting markers, collect the center (or a specific corner) of each ArUco
    # You need to map each marker ID to a role: top-left, top-right, etc.
    src_points = np.float32([
        marker_centers[ID_TOP_LEFT],
        marker_centers[ID_TOP_RIGHT],
        marker_centers[ID_BOTTOM_RIGHT],
        marker_centers[ID_BOTTOM_LEFT],
    ])

    # Define where you want those points to land in the output image
    w, h = WARP_WIDTH, WARP_HEIGHT
    dst_points = np.float32([[0,0], [w,0], [w,h], [0,h]])

    # Compute the transform and apply it
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(frame, M, (w, h))

    return warped, M

def detect_blue_lines(warped):
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

    # Blue range in HSV
    lower_blue = LOWER_BLUE
    upper_blue = UPPER_BLUE
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # remove speckles
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill gaps

    # Find the lines within the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines_info = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 500:  # ignore tiny blobs
            continue

        # Fit a line through the contour
        [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        angle = np.degrees(np.arctan2(float(vy[0]), float(vx[0])))

        # Draw the line across the full contour bounding box
        x1, y1, w, h = cv2.boundingRect(cnt)
        x2, y2 = x1 + w, y1 + h
        cv2.line(warped, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw the angle
        cv2.putText(warped, f"{float(angle):.1f}deg",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        lines_info.append({
            "angle": float(angle),
            "center": (int(x[0]), int(y[0])),
            "contour": cnt,
        })

    return warped, mask, lines_info

def create_grid(warped, cols, rows):
    h, w = warped.shape[:2]

    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    lower_blue = LOWER_BLUE
    upper_blue = UPPER_BLUE
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    grid = np.zeros((rows, cols), dtype=np.uint8)

    # Use linspace to get evenly spaced boundaries that cover the full image
    row_edges = np.linspace(0, h, rows + 1, dtype=int)
    col_edges = np.linspace(0, w, cols + 1, dtype=int)

    for row in range(rows):
        for col in range(cols):
            cell = mask[row_edges[row]:row_edges[row+1],
                        col_edges[col]:col_edges[col+1]]
            cell_area = cell.size
            if cell_area == 0:
                continue
            blue_ratio = np.count_nonzero(cell) / cell_area
            if blue_ratio > 0.2:
                grid[row, col] = 1

    return grid

def draw_grid(warped, grid, cols, rows):
    h, w = warped.shape[:2]

    row_edges = np.linspace(0, h, rows + 1, dtype=int)
    col_edges = np.linspace(0, w, cols + 1, dtype=int)

    for row in range(rows):
        for col in range(cols):
            x1, y1 = col_edges[col],   row_edges[row]
            x2, y2 = col_edges[col+1], row_edges[row+1]

            if grid[row, col] == 1:
                cv2.rectangle(warped, (x1, y1), (x2, y2), (0, 0, 255), -1)
            else:
                cv2.rectangle(warped, (x1, y1), (x2, y2), (50, 50, 50), 1)

    return warped

def get_agv_pos(marker_id, warped_shape, cols, rows):
    """
    Returns (row, col) grid position of the given marker,
    or None if the marker isn't currently detected.
    """
    if marker_id not in marker_centers:
        return None

    x, y = marker_centers[marker_id]
    h, w = warped_shape[:2]

    col = int(x / w * cols)
    row = int(y / h * rows)

    # Clamp in case the marker is right at the edge
    col = min(max(col, 0), cols - 1)
    row = min(max(row, 0), rows - 1)

    return (row, col)

def warp_point(point, M):
    """Transforms a single (x, y) point through the perspective matrix M."""
    px = np.array([[point]], dtype=np.float32)  # shape (1, 1, 2)
    warped_pt = cv2.perspectiveTransform(px, M)
    return warped_pt[0][0]  # (x, y)

def marker_to_grid(marker_id, M, warped_shape, cols, rows):
    """
    Returns (row, col) grid position of the given marker in the WARPED image,
    or None if the marker isn't currently detected.
    """
    if marker_id not in marker_centers:
        return None

    raw_point = marker_centers[marker_id]
    warped_x, warped_y = warp_point(raw_point, M)

    h, w = warped_shape[:2]

    col = int(warped_x / w * cols)
    row = int(warped_y / h * rows)

    col = min(max(col, 0), cols - 1)
    row = min(max(row, 0), rows - 1)

    return (row, col)

# Pick the dictionary that matches your printed markers
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)
