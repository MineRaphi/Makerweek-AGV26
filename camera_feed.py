import cv2
import cv2.aruco as aruco
import numpy as np
from config import *


# --- Camera stream setup ---
stream_url = "http://10.250.150.224:8081"
cap = cv2.VideoCapture(stream_url)

# Stores the latest known pixel position of each detected marker (by ID)
marker_centers = {}
# Stores the latest known 4 corner points of each detected marker (by ID)
marker_corners = {}


def detect_markers(frame, gray):
    """
    Detects all ArUco markers in the grayscale frame, draws their outlines
    on the color frame, and updates marker_centers / marker_corners with
    the latest detected positions.
    """
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        # Draw outlines + IDs on frame (for visualization/debugging)
        aruco.drawDetectedMarkers(frame, corners, ids)

        for i, marker_id in enumerate(ids.flatten()):
            c = corners[i][0]  # the 4 (x, y) corner points of this marker

            # Compute the marker's center as the average of its 4 corners
            center_x = int(c[:, 0].mean())
            center_y = int(c[:, 1].mean())
            print(f"Marker ID: {marker_id}  center: ({center_x}, {center_y})")

            # Save this marker's position for use by other functions
            marker_centers[marker_id] = [center_x, center_y]
            marker_corners[marker_id] = corners[i][0]


def get_marker_direction(marker_id):
    """
    Computes the heading (angle) a given marker is facing, based on its
    corner orientation. Returns None if the marker hasn't been detected.

    Returns: (angle_degrees, center_point, arrow_tip_point)
    """
    if marker_id not in marker_corners:
        return None  # marker not currently visible

    tl, tr, br, bl = marker_corners[marker_id]

    # "Forward" direction = vector from the bottom edge midpoint to the top edge midpoint
    # (Originally based on top of marker = forward, adjusted to fix a 90° offset)
    top_center    = (tl + tr) / 2
    bottom_center = (bl + br) / 2
    direction     = top_center - bottom_center

    # Center of the marker = average of all 4 corners
    center = (tl + tr + br + bl) / 4
    # Arrow tip = a point extended out from the center in the direction the marker faces
    tip    = center + direction * 1.5

    # Convert direction vector to an angle in degrees
    # (0° = pointing right, 90° = pointing up — y is flipped because image coords grow downward)
    angle  = np.degrees(np.arctan2(-direction[1], direction[0]))

    return angle, center.astype(int), tip.astype(int)


def draw_marker_direction(marker_id, frame):
    """Draws a direction arrow and angle label for the given marker on the frame."""
    result = get_marker_direction(marker_id)
    if result is None:
        return  # marker not visible this frame, nothing to draw

    angle, center, tip = result

    # Draw an arrow from the marker's center pointing in its facing direction
    cv2.arrowedLine(frame,
        tuple(center),
        tuple(tip),
        (0, 255, 0), 2, tipLength=0.3)

    # Label the marker with its ID and current heading angle
    label_pos = (center[0] + 10, center[1] - 10)
    cv2.putText(frame, f"ID{marker_id}: {angle:.1f}deg",
        label_pos,
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    print(f"Marker {marker_id} pointing at {angle:.1f}°")


def flatten_image(frame):
    """
    Uses the 4 corner ArUco markers to compute a perspective transform,
    then warps the camera frame into a flat top-down view of the field.

    Returns: (warped_image, transform_matrix M)
    M is returned so other functions can map marker positions into
    the same warped coordinate space.
    """
    # Source points: where the 4 corner markers currently are in the raw frame
    src_points = np.float32([
        marker_centers[ID_TOP_LEFT],
        marker_centers[ID_TOP_RIGHT],
        marker_centers[ID_BOTTOM_RIGHT],
        marker_centers[ID_BOTTOM_LEFT],
    ])

    # Destination points: where those corners should land in the output image
    # (i.e. the 4 corners of a clean rectangle)
    w, h = WARP_WIDTH, WARP_HEIGHT
    dst_points = np.float32([[0,0], [w,0], [w,h], [0,h]])

    # Compute the transform matrix and apply it to get a flat, undistorted view
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(frame, M, (w, h))

    return warped, M


def detect_blue_lines(warped):
    """
    Detects blue obstacle lines in the warped (flat) image using HSV color masking.

    Returns: (annotated_warped_image, binary_mask, list_of_detected_line_info)
    """
    # Convert to HSV — easier to isolate a specific color range than in BGR
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

    lower_blue = LOWER_BLUE
    upper_blue = UPPER_BLUE
    mask = cv2.inRange(hsv, lower_blue, upper_blue)  # white = blue pixels, black = everything else

    # Clean up the mask to remove noise and fill small gaps
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # remove small speckles
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill small holes

    # Find separate blue regions (each one is a "line" or blob)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines_info = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 500:  # ignore tiny blobs / noise
            continue

        # Fit a straight line through this contour's points to get its angle
        [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        angle = np.degrees(np.arctan2(float(vy[0]), float(vx[0])))

        # Draw the line across its bounding box, for visualization
        x1, y1, w, h = cv2.boundingRect(cnt)
        x2, y2 = x1 + w, y1 + h
        cv2.line(warped, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label the line with its angle
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
    """
    Divides the warped image into a cols x rows grid and marks each cell
    as free (0) or blocked (1) based on how much blue is inside it.

    Returns: 2D numpy array of shape (rows, cols)
    """
    h, w = warped.shape[:2]

    # Re-detect blue pixels (same as detect_blue_lines, but without the drawing)
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    lower_blue = LOWER_BLUE
    upper_blue = UPPER_BLUE
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    grid = np.zeros((rows, cols), dtype=np.uint8)

    # Use linspace (not integer division) so the grid cells always exactly
    # cover the full image, with no leftover pixels at the right/bottom edge
    row_edges = np.linspace(0, h, rows + 1, dtype=int)
    col_edges = np.linspace(0, w, cols + 1, dtype=int)

    for row in range(rows):
        for col in range(cols):
            # Crop out the mask region belonging to this grid cell
            cell = mask[row_edges[row]:row_edges[row+1],
                        col_edges[col]:col_edges[col+1]]
            cell_area = cell.size
            if cell_area == 0:
                continue

            # If more than 20% of this cell is blue, mark it as blocked
            blue_ratio = np.count_nonzero(cell) / cell_area
            if blue_ratio > 0.2:
                grid[row, col] = 1

    return grid


def draw_grid(warped, grid, cols, rows):
    """
    Draws the grid overlay on the warped image — red filled cells for
    obstacles, thin gray outlines for free cells. Used for visualization.
    """
    h, w = warped.shape[:2]

    row_edges = np.linspace(0, h, rows + 1, dtype=int)
    col_edges = np.linspace(0, w, cols + 1, dtype=int)

    for row in range(rows):
        for col in range(cols):
            x1, y1 = col_edges[col],   row_edges[row]
            x2, y2 = col_edges[col+1], row_edges[row+1]

            if grid[row, col] == 1:
                # Filled red rectangle = blocked cell
                cv2.rectangle(warped, (x1, y1), (x2, y2), (0, 0, 255), -1)
            else:
                # Thin gray outline = free cell
                cv2.rectangle(warped, (x1, y1), (x2, y2), (50, 50, 50), 1)

    return warped


def warp_point(point, M):
    """
    Transforms a single (x, y) point from the raw frame's coordinate space
    into the warped frame's coordinate space, using the same matrix M
    produced by flatten_image().
    """
    px = np.array([[point]], dtype=np.float32)  # shape (1, 1, 2) as required by cv2
    warped_pt = cv2.perspectiveTransform(px, M)
    return warped_pt[0][0]  # extract the (x, y) result


def marker_to_grid(marker_id, M, warped_shape, cols, rows):
    """
    Correctly converts a marker's RAW frame position into grid coordinates
    by first warping it through M into the flattened image space, then
    scaling it into grid cell indices.

    This is the preferred function (over get_agv_pos) for finding where
    the AGV currently sits on the grid.

    Returns: (row, col) or None if marker not detected.
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


# --- ArUco detector setup ---
# Dictionary must match whatever family/size was used to generate your printed markers
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)
