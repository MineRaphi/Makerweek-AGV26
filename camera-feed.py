import cv2
import cv2.aruco as aruco
import numpy as np

ID_BOTTOM_LEFT = 1
ID_TOP_LEFT = 2
ID_TOP_RIGHT = 3
ID_BOTTOM_RIGHT = 4

AGV_MARKER_ID = 8

# --- Setup ---
stream_url = "http://10.250.150.224:8081"
cap = cv2.VideoCapture(stream_url)
marker_centers = {}
marker_corners = {}

def detect_markers():
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
    right_center = (tr + br) / 2
    left_center  = (tl + bl) / 2
    direction    = right_center - left_center  # now points "right" on the marker

    center = (tl + tr + br + bl) / 4
    tip    = center + direction * 1.5
    angle  = np.degrees(np.arctan2(-direction[1], direction[0]))

    return angle, center.astype(int), tip.astype(int)

def draw_marker_direction(marker_id):
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


def flatten_image():
    # After detecting markers, collect the center (or a specific corner) of each ArUco
    # You need to map each marker ID to a role: top-left, top-right, etc.
    src_points = np.float32([
        marker_centers[ID_TOP_LEFT],
        marker_centers[ID_TOP_RIGHT],
        marker_centers[ID_BOTTOM_RIGHT],
        marker_centers[ID_BOTTOM_LEFT],
    ])

    # Define where you want those points to land in the output image
    w, h = 1000, 600
    dst_points = np.float32([[0,0], [w,0], [w,h], [0,h]])

    # Compute the transform and apply it
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(frame, M, (w, h))

    return warped



# Pick the dictionary that matches your printed markers
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

     # 1. Detect all markers on the raw frame
    detect_markers()

    # 2. Draw direction arrow for your chosen marker (on raw frame)
    draw_marker_direction(AGV_MARKER_ID)

    # 3. Warp to flat view
    warped = flatten_image()

    cv2.imshow("ArUco Detection", warped)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

