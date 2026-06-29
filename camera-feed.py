import cv2
import cv2.aruco as aruco
import numpy as np

ID_BOTTOM_LEFT = 1
ID_TOP_LEFT = 2
ID_TOP_RIGHT = 3
ID_BOTTOM_RIGHT = 4

# --- Setup ---
stream_url = "http://10.250.150.224:8081"
cap = cv2.VideoCapture(stream_url)
marker_centers = {}

def detectmarkers():
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        # Draw outlines + IDs on frame
        aruco.drawDetectedMarkers(frame, corners, ids)

        for i, marker_id in enumerate(ids.flatten()):
            # corners[i] shape: (1, 4, 2) — four (x, y) corner points
            c = corners[i][0]
            center_x = int(c[:, 0].mean())
            center_y = int(c[:, 1].mean())
            print(f"Marker ID: {marker_id}  center: ({center_x}, {center_y})")

            marker_centers[marker_id] = [center_x, center_y]

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

    # Detect markers
    detectmarkers()
    warped = flatten_image()

    cv2.imshow("ArUco Detection", warped)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

