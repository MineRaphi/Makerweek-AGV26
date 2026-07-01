import numpy as np

AGV_MARKER_ID = 21   # ArUco ID printed on the AGV itself (used to track its position/heading)
COLS = 80            # number of grid columns to divide the playing field into
ROWS = 60            # number of grid rows to divide the playing field into
INFLATION_RADIUS = 2 # radius increase around obstacles

# --- ArUco marker IDs that mark the four corners of the playing field ---
# These are used to compute the perspective warp
ID_BOTTOM_LEFT = 1
ID_TOP_LEFT = 2
ID_TOP_RIGHT = 3
ID_BOTTOM_RIGHT = 4

# --- HSV color range used to detect the blue obstacle lines ---
LOWER_BLUE = np.array([100, 100, 200])
UPPER_BLUE = np.array([130, 200, 255])

# --- Output size (in pixels) of the flattened/warped top-down view ---
WARP_WIDTH = 2000
WARP_HEIGHT = 1300

# --- AGV HTTP API endpoint ---
AGV_BASE_URL = "http://172.17.1.61/api/agv"

# --- Physical robot dimensions (in mm) ---
WHEEL_DIAMETER = 95     # diameter of the drive wheels
WHEEL_DISTANCE = 210    # distance between the left and right wheels (track width)

# --- Field calibration ---
# Real-world distance (mm) between the corner ArUco markers, used to
# relate pixel measurements in the warped image to real-world distances
MAX_COL = 288
MAX_ROW = 188

# --- Stepper motor calibration ---
MAX_STEPS = 3550        # number of motor steps for one full AGV rotation (360°) when wheels turn opposite directions
STEPS_PER_DEGREE = 10   # motor steps needed to rotate the AGV by 1 degree
STEPS_PER_MM = 3        # motor steps needed to drive forward 1 mm
PIXEL_PER_MM = 6.944    # conversion factor: how many image pixels correspond to 1 mm on the real field

LOGGING_ENABLED = True
LOG_DIR = "logs"