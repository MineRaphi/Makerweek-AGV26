import heapq
import numpy as np

def compute_distance_map(grid):
    dist = np.full(grid.shape, np.inf)
    queue = []

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            # Treat obstacles AND edges as distance 0
            if grid[r, c] == 1 or r == 0 or c == 0 or r == grid.shape[0]-1 or c == grid.shape[1]-1:
                dist[r, c] = 0
                heapq.heappush(queue, (0, r, c))

    # Rest stays the same
    while queue:
        d, r, c = heapq.heappop(queue)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                nd = d + 1
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    heapq.heappush(queue, (nd, nr, nc))

    return dist

def astar(grid, start, dist_map, clearance_weight=3.0):
    """
    A* from start to ANY cell on the right edge.
    """
    rows, cols = grid.shape
    max_dist = dist_map.max() or 1

    def heuristic(r, c):
        # Distance to the right edge instead of a fixed goal
        return cols - 1 - c

    def cell_cost(r, c):
        clearance = dist_map[r, c] / max_dist
        return 1 + clearance_weight * (1 - clearance)

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        r, c = current

        # Stop as soon as we reach the right edge
        if c == cols - 1:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr, nc] == 1:
                continue

            move_cost = 1.4 if (dr != 0 and dc != 0) else 1.0
            tentative_g = g_score[current] + move_cost * cell_cost(nr, nc)

            if tentative_g < g_score.get((nr, nc), np.inf):
                came_from[(nr, nc)] = current
                g_score[(nr, nc)] = tentative_g
                f = tentative_g + heuristic(nr, nc)
                heapq.heappush(open_set, (f, (nr, nc)))

    return None

def draw_path(cv2, warped, path, grid, cols=20, rows=20):
    if path is None:
        cv2.putText(warped, "No path found!", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return warped

    h, w = warped.shape[:2]

    row_edges = np.linspace(0, h, rows + 1, dtype=int)
    col_edges = np.linspace(0, w, cols + 1, dtype=int)

    def cell_center(r, c):
        cx = (col_edges[c] + col_edges[c+1]) // 2
        cy = (row_edges[r] + row_edges[r+1]) // 2
        return cx, cy

    for i, (r, c) in enumerate(path):
        cx, cy = cell_center(r, c)
        cv2.circle(warped, (cx, cy), 4, (0, 255, 0), -1)

        if i + 1 < len(path):
            nr, nc = path[i + 1]
            nx, ny = cell_center(nr, nc)
            cv2.line(warped, (cx, cy), (nx, ny), (0, 255, 0), 2)

    sr, sc = path[0]
    gr, gc = path[-1]
    cv2.circle(warped, cell_center(sr, sc), 8, (255, 0, 0), -1)
    cv2.circle(warped, cell_center(gr, gc), 8, (0, 0, 255), -1)

    return warped

def get_next_segment(path, current_pos, grid_shape, warped_shape, cols, rows, lookahead=3):
    """
    Given the current path and the AGV's current grid position,
    returns the angle and distance to steer towards.

    lookahead: how many cells ahead on the path to target
               (helps smooth out steering instead of chasing the very next cell)

    Returns: (angle_degrees, distance_pixels, target_cell) or None if path is too short
    """
    if path is None or len(path) < 2:
        return None
    
    if current_pos is None:
        return None  # AGV marker not detected this frame

    # Find the closest point on the path to where the AGV currently is
    closest_idx = min(
        range(len(path)),
        key=lambda i: (path[i][0]-current_pos[0])**2 + (path[i][1]-current_pos[1])**2
    )

    # Look ahead a few cells from the closest point (clamped to path length)
    target_idx = min(closest_idx + lookahead, len(path) - 1)
    target_cell = path[target_idx]

    # Convert both points from grid coords to pixel coords (warped image space)
    h, w = warped_shape[:2]
    row_edges = np.linspace(0, h, rows + 1, dtype=int)
    col_edges = np.linspace(0, w, cols + 1, dtype=int)

    def cell_center(r, c):
        cx = (col_edges[c] + col_edges[c+1]) // 2
        cy = (row_edges[r] + row_edges[r+1]) // 2
        return cx, cy

    cur_x, cur_y = cell_center(*current_pos)
    tgt_x, tgt_y = cell_center(*target_cell)

    dx = tgt_x - cur_x
    dy = tgt_y - cur_y

    distance = np.hypot(dx, dy)
    angle = np.degrees(np.arctan2(-dy, dx))  # 0° = right, 90° = up (matches your marker angle convention)

    return angle, distance, target_cell
