import heapq
import numpy as np


def compute_distance_map(grid):
    """
    For every cell in the grid, computes how many steps away it is from
    the nearest obstacle OR the field edge (multi-source BFS).

    Free cells deep in open space get a high value; cells near walls or
    edges get a low value (close to 0). Used later to bias pathfinding
    away from obstacles and edges.

    Returns: 2D numpy array of the same shape as grid, with distance values.
    """
    dist = np.full(grid.shape, np.inf)
    queue = []

    # Seed the BFS: every obstacle cell AND every edge cell starts at distance 0
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 1 or r == 0 or c == 0 or r == grid.shape[0]-1 or c == grid.shape[1]-1:
                dist[r, c] = 0
                heapq.heappush(queue, (0, r, c))

    # Expand outward from all seed cells simultaneously (multi-source BFS),
    # so each free cell ends up with the distance to its NEAREST obstacle/edge
    while queue:
        d, r, c = heapq.heappop(queue)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:  # check up/down/left/right neighbors
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                nd = d + 1
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    heapq.heappush(queue, (nd, nr, nc))

    return dist


def astar(grid, start, dist_map, clearance_weight=3.0):
    """
    Finds a path from `start` to ANY free cell on the right edge of the grid,
    using A* search. Prefers paths that stay further from obstacles/edges,
    controlled by clearance_weight.

    grid:             2D array, 0 = free, 1 = blocked
    start:            (row, col) starting cell
    dist_map:         output of compute_distance_map() — distance to nearest obstacle/edge
    clearance_weight: how strongly to prefer open space over the shortest route.
                       0 = pure shortest path, higher = hugs the center more

    Returns: list of (row, col) cells forming the path, or None if no path exists.
    """
    rows, cols = grid.shape
    max_dist = dist_map.max() or 1  # avoid divide-by-zero if dist_map is all zeros

    def heuristic(r, c):
        # Estimated remaining distance to the goal = distance to the right edge
        # (since the goal is "any cell in the last column", not a fixed point)
        return cols - 1 - c

    def cell_cost(r, c):
        # Cost of entering this cell: cells close to obstacles/edges are more expensive,
        # cells with lots of clearance are cheaper — this is what pulls the path
        # towards the middle of open spaces
        clearance = dist_map[r, c] / max_dist  # normalized 0 (touching wall) -> 1 (max clearance)
        return 1 + clearance_weight * (1 - clearance)

    # Standard A* open set, using a min-heap keyed by f-score
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}              # for reconstructing the path at the end
    g_score = {start: 0}        # cost of cheapest known path to each cell

    while open_set:
        _, current = heapq.heappop(open_set)
        r, c = current

        # Goal check: reaching ANY cell in the last column counts as success
        if c == cols - 1:
            # Reconstruct the path by walking backwards through came_from
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # reverse to get start -> goal order

        # Check all 8 neighboring cells (orthogonal + diagonal movement)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue  # out of bounds
            if grid[nr, nc] == 1:
                continue  # blocked cell, can't move here

            # Diagonal moves are slightly more expensive (true distance is sqrt(2) ≈ 1.4)
            move_cost = 1.4 if (dr != 0 and dc != 0) else 1.0
            tentative_g = g_score[current] + move_cost * cell_cost(nr, nc)

            # If this is a cheaper way to reach (nr, nc) than previously found, update it
            if tentative_g < g_score.get((nr, nc), np.inf):
                came_from[(nr, nc)] = current
                g_score[(nr, nc)] = tentative_g
                f = tentative_g + heuristic(nr, nc)  # f = known cost + estimated remaining cost
                heapq.heappush(open_set, (f, (nr, nc)))

    return None  # open set exhausted without reaching the right edge — no path exists


def draw_path(cv2, warped, path, grid, cols=20, rows=20):
    """
    Draws the computed path onto the warped image: a line connecting
    each cell center, with markers for the start (blue) and end (red).
    If no path was found, displays a warning message instead.
    """
    if path is None:
        cv2.putText(warped, "No path found!", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return warped

    h, w = warped.shape[:2]

    # Same evenly-spaced grid boundaries used elsewhere, so drawing lines up with the grid
    row_edges = np.linspace(0, h, rows + 1, dtype=int)
    col_edges = np.linspace(0, w, cols + 1, dtype=int)

    def cell_center(r, c):
        """Pixel coordinates of the center of grid cell (r, c)."""
        cx = (col_edges[c] + col_edges[c+1]) // 2
        cy = (row_edges[r] + row_edges[r+1]) // 2
        return cx, cy

    # Draw a dot at each path cell, and a line connecting it to the next one
    for i, (r, c) in enumerate(path):
        cx, cy = cell_center(r, c)
        cv2.circle(warped, (cx, cy), 4, (0, 255, 0), -1)

        if i + 1 < len(path):
            nr, nc = path[i + 1]
            nx, ny = cell_center(nr, nc)
            cv2.line(warped, (cx, cy), (nx, ny), (0, 255, 0), 2)

    # Highlight the start (blue) and end (red) of the path
    sr, sc = path[0]
    gr, gc = path[-1]
    cv2.circle(warped, cell_center(sr, sc), 8, (255, 0, 0), -1)
    cv2.circle(warped, cell_center(gr, gc), 8, (0, 0, 255), -1)

    return warped


def get_next_segment(path, current_pos, grid_shape, warped_shape, cols, rows, lookahead=3):
    """
    Given the planned path and the AGV's current grid position, computes
    a steering target: the angle to point towards and the distance to it.

    lookahead: how many cells ahead of the AGV's closest path point to aim for.
               Targeting a point further ahead (instead of the very next cell)
               produces smoother steering instead of jittery zigzag movement.

    Returns: (angle_degrees, distance_pixels, target_cell), or None if the
             path is too short or the AGV's position isn't known this frame.
    """
    if path is None or len(path) < 2:
        return None  # no usable path

    if current_pos is None:
        return None  # AGV marker not detected this frame

    # Find which point on the path is closest to where the AGV actually is right now
    # (the AGV may have drifted slightly off the planned path)
    closest_idx = min(
        range(len(path)),
        key=lambda i: (path[i][0]-current_pos[0])**2 + (path[i][1]-current_pos[1])**2
    )

    # Pick a target a few cells further along the path than the closest point,
    # clamped so it doesn't go past the end of the path
    target_idx = min(closest_idx + lookahead, len(path) - 1)
    target_cell = path[target_idx]

    # Convert grid cell coordinates into pixel coordinates (warped image space)
    h, w = warped_shape[:2]
    row_edges = np.linspace(0, h, rows + 1, dtype=int)
    col_edges = np.linspace(0, w, cols + 1, dtype=int)

    def cell_center(r, c):
        cx = (col_edges[c] + col_edges[c+1]) // 2
        cy = (row_edges[r] + row_edges[r+1]) // 2
        return cx, cy

    cur_x, cur_y = cell_center(*current_pos)
    tgt_x, tgt_y = cell_center(*target_cell)

    # Vector from the AGV's current position to the target point
    dx = tgt_x - cur_x
    dy = tgt_y - cur_y

    distance = np.hypot(dx, dy)  # straight-line pixel distance to the target

    # Angle of that vector, using the same convention as get_marker_direction():
    # 0° = pointing right, 90° = pointing up (y is flipped since image coords grow downward)
    angle = np.degrees(np.arctan2(-dy, dx))

    return angle, distance, target_cell
