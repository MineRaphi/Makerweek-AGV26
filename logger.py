import csv
import os
from datetime import datetime

_log_file = None
_log_writer = None


def init(log_dir="."):
    """
    Opens a new CSV log file named with the current timestamp.
    Call this once at the start of your program before any write_frame() calls.
    """
    global _log_file, _log_writer

    os.makedirs(log_dir, exist_ok=True)
    filename = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    _log_file   = open(filename, "w", newline="")
    _log_writer = csv.writer(_log_file)

    # Write the header row
    _log_writer.writerow([
        "timestamp",
        "agv_grid_row", "agv_grid_col",
        "agv_angle",
        "target_angle", "target_distance",
        "target_cell_row", "target_cell_col",
        "turn_amount",
        "action",
        "action_value",
        "path_length",
        "blue_lines_found",
        "marker_ids_visible",
    ])

    print(f"Logging to: {filename}")
    return filename


def write_frame(
    agv_pos=None,
    agv_angle=None,
    target_angle=None,
    target_distance=None,
    target_cell=None,
    turn_amount=None,
    action="none",
    action_value=None,
    path=None,
    lines=None,
    marker_centers=None,
):
    """
    Writes one row to the log file for the current frame.
    All parameters are optional — missing values are written as empty cells.
    Call this once per frame at the end of your main loop.
    """
    if _log_writer is None:
        raise RuntimeError("Logger not initialised — call logger.init() first.")

    _log_writer.writerow([
        datetime.now().isoformat(),
        agv_pos[0] if agv_pos else "",
        agv_pos[1] if agv_pos else "",
        round(agv_angle, 2)       if agv_angle       is not None else "",
        round(target_angle, 2)    if target_angle     is not None else "",
        round(target_distance, 2) if target_distance  is not None else "",
        target_cell[0]            if target_cell      else "",
        target_cell[1]            if target_cell      else "",
        round(turn_amount, 2)     if turn_amount      is not None else "",
        action,
        action_value              if action_value      is not None else "",
        len(path)                 if path              is not None else 0,
        len(lines)                if lines             is not None else 0,
        " ".join(str(k) for k in marker_centers.keys()) if marker_centers else "",
    ])

    # Flush immediately so data isn't lost if the program crashes mid-run
    _log_file.flush()


def close():
    """Closes the log file cleanly. Call this when the main loop exits."""
    global _log_file, _log_writer

    if _log_file is not None:
        _log_file.close()
        _log_file   = None
        _log_writer = None
        print("Log file closed.")