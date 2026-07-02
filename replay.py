"""
replay.py — AGV Session Recorder and Replay Player

RECORDING:
    Import and use the Recorder class in main.py to capture frames + data.

    from replay import Recorder
    recorder = Recorder()
    recorder.start()

    # Inside main loop, after all processing:
    recorder.record_frame(frame=warped, agv_pos=agv_pos, ...)

    # On exit:
    recorder.stop()

PLAYBACK:
    Run this file directly to replay a saved session:

        python replay.py sessions/session_20260701_142305/

    Controls during playback:
        SPACE   — pause / resume
        LEFT    — step back one frame
        RIGHT   — step forward one frame
        Q       — quit
        +/-     — speed up / slow down
"""

import cv2
import csv
import os
import json
import time
import sys
from datetime import datetime
from config import *


# ── Recorder ─────────────────────────────────────────────────────────────────

class Recorder:
    """
    Records every frame of the warped image alongside its associated data
    into a session folder. Each session contains:
        - video.mp4       — all processed frames as a video
        - data.csv        — one row of data per frame
        - meta.json       — session metadata (fps, resolution, start time)
    """

    def __init__(self, session_dir="sessions", fps=20, resolution=(1000, 600)):
        """
        session_dir:  root folder where session subfolders are created
        fps:          frames per second for the output video
        resolution:   (width, height) of the warped frame — must match your WARP_WIDTH/WARP_HEIGHT
        """
        self.fps        = fps
        self.resolution = resolution
        self.session_path = None
        self.video_writer = None
        self.csv_file     = None
        self.csv_writer   = None
        self.frame_index  = 0
        self.start_time   = None
        self.session_dir  = session_dir

    def start(self):
        """
        Creates a new session folder and opens the video + CSV files for writing.
        Call this once before the main loop starts.
        """
        # Create a uniquely named session folder
        timestamp         = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_path = os.path.join(self.session_dir, f"session_{timestamp}")
        os.makedirs(self.session_path, exist_ok=True)

        # Set up the video writer (mp4v codec → .mp4)
        video_path   = os.path.join(self.session_path, "video.mp4")
        fourcc       = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(video_path, fourcc, self.fps, self.resolution)

        # Set up the CSV writer
        csv_path       = os.path.join(self.session_path, "data.csv")
        self.csv_file  = open(csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "frame_index",
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
            "at_goal",
        ])

        # Save session metadata
        meta = {
            "start_time": timestamp,
            "fps":        self.fps,
            "resolution": self.resolution,
        }
        with open(os.path.join(self.session_path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        self.start_time  = time.time()
        self.frame_index = 0

        print(f"Recording to: {self.session_path}")

    def record_frame(
        self,
        frame,
        agv_pos        = None,
        agv_angle      = None,
        target_angle   = None,
        target_distance= None,
        target_cell    = None,
        turn_amount    = None,
        action         = "none",
        action_value   = None,
        path           = None,
        lines          = None,
        marker_centers = None,
        at_goal        = False,
    ):
        """
        Records one frame. Call this once per iteration of your main loop,
        after all processing is done and the warped frame is fully annotated.

        frame: the annotated warped image (numpy array, BGR)
        All other parameters mirror the logger.write_frame() signature.
        """
        if self.video_writer is None:
            raise RuntimeError("Recorder not started — call recorder.start() first.")

        # Resize frame to the configured resolution if it doesn't match
        h, w = frame.shape[:2]
        if (w, h) != self.resolution:
            frame = cv2.resize(frame, self.resolution)

        # Write the frame into the video
        self.video_writer.write(frame)

        # Write the matching data row into the CSV
        self.csv_writer.writerow([
            self.frame_index,
            datetime.now().isoformat(),
            agv_pos[0]              if agv_pos        else "",
            agv_pos[1]              if agv_pos        else "",
            round(agv_angle, 2)     if agv_angle      is not None else "",
            round(target_angle, 2)  if target_angle   is not None else "",
            round(target_distance, 2) if target_distance is not None else "",
            target_cell[0]          if target_cell    else "",
            target_cell[1]          if target_cell    else "",
            round(turn_amount, 2)   if turn_amount    is not None else "",
            action,
            action_value            if action_value   is not None else "",
            len(path)               if path           is not None else 0,
            len(lines)              if lines          is not None else 0,
            " ".join(str(k) for k in marker_centers.keys()) if marker_centers else "",
            at_goal,
        ])

        self.csv_file.flush()
        self.frame_index += 1

    def stop(self):
        """
        Finalises and closes all files. Call this when the main loop exits.
        """
        if self.video_writer:
            self.video_writer.release()
        if self.csv_file:
            self.csv_file.close()

        duration = time.time() - self.start_time if self.start_time else 0
        print(f"Recording stopped. {self.frame_index} frames saved in {duration:.1f}s → {self.session_path}")


# ── Player ────────────────────────────────────────────────────────────────────

class Player:
    """
    Replays a recorded session, showing each video frame side-by-side with
    its corresponding CSV data in an overlay panel.

    Controls:
        SPACE       pause / resume
        LEFT / A    step back one frame
        RIGHT / D   step forward one frame
        + / =       increase playback speed
        - / _       decrease playback speed
        Q / ESC     quit
    """

    def __init__(self, session_path):
        self.session_path = session_path
        self.frames_data  = []   # list of dicts, one per frame
        self.cap          = None
        self.meta         = {}

    def load(self):
        """Loads the CSV data and opens the video file."""
        csv_path   = os.path.join(self.session_path, "data.csv")
        video_path = os.path.join(self.session_path, "video.mp4")
        meta_path  = os.path.join(self.session_path, "meta.json")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"No video found at {video_path}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"No data CSV found at {csv_path}")

        # Load metadata
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.meta = json.load(f)

        # Load all CSV rows into memory
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            self.frames_data = list(reader)

        # Open the video
        self.cap = cv2.VideoCapture(video_path)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Loaded session: {self.session_path}")
        print(f"  Frames: {total_frames}  |  Data rows: {len(self.frames_data)}")

    def _draw_overlay(self, frame, data, frame_idx, total_frames, paused, speed):
        """
        Draws a semi-transparent data panel on the left side of the frame
        showing all the recorded values for the current frame.
        """
        panel_w = 320
        overlay = frame.copy()

        # Dark background panel
        cv2.rectangle(overlay, (0, 0), (panel_w, frame.shape[0]), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        def put(text, row, color=(220, 220, 220)):
            cv2.putText(frame, text, (10, 24 + row * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

        # Header
        put(f"FRAME  {frame_idx + 1} / {total_frames}", 0, (100, 220, 100))
        put(f"{'PAUSED' if paused else f'SPEED {speed:.1f}x'}", 1,
            (80, 80, 255) if paused else (220, 180, 50))
        put(f"TIME   {data.get('timestamp','')[:19]}", 2, (160, 160, 160))

        put("─── AGV ───────────────────", 4, (100, 180, 255))
        put(f"Grid pos  row={data.get('agv_grid_row','?')}  col={data.get('agv_grid_col','?')}", 5)
        put(f"Angle     {data.get('agv_angle','?')}°", 6)
        put(f"At goal   {data.get('at_goal','?')}", 7,
            (50, 255, 50) if data.get("at_goal") == "True" else (220, 220, 220))

        put("─── PATH ──────────────────", 9, (100, 180, 255))
        put(f"Length    {data.get('path_length','?')} cells", 10)
        put(f"Target    row={data.get('target_cell_row','?')}  col={data.get('target_cell_col','?')}", 11)
        put(f"T.angle   {data.get('target_angle','?')}°", 12)
        put(f"T.dist    {data.get('target_distance','?')} px", 13)

        put("─── STEERING ──────────────", 15, (100, 180, 255))
        put(f"Turn amt  {data.get('turn_amount','?')}°", 16)
        action = data.get("action", "none")
        color  = (50, 200, 50) if action == "move" else (50, 100, 255) if action == "rotate" else (160,160,160)
        put(f"Action    {action}  ({data.get('action_value','?')})", 17, color)

        put("─── FIELD ─────────────────", 19, (100, 180, 255))
        put(f"Blue lines  {data.get('blue_lines_found','?')}", 20)
        put(f"Markers     {data.get('marker_ids_visible','?')}", 21)

        put("─── CONTROLS ──────────────", 23, (130, 130, 130))
        put("SPACE pause  ←/→ step", 24, (130, 130, 130))
        put("+/- speed    Q quit",   25, (130, 130, 130))

        return frame

    def play(self):
        """Starts the interactive replay window."""
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = self.meta.get("fps", 20)
        frame_idx    = 0
        paused       = False
        speed        = 1.0

        # Seek to a specific frame
        def seek(idx):
            idx = max(0, min(idx, total_frames - 1))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            return idx

        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("End of replay.")
                    break

                current_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

                # Match CSV row to this frame
                data = self.frames_data[current_idx] if current_idx < len(self.frames_data) else {}

                annotated = self._draw_overlay(frame, data, current_idx, total_frames, paused, speed)
                cv2.imshow("AGV Replay", annotated)
                frame_idx = current_idx

                delay = max(1, int((1000 / fps) / speed))
            else:
                delay = 30  # just poll for keypresses while paused

            key = cv2.waitKey(delay) & 0xFF

            if key == ord('q') or key == 27:        # Q or ESC — quit
                break
            elif key == ord(' '):                    # SPACE — pause/resume
                paused = not paused
            elif key == 83 or key == ord('d'):       # RIGHT or D — step forward
                paused    = True
                frame_idx = seek(frame_idx + 1)
                ret, frame = self.cap.read()
                if ret:
                    data      = self.frames_data[frame_idx] if frame_idx < len(self.frames_data) else {}
                    annotated = self._draw_overlay(frame, data, frame_idx, total_frames, paused, speed)
                    cv2.imshow("AGV Replay", annotated)
            elif key == 81 or key == ord('a'):       # LEFT or A — step back
                paused    = True
                frame_idx = seek(max(0, frame_idx - 1))
                ret, frame = self.cap.read()
                if ret:
                    data      = self.frames_data[frame_idx] if frame_idx < len(self.frames_data) else {}
                    annotated = self._draw_overlay(frame, data, frame_idx, total_frames, paused, speed)
                    cv2.imshow("AGV Replay", annotated)
            elif key in (ord('+'), ord('=')):        # + — speed up
                speed = min(speed + 0.25, 4.0)
            elif key in (ord('-'), ord('_')):        # - — slow down
                speed = max(speed - 0.25, 0.25)

        self.cap.release()
        cv2.destroyAllWindows()


# ── Integration snippet for main.py ──────────────────────────────────────────
#
#   from replay import Recorder
#
#   recorder = Recorder(fps=20, resolution=(WARP_WIDTH, WARP_HEIGHT))
#   recorder.start()
#
#   while True:
#       ...all your existing code...
#
#       recorder.record_frame(
#           frame          = warped,
#           agv_pos        = agv_pos,
#           agv_angle      = agv_angle,
#           target_angle   = target_angle,
#           target_distance= target_distance,
#           target_cell    = target_cell,
#           turn_amount    = turn_amount,
#           action         = action,
#           action_value   = action_value,
#           path           = path,
#           lines          = lines,
#           marker_centers = cf.marker_centers,
#           at_goal        = pa.is_at_goal(agv_pos, grid),
#       )
#
#   recorder.stop()
#
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # List available sessions if no argument given
        sessions_root = "sessions"
        if not os.path.exists(sessions_root):
            print("No sessions folder found. Run main.py first to record a session.")
            sys.exit(1)

        sessions = sorted([
            d for d in os.listdir(sessions_root)
            if os.path.isdir(os.path.join(sessions_root, d))
        ])

        if not sessions:
            print("No sessions found in ./sessions/")
            sys.exit(1)

        print("Available sessions:")
        for i, s in enumerate(sessions):
            print(f"  [{i}] {s}")

        choice = input("\nEnter session number to replay: ").strip()
        try:
            session_path = os.path.join(sessions_root, sessions[int(choice)])
        except (ValueError, IndexError):
            print("Invalid choice.")
            sys.exit(1)
    else:
        session_path = sys.argv[1]

    player = Player(session_path)
    player.load()
    player.play()