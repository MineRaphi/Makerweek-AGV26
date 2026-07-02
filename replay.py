"""
replay.py — AGV Session Recorder and Replay Player

RECORDING:
    Import and use the Recorder class in main.py to capture frames + data.

    from replay import Recorder
    recorder = Recorder()
    recorder.start()

    # Inside main loop, after all processing:
    recorder.record_frame(
        frames={
            "warped": warped,   # annotated top-down view
            "raw":    frame,    # original camera frame
        },
        agv_pos=agv_pos,
        ...
    )

    # On exit:
    recorder.stop()

PLAYBACK:
    Run this file directly to replay a saved session:

        python replay.py sessions/session_20260701_142305/

    Controls during playback:
        SPACE       — pause / resume
        LEFT / A    — step back one frame
        RIGHT / D   — step forward one frame
        TAB         — switch between recorded video views
        + / =       — speed up
        - / _       — slow down
        Q / ESC     — quit
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
    Records multiple video streams alongside frame data into a session folder.

    Each session contains:
        - video_<name>.mp4  — one video file per named frame passed to record_frame()
        - data.csv          — one row of data per frame, synced by frame_index
        - meta.json         — session metadata (fps, resolutions, stream names, start time)
    """

    def __init__(self, session_dir="sessions", fps=20):
        """
        session_dir: root folder where session subfolders are created
        fps:         frames per second for all output videos
        """
        self.fps          = fps
        self.session_path = None
        self.session_dir  = session_dir

        # One VideoWriter per named stream, populated on first record_frame() call
        self._writers     = {}
        # Resolution per stream, inferred from the first frame of each
        self._resolutions = {}

        self.csv_file     = None
        self.csv_writer   = None
        self.frame_index  = 0
        self.start_time   = None

    def start(self):
        """
        Creates a new session folder and opens the CSV file.
        Video writers are opened lazily on the first frame so their
        resolution can be inferred automatically.
        Call this once before the main loop starts.
        """
        timestamp         = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_path = os.path.join(self.session_dir, f"session_{timestamp}")
        os.makedirs(self.session_path, exist_ok=True)

        # CSV log
        csv_path        = os.path.join(self.session_path, "data.csv")
        self.csv_file   = open(csv_path, "w", newline="")
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

        # Metadata — video stream names are added later when writers are created
        self._meta_path = os.path.join(self.session_path, "meta.json")
        self._meta = {
            "start_time": timestamp,
            "fps":        self.fps,
            "streams":    [],
        }
        self._save_meta()

        self.start_time  = time.time()
        self.frame_index = 0
        print(f"Recording to: {self.session_path}")

    def _save_meta(self):
        with open(self._meta_path, "w") as f:
            json.dump(self._meta, f, indent=2)

    def _get_writer(self, name, frame):
        """
        Returns the VideoWriter for the given stream name, creating it if
        this is the first frame for that stream.
        """
        if name not in self._writers:
            h, w    = frame.shape[:2]
            resolution = (w, h)
            path    = os.path.join(self.session_path, f"video_{name}.mp4")
            fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
            self._writers[name]     = cv2.VideoWriter(path, fourcc, self.fps, resolution)
            self._resolutions[name] = resolution

            # Register this stream in metadata
            self._meta["streams"].append({"name": name, "resolution": resolution})
            self._save_meta()

        return self._writers[name]

    def record_frame(
        self,
        frames,                  # dict of {stream_name: image}, e.g. {"warped": warped, "raw": frame}
        agv_pos         = None,
        agv_angle       = None,
        target_angle    = None,
        target_distance = None,
        target_cell     = None,
        turn_amount     = None,
        action          = "none",
        action_value    = None,
        path            = None,
        lines           = None,
        marker_centers  = None,
        at_goal         = False,
    ):
        """
        Records one frame across all video streams, plus one CSV data row.
        Call this once per main loop iteration after all processing.

        frames: dict mapping stream name → annotated image (numpy BGR array)
                e.g. {"warped": warped_img, "raw": raw_frame, "mask": blue_mask}
        """
        if self.csv_file is None:
            raise RuntimeError("Recorder not started — call recorder.start() first.")

        # Write each named video stream
        for name, img in frames.items():
            writer = self._get_writer(name, img)

            # Convert grayscale to BGR if needed (e.g. for the blue mask)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # Resize if the frame doesn't match this stream's registered resolution
            expected_res = self._resolutions[name]
            h, w = img.shape[:2]
            if (w, h) != expected_res:
                img = cv2.resize(img, expected_res)

            writer.write(img)

        # Write CSV row
        self.csv_writer.writerow([
            self.frame_index,
            datetime.now().isoformat(),
            agv_pos[0]                if agv_pos          else "",
            agv_pos[1]                if agv_pos          else "",
            round(agv_angle, 2)       if agv_angle        is not None else "",
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
            at_goal,
        ])

        self.csv_file.flush()
        self.frame_index += 1

    def stop(self):
        """Finalises and closes all files. Call this when the main loop exits."""
        for writer in self._writers.values():
            writer.release()
        if self.csv_file:
            self.csv_file.close()

        duration = time.time() - self.start_time if self.start_time else 0
        streams  = list(self._writers.keys())
        print(f"Recording stopped. {self.frame_index} frames across {len(streams)} streams "
              f"({', '.join(streams)}) in {duration:.1f}s → {self.session_path}")


# ── Player ────────────────────────────────────────────────────────────────────

class Player:
    """
    Replays a recorded session. Multiple video streams are loaded; press TAB
    to switch between them. Each frame shows the video alongside a data overlay.

    Controls:
        SPACE       pause / resume
        LEFT / A    step back one frame
        RIGHT / D   step forward one frame
        TAB         cycle through the recorded video streams
        + / =       increase playback speed
        - / _       decrease playback speed
        Q / ESC     quit
    """

    def __init__(self, session_path):
        self.session_path = session_path
        self.frames_data  = []
        self.meta         = {}
        self.caps         = {}   # {stream_name: VideoCapture}
        self.stream_names = []   # ordered list of stream names
        self.active_stream = 0  # index into stream_names for the currently shown video

    def load(self):
        """Loads metadata, CSV data, and opens all video files found in the session."""
        csv_path  = os.path.join(self.session_path, "data.csv")
        meta_path = os.path.join(self.session_path, "meta.json")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"No data CSV found at {csv_path}")

        # Load metadata
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.meta = json.load(f)

        # Load all CSV rows
        with open(csv_path, newline="") as f:
            self.frames_data = list(csv.DictReader(f))

        # Find and open all video files in the session folder
        video_files = sorted([
            f for f in os.listdir(self.session_path)
            if f.startswith("video_") and f.endswith(".mp4")
        ])

        if not video_files:
            raise FileNotFoundError(f"No video files found in {self.session_path}")

        for vf in video_files:
            # Strip "video_" prefix and ".mp4" suffix to get the stream name
            name = vf[len("video_"):-len(".mp4")]
            cap  = cv2.VideoCapture(os.path.join(self.session_path, vf))
            self.caps[name]   = cap
            self.stream_names.append(name)

        total_frames = int(list(self.caps.values())[0].get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Loaded session: {self.session_path}")
        print(f"  Streams : {', '.join(self.stream_names)}")
        print(f"  Frames  : {total_frames}  |  Data rows: {len(self.frames_data)}")

    def _read_frame(self, name):
        """Reads the next frame from the named stream."""
        ret, frame = self.caps[name].read()
        return frame if ret else None

    def _seek_all(self, idx):
        """Seeks all video streams to the same frame index."""
        idx = max(0, min(idx, self._total_frames() - 1))
        for cap in self.caps.values():
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        return idx

    def _total_frames(self):
        return int(list(self.caps.values())[0].get(cv2.CAP_PROP_FRAME_COUNT))

    def _draw_overlay(self, frame, data, frame_idx, total_frames, paused, speed, stream_name):
        """Draws the data panel on the left and stream name label at the top."""
        panel_w = 320
        overlay = frame.copy()

        cv2.rectangle(overlay, (0, 0), (panel_w, frame.shape[0]), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        def put(text, row, color=(220, 220, 220)):
            cv2.putText(frame, text, (10, 24 + row * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

        # Header
        put(f"FRAME  {frame_idx + 1} / {total_frames}", 0, (100, 220, 100))
        put(f"VIEW   {stream_name}  (TAB to switch)", 1, (180, 140, 50))
        put(f"{'PAUSED' if paused else f'SPEED {speed:.1f}x'}", 2,
            (80, 80, 255) if paused else (220, 180, 50))
        put(f"TIME   {data.get('timestamp', '')[:19]}", 3, (160, 160, 160))

        put("--- AGV ---", 5, (100, 180, 255))
        put(f"Grid pos  row={data.get('agv_grid_row','?')}  col={data.get('agv_grid_col','?')}", 6)
        put(f"Angle     {data.get('agv_angle','?')}", 7)
        put(f"At goal   {data.get('at_goal','?')}", 8,
            (50, 255, 50) if data.get("at_goal") == "True" else (220, 220, 220))

        put("--- PATH ---", 10, (100, 180, 255))
        put(f"Length    {data.get('path_length','?')} cells", 11)
        put(f"Target    row={data.get('target_cell_row','?')}  col={data.get('target_cell_col','?')}", 12)
        put(f"T.angle   {data.get('target_angle','?')}", 13)
        put(f"T.dist    {data.get('target_distance','?')} px", 14)

        put("--- STEERING ---", 16, (100, 180, 255))
        put(f"Turn amt  {data.get('turn_amount','?')}", 17)
        action = data.get("action", "none")
        color  = (50, 200, 50) if action == "move" else (50, 100, 255) if action == "rotate" else (160,160,160)
        put(f"Action    {action}  ({data.get('action_value','?')})", 18, color)

        put("--- FIELD ---", 20, (100, 180, 255))
        put(f"Blue lines  {data.get('blue_lines_found','?')}", 21)
        put(f"Markers     {data.get('marker_ids_visible','?')}", 22)

        put("--- CONTROLS ---", 24, (130, 130, 130))
        put("SPACE pause  TAB stream",   25, (130, 130, 130))
        put("A/D step  +/- speed  Q quit", 26, (130, 130, 130))

        return frame

    def play(self):
        """Starts the interactive replay window."""
        total  = self._total_frames()
        fps    = self.meta.get("fps", 20)
        idx    = 0
        paused = False
        speed  = 1.0

        while True:
            active_name = self.stream_names[self.active_stream]

            if not paused:
                # Read the active stream's next frame
                frame = self._read_frame(active_name)

                # Advance (and discard) all other streams to stay in sync
                for name in self.stream_names:
                    if name != active_name:
                        self.caps[name].read()

                if frame is None:
                    # Loop back to the beginning
                    idx = self._seek_all(0)
                    frame = self._read_frame(self.stream_names[self.active_stream])
                    for name in self.stream_names:
                        if name != self.stream_names[self.active_stream]:
                            self.caps[name].read()
                    if frame is None:
                        break  # truly unreadable, give up

                idx  = int(self.caps[active_name].get(cv2.CAP_PROP_POS_FRAMES)) - 1
                data = self.frames_data[idx] if idx < len(self.frames_data) else {}

                annotated = self._draw_overlay(frame, data, idx, total, paused, speed, active_name)
                cv2.imshow("AGV Replay", annotated)

                delay = max(1, int((1000 / fps) / speed))
            else:
                delay = 30

            key = cv2.waitKey(delay) & 0xFF

            if key == ord('q') or key == 27:           # Q / ESC — quit
                break
            elif key == ord(' '):                       # SPACE — pause/resume
                paused = not paused
            elif key == 9:                              # TAB — switch stream
                self.active_stream = (self.active_stream + 1) % len(self.stream_names)
                # Re-show the current frame in the new stream without advancing
                self._seek_all(idx)
                frame = self._read_frame(self.stream_names[self.active_stream])
                if frame is not None:
                    data      = self.frames_data[idx] if idx < len(self.frames_data) else {}
                    annotated = self._draw_overlay(frame, data, idx, total, paused, speed,
                                                   self.stream_names[self.active_stream])
                    cv2.imshow("AGV Replay", annotated)
            elif key == 83 or key == ord('d'):          # RIGHT / D — step forward
                paused = True
                idx    = self._seek_all(idx + 1)
                frame  = self._read_frame(self.stream_names[self.active_stream])
                if frame is not None:
                    data      = self.frames_data[idx] if idx < len(self.frames_data) else {}
                    annotated = self._draw_overlay(frame, data, idx, total, paused, speed, active_name)
                    cv2.imshow("AGV Replay", annotated)
            elif key == 81 or key == ord('a'):          # LEFT / A — step back
                paused = True
                idx    = self._seek_all(max(0, idx - 1))
                frame  = self._read_frame(self.stream_names[self.active_stream])
                if frame is not None:
                    data      = self.frames_data[idx] if idx < len(self.frames_data) else {}
                    annotated = self._draw_overlay(frame, data, idx, total, paused, speed, active_name)
                    cv2.imshow("AGV Replay", annotated)
            elif key in (ord('+'), ord('=')):           # + — speed up
                speed = min(speed + 0.25, 4.0)
            elif key in (ord('-'), ord('_')):           # - — slow down
                speed = max(speed - 0.25, 0.25)

        for cap in self.caps.values():
            cap.release()
        cv2.destroyAllWindows()


# ── Integration snippet for main.py ──────────────────────────────────────────
#
#   from replay import Recorder
#
#   recorder = Recorder(fps=20)
#   recorder.start()
#
#   while True:
#       ...all your existing code...
#
#       recorder.record_frame(
#           frames = {
#               "warped": warped,   # annotated top-down view
#               "raw":    frame,    # original camera frame
#               "mask":   mask,     # blue line mask (grayscale OK)
#           },
#           agv_pos         = agv_pos,
#           agv_angle       = agv_angle,
#           target_angle    = target_angle,
#           target_distance = target_distance,
#           target_cell     = target_cell,
#           turn_amount     = turn_amount,
#           action          = action,
#           action_value    = action_value,
#           path            = path,
#           lines           = lines,
#           marker_centers  = cf.marker_centers,
#           at_goal         = pa.is_at_goal(agv_pos, grid),
#       )
#
#   recorder.stop()
#
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
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