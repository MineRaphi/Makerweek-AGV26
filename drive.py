import requests
import numpy as np

AGV_BASE_URL = "http://172.17.1.58"

def send_velocity(vel_left_perc, vel_right_perc):
    """Sendet Soll-Geschwindigkeiten an den AGV über die REST-API."""
    vel_left_perc = float(np.clip(vel_left_perc, -100.0, 100.0))
    vel_right_perc = float(np.clip(vel_right_perc, -100.0, 100.0))

    try:
        resp = requests.post(
            f"{AGV_BASE_URL}/api/agv/stepper/setVelocity",
            json={"velLeft_perc": vel_left_perc, "velRight_perc": vel_right_perc},
            timeout=0.5,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"AGV-Request fehlgeschlagen: {e}")


def follow_path(path, agv_angle, agv_center, grid_to_pixel,
                 base_speed=40.0, kp=0.8, waypoint_radius_px=30):
    """
    Berechnet aus path + aktueller AGV-Pose die Motor-Geschwindigkeiten
    und sendet sie an den AGV.

    path: Liste von (row, col) aus astar()
    agv_angle: aktueller Winkel (Grad) aus get_marker_direction()
    agv_center: aktuelle Pixelposition (x, y) des AGV-Markers
    grid_to_pixel: Funktion (row, col) -> (x, y) im warped-Bild
    """
    if not path or len(path) < 2:
        send_velocity(0, 0)
        return

    # Nächsten Wegpunkt suchen, der noch ein Stück voraus liegt
    target = None
    for r, c in path:
        tx, ty = grid_to_pixel(r, c)
        dist = np.hypot(tx - agv_center[0], ty - agv_center[1])
        if dist > waypoint_radius_px:
            target = (tx, ty)
            break

    if target is None:
        # Pfad fertig abgefahren
        send_velocity(0, 0)
        return

    # Soll-Winkel zum Zielpunkt berechnen (gleiche Konvention wie get_marker_direction)
    dx = target[0] - agv_center[0]
    dy = target[1] - agv_center[1]
    target_angle = np.degrees(np.arctan2(-dy, dx))

    # Winkel-Differenz normalisieren auf [-180, 180]
    angle_error = (target_angle - agv_angle + 180) % 360 - 180

    # P-Regler: Lenkdifferenz proportional zum Winkelfehler
    turn = kp * angle_error

    vel_left = base_speed - turn
    vel_right = base_speed + turn

    send_velocity(vel_left, vel_right)