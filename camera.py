import cv2
import numpy as np

# 1. Kamera starten (0 für Webcam, oder deine Stream-URL als String)
kamera = cv2.VideoCapture("http://10.250.150.224:8081/")

# FÜR DEN ZOOM-FIX: Kamera auf Full-HD-Auflösung zwingen,
# damit sie den vollen Weitwinkel-Sensor nutzt und herauszoomt.
kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# 2. ArUco-Detektor für die Ecken-Erkennung vorbereiten
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# 3. Gewünschte Größe des fertigen, flachen Bildes definieren
BREITE, HOEHE = 800, 600

# FÜR DEN ZOOM-FIX IM ERGEBNISFENSTER: 
# Wir lassen einen Puffer (z.B. 80 Pixel) Platz zu den Rändern.
# Dadurch kleben die Marker nicht direkt am Fensterrand und es wirkt weniger reingezoomt.
PUFFER = 80

ziel_punkte = np.float32([
    [PUFFER, PUFFER],                  # Oben-Links rückt nach innen
    [BREITE - PUFFER, PUFFER],         # Oben-Rechts rückt nach innen
    [BREITE - PUFFER, HOEHE - PUFFER], # Unten-Rechts rückt nach innen
    [PUFFER, HOEHE - PUFFER]           # Unten-Links rückt nach innen
])

print("Programm gestartet. Drücke 'q' in einem der Fenster, um es zu beenden.")

while True:
    ret, frame = kamera.read()
    if not ret:
        print("Fehler: Kein Kamerabild empfangen.")
        break

    # 4. Marker im aktuellen Weitwinkel-Kamerabild suchen
    ecken, ids, _ = detector.detectMarkers(frame)

    # Wir transformieren nur, wenn mindestens 4 Marker im Bild zu sehen sind
    if ids is not None and len(ids) >= 4:
        ids = ids.flatten()
        
        try:
            # Holen der Pixel-Koordinaten für die IDs 0, 1, 2 und 3
            ol = ecken[np.where(ids == 0)[0][0]][0][0] # Oben-Links (ID 0)
            or_ = ecken[np.where(ids == 1)[0][0]][0][0] # Oben-Rechts (ID 1)
            ur = ecken[np.where(ids == 2)[0][0]][0][0] # Unten-Rechts (ID 2)
            ul = ecken[np.where(ids == 3)[0][0]][0][0] # Unten-Links (ID 3)

            # Die 4 schrägen Punkte sammeln
            quell_punkte = np.float32([ol, or_, ur, ul])

            # 5. Mathematische Transformation berechnen und anwenden
            matrix = cv2.getPerspectiveTransform(quell_punkte, ziel_punkte)
            vogelperspektive = cv2.warpPerspective(frame, matrix, (BREITE, HOEHE))

            # Das entzerrte, flache Feld anzeigen
            cv2.imshow("Das geradegezogene Feld (Draufsicht)", vogelperspektive)
            
        except IndexError:
            # Falls zwar 4 Marker da sind, aber nicht die benötigten IDs 0 bis 3
            pass

    # Zur Kontrolle: Gefundene Marker im Originalbild bunt umranden
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, ecken, ids)

    # Das originale (Weitwinkel-)Kamerabild anzeigen
    cv2.imshow("Originale Kamera-Ansicht", frame)

    # Abbrechen mit der Taste 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Nach dem Beenden aufräumen
kamera.release()
cv2.destroyAllWindows()
print("Programm sauber beendet.")