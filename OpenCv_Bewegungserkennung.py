import cv2
import numpy as np
import time
from collections import deque
from datetime import datetime

# Videoaufnahme-Parameter
VIDEO_PATH = '/home/klaule/Videos/'  # Pfad zum Speichern der Videos
BUFFER_SIZE = 150  # Anzahl der Frames im Puffer (z.B. 30 fps -> 5 Sekunden Puffer)
FRAMES_PER_SECOND = 30  # Bildrate des Videos
PRE_RECORD_SECONDS = 5  # Anzahl der Sekunden vor der Bewegung
POST_RECORD_SECONDS = 5  # Anzahl der Sekunden nach der Bewegung

# Puffer zur Speicherung von Frames vor der Bewegung
buffer = deque(maxlen=BUFFER_SIZE)  # 150 Frames bei 30 fps -> 5 Sekunden
recording = False  # Status der Videoaufnahme
movement_detected = False  # Bewegungserkennung

# Initialisierung der Kamera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Setze die Auflösung (Breite)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Setze die Auflösung (Höhe)

# Überprüfen, ob die Kamera geöffnet werden konnte
if not cap.isOpened():
    print("Fehler: Kamera konnte nicht geöffnet werden.")
    exit()

# Vorherigen Frame (Graustufen) für Optical Flow speichern
ret, prev_frame = cap.read()
if not ret:
    print("Fehler: Konnte den ersten Frame nicht lesen.")
    exit()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Video-Writer für die Videoaufnahme initialisieren
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID-Codec für die Videoaufnahme

# Hauptschleife zur Frameverarbeitung
while True:
    ret, frame = cap.read()  # Lese den aktuellen Frame
    if not ret:
        print("Fehler: Konnte den Frame nicht lesen.")
        break

    # Konvertiere den aktuellen Frame in Graustufen
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Optical Flow berechnen (Farneback-Algorithmus)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Berechnung der Bewegung (Magnitude und Winkel)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    movement = np.sum(magnitude)  # Gesamte Bewegungsmenge berechnen

    # Bewegung erkennen, wenn die Summe der Magnitude einen Schwellwert überschreitet
    MOVEMENT_THRESHOLD = 1000  # Schwellenwert für Bewegungserkennung
    if movement > MOVEMENT_THRESHOLD:
        movement_detected = True
        print(f"Bewegung erkannt mit Wert: {movement}")
    else:
        movement_detected = False

    # Puffer mit aktuellen Frame füllen
    buffer.append(frame)

    # Starte die Aufnahme, wenn Bewegung erkannt wird und noch nicht aufgenommen wird
    if movement_detected and not recording:
        print("Starte Videoaufnahme...")
        recording = True

        # Erzeuge Dateinamen basierend auf Zeitstempel
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'{VIDEO_PATH}video_{timestamp}.avi'
        out = cv2.VideoWriter(filename, fourcc, FRAMES_PER_SECOND, (frame.shape[1], frame.shape[0]))

        # Schreibe gepufferte Frames in die Datei (Vor-Aufnahme)
        for buffered_frame in buffer:
            out.write(buffered_frame)

    # Schreibe Frames, wenn Aufnahme läuft
    if recording:
        out.write(frame)  # Schreibe den aktuellen Frame
        post_recording_frames = 0  # Zähle die Frames nach der Bewegung

        # Warte 5 Sekunden nach der Bewegung, bevor die Aufnahme beendet wird
        while post_recording_frames < (POST_RECORD_SECONDS * FRAMES_PER_SECOND):
            ret, post_frame = cap.read()
            if not ret:
                break
            out.write(post_frame)
            post_recording_frames += 1

        # Stoppe die Aufnahme
        recording = False
        print(f"Video gespeichert unter: {filename}")
        out.release()  # Beende die Videoaufnahme

    # Aktualisiere den vorherigen Frame
    prev_gray = gray_frame

    # Zeige das aktuelle Bild mit der Bewegungserkennung
    cv2.putText(frame, f"Bewegung: {'Erkannt' if movement_detected else 'Nicht erkannt'}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Bewegungserkennung", frame)

    # Drücke 'q', um das Programm zu beenden
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Ressourcen freigeben und Fenster schließen
cap.release()
cv2.destroyAllWindows()
