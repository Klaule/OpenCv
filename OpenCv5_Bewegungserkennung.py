from picamera2 import Picamera2
import cv2
import numpy as np
import time
from collections import deque
from datetime import datetime


# Globale Parameter
BUFFER_SIZE = 150  # Puffergröße (5 Sekunden bei 30 FPS)
FRAMES_PER_SECOND = 30  # Bildrate des Videos
PRE_RECORD_SECONDS = 5  # Sekunden, die vor der Bewegung aufgezeichnet werden
POST_RECORD_SECONDS = 5  # Sekunden, die nach der Bewegung aufgezeichnet werden
VIDEO_PATH = '/home/klaule/Videos/'  # Pfad zum Speichern der Videos

# Puffer zur Speicherung von Frames vor der Bewegung
buffer = deque(maxlen=BUFFER_SIZE)
recording = False  # Status der Videoaufnahme
movement_detected = False  # Bewegungserkennung

motion_threshold = 10 # Bewegungsschwelle (je niedriger, desto empfindlicher)

def detect_motion(frame1, frame2):
    # Berechne die Differenz zwischen zwei Frames
    diff = cv2.absdiff(frame1, frame2)
    # Wandelt das Bild in Graustufen um
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # Wende einen Schwellenwert an, um Bewegungen zu erkennen
    _, thresh = cv2.threshold(gray, motion_threshold, 255, cv2.THRESH_BINARY)
    return thresh

# Initialisiere Picamera2
picam2 = Picamera2()
# Setze den Weißabgleich
picam2.set_controls({"AwbMode": "auto"})  # Oder "tungsten", "fluorescent", je nach Lichtquelle

camera_config = picam2.create_preview_configuration(main={"size": (640, 480)})  # Kameraauflösung einstellen
picam2.configure(camera_config)

# Kamera starten
picam2.start()

# Hintergrundsubtraktor für die Bewegungserkennung (MOG2-Algorithmus)
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Mindestfläche für die Erkennung einer Bewegung (verhindert Rauschen)
MIN_CONTOUR_AREA = 500

# Video-Writer für die Videoaufzeichnung initialisieren (H.264-Codec)
fourcc = cv2.VideoWriter_fourcc(*'X264')  # Verwende 'X264' für H.264 Codec

# Hauptschleife zur Verarbeitung der Kamera-Frames
while True:
    # Hole den Frame von der Kamera als NumPy-Array
    frame = picam2.capture_array()

    # Konvertiere den aktuellen Frame in Graustufen (für eine einfachere Verarbeitung)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Hintergrundsubtraktion anwenden, um Bewegungen zu erkennen
    fg_mask = background_subtractor.apply(gray_frame)

    # Maske glätten (Rauschen reduzieren)
    fg_mask = cv2.GaussianBlur(fg_mask, (21, 21), 0)

    # Konturen im maskierten Bild finden
    contours, _ = cv2.findContours(fg_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Über alle gefundenen Konturen iterieren und prüfen, ob eine Bewegung vorliegt
    movement_detected = False
    for contour in contours:
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
            # Zu kleine Konturen (Rauschen) ignorieren
            continue

        # Bewegung erkannt
        movement_detected = True

        # Bounding-Box (rechteckiger Bereich) um die erkannte Bewegung zeichnen
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Text für erkannte Bewegung hinzufügen
        cv2.putText(frame, "Bewegung erkannt!", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Puffer mit aktuellen Frames füllen
    buffer.append(frame)

    # Starte die Aufnahme, wenn Bewegung erkannt wird und noch nicht aufgenommen wird
    if movement_detected and not recording:
        print("Starte Videoaufnahme...")
        recording = True
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'{VIDEO_PATH}video_{timestamp}.h264'  # Ändere die Dateiendung auf .h264
        out = cv2.VideoWriter(filename, fourcc, FRAMES_PER_SECOND, (frame.shape[1], frame.shape[0]))

        # Schreibe die gepufferten Frames in die Datei (Vor-Aufnahme)
        for buffered_frame in buffer:
            out.write(buffered_frame)

    # Wenn eine Aufnahme gestartet wurde, schreibe weiter in die Datei
    if recording:
        out.write(frame)

        # Wenn keine Bewegung erkannt wurde, zähle die Frames nach der letzten Bewegung
        if not movement_detected:
            post_recording_frames = 0  # Zähle die Frames nach der Bewegung

            # Warte 5 Sekunden nach der Bewegung, bevor die Aufnahme beendet wird
            while post_recording_frames < (POST_RECORD_SECONDS * FRAMES_PER_SECOND):
                frame = picam2.capture_array()
                out.write(frame)
                post_recording_frames += 1

            # Stoppe die Aufnahme
            print(f"Video gespeichert unter: {filename}")
            recording = False
            out.release()  # Videoaufnahme beenden

    # Zeige das aktuelle Bild mit der Bewegungserkennung an
    cv2.imshow("Bewegungserkennung", frame)

    # Drücke 'q', um das Programm zu beenden
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Ressourcen freigeben
picam2.stop()
cv2.destroyAllWindows()
