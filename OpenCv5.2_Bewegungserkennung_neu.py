from picamera2 import Picamera2
import cv2
import numpy as np
from collections import deque
from datetime import datetime

# === Globale Parameter ===
FRAMES_PER_SECOND = 30
PRE_RECORD_SECONDS = 7  # Voraufnehmedauer in Sekunden
POST_RECORD_SECONDS = 5  # Nachaufnahmedauer in Sekunden
BUFFER_SIZE = PRE_RECORD_SECONDS * FRAMES_PER_SECOND
VIDEO_PATH = '/home/klaule/Videos/'

# Puffer zur Speicherung von Frames vor der Bewegung
buffer = deque(maxlen=BUFFER_SIZE)
recording = False
movement_detected = False
frame_count = 0

# === Kameraeinstellungen ===
picam2 = Picamera2()
picam2.set_controls({
    "Sharpness": 1.0,
    "Contrast": 1.0,
    "ExposureTime": 20000,   # Belichtungszeit in Mikrosekunden
    "AnalogueGain": 4.0,
    "AwbEnable": True,
    "AeEnable": False,
    
})

camera_config = picam2.create_preview_configuration(main={"size": (640, 480)})  # Auflösung einstellen
picam2.configure(camera_config)
picam2.start()

# === Bewegungserkennungseinstellungen ===
background_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=700, varThreshold=100, detectShadows=False)
MIN_CONTOUR_AREA = 2000  # Mindestfläche für Konturen
MOVEMENT_THRESHOLD = 8000  # Schwellenwert für die Anzahl veränderter Pixel

# === Hauptschleife zur Bildaufnahme und Verarbeitung ===
while True:
    # === Bildaufnahme ===
    frame = picam2.capture_array()
    
    # Frame um 180° drehen
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    # Überprüfen und konvertieren des Frames, wenn nötig
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # === Bewegungserkennung ===
    fg_mask = background_subtractor.apply(gray_frame)  # Hintergrundsubtraktion anwenden
    fg_mask = cv2.GaussianBlur(fg_mask, (31, 31), 0)  # Weichzeichnen

    # Morphologische Operationen anwenden
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Anzahl der veränderten Pixel zählen
    movement_pixels = cv2.countNonZero(fg_mask)

    movement_detected = False
    if movement_pixels > MOVEMENT_THRESHOLD:
        contours, _ = cv2.findContours(fg_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                continue
            movement_detected = True
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(gray_frame, (x, y), (x + w, y + h), (255), 2)
            cv2.putText(gray_frame, "Bewegung erkannt!", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255), 2)
            break

    # === Frame speichern ===
    buffer.append(gray_frame.copy())  # Frames in den Puffer hinzufügen

    # === Videoaufnahme starten ===
    if movement_detected and not recording:
        print("Bewegung erkannt! Starte Videoaufnahme...")
        recording = True
        frame_count = 0
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Video-Datei vorbereiten
        filename = f'{VIDEO_PATH}video_{timestamp}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_height, frame_width = gray_frame.shape[:2]
        out = cv2.VideoWriter(filename, fourcc, FRAMES_PER_SECOND,
                              (frame_width, frame_height), isColor=False)

        # Gepufferte Frames schreiben (alle Frames im Puffer sind bereits gedreht)
        for buffered_frame in buffer:
            out.write(buffered_frame)
            frame_count += 1

    # === Aufnahme fortsetzen ===
    if recording:
        frame = picam2.capture_array()  # Neues Frame erfassen
        frame = cv2.rotate(frame, cv2.ROTATE_180)  # Frame um 180° drehen
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        out.write(gray_frame)
        frame_count += 1

        # Nach-Aufnahme, wenn keine Bewegung mehr erkannt wird
        if not movement_detected:
            post_recording_frames = 0
            while post_recording_frames < (POST_RECORD_SECONDS * FRAMES_PER_SECOND):
                frame = picam2.capture_array()
                frame = cv2.rotate(frame, cv2.ROTATE_180)  # Frame um 180° drehen
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                out.write(gray_frame)
                frame_count += 1
                post_recording_frames += 1
            print(f"Video gespeichert unter: {filename}, Gesamtzahl der geschriebenen Frames: {frame_count}")
            recording = False
            out.release()

    # === Anzeige des aktuellen Frames ===
    display_frame_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    cv2.imshow("Bewegungserkennung", display_frame_bgr)

    # Beenden mit 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Ressourcen freigeben ===
picam2.stop()
cv2.destroyAllWindows()
