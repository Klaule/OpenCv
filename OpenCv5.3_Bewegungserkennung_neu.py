from picamera2 import Picamera2
import cv2
import numpy as np
from collections import deque
from datetime import datetime
import tflite_runtime.interpreter as tflite

# === Globale Parameter ===
FRAMES_PER_SECOND = 10
PRE_RECORD_SECONDS = 7  # Voraufnehmedauer in Sekunden
POST_RECORD_SECONDS = 5  # Nachaufnahmedauer in Sekunden
BUFFER_SIZE = PRE_RECORD_SECONDS * FRAMES_PER_SECOND
VIDEO_PATH = '/home/klaule/Videos/'

# Puffer zur Speicherung von Frames vor der Bewegung
buffer = deque(maxlen=BUFFER_SIZE)
recording = False
movement_detected = False
frame_count = 0

# === TensorFlow Lite Modell laden ===
model_path = "/home/pi/tensorflow_models/detect.tflite"
label_path = "/home/pi/tensorflow_models/labelmap.txt"

interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Labels laden
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Funktion zur Katzenerkennung
def detect_cat(frame):
    # Frame vorbereiten
    input_shape = input_details[0]['shape'][1:3]
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[0]))
    input_data = np.expand_dims(resized_frame, axis=0).astype(np.uint8)

    # Modell aufrufen
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Ergebnisse verarbeiten
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    for i in range(len(scores)):
        if scores[i] > 0.5:  # Erkennungswahrscheinlichkeit > 50%
            label = labels[int(classes[i])]
            if "cat" in label:
                return True
    return False

# === Kameraeinstellungen ===
picam2 = Picamera2()
picam2.set_controls({
    "Sharpness": 1.0,
    "Contrast": 1.0,
    "ExposureTime": 20000,  # Belichtungszeit in Mikrosekunden
    "AnalogueGain": 4.0,
    "AwbEnable": True,
    "AeEnable": False,
})

camera_config = picam2.create_preview_configuration(main={"size": (320, 240)})
picam2.configure(camera_config)
picam2.start()

# === Bewegungserkennungseinstellungen ===
background_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=700, varThreshold=150, detectShadows=False)
MIN_CONTOUR_AREA = 5000  # Mindestfläche für Konturen
MOVEMENT_THRESHOLD = 12000  # Schwellenwert für Bewegungspixel

# === Hauptschleife ===
while True:
    try:
        frame = picam2.capture_array()
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Bewegungserkennung
        fg_mask = background_subtractor.apply(gray_frame)
        fg_mask = cv2.GaussianBlur(fg_mask, (31, 31), 0)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

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

        # Frame speichern
        buffer.append(gray_frame.copy())

        # Videoaufnahme starten
        if movement_detected and not recording:
            print("Bewegung erkannt! Prüfe auf Katze...")
            if detect_cat(frame):  # Prüfe, ob eine Katze erkannt wird
                print("Katze erkannt! Starte Videoaufnahme...")
                recording = True
                frame_count = 0
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                filename = f'{VIDEO_PATH}video_{timestamp}.mp4'
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                frame_height, frame_width = gray_frame.shape[:2]
                out = cv2.VideoWriter(filename, fourcc, FRAMES_PER_SECOND,
                                      (frame_width, frame_height), isColor=False)

                for buffered_frame in buffer:
                    out.write(buffered_frame)
                    frame_count += 1
            else:
                print("Keine Katze erkannt. Video wird nicht gespeichert.")

        if recording:
            frame = picam2.capture_array()
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            out.write(gray_frame)
            frame_count += 1

            if not movement_detected:
                post_recording_frames = 0
                while post_recording_frames < (POST_RECORD_SECONDS * FRAMES_PER_SECOND):
                    frame = picam2.capture_array()
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    out.write(gray_frame)
                    frame_count += 1
                    post_recording_frames += 1
                print(f"Video gespeichert unter: {filename}, Gesamtzahl der Frames: {frame_count}")
                recording = False
                out.release()

        display_frame_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        cv2.imshow("Bewegungserkennung", display_frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Fehler: {e}")
        picam2.stop()
        picam2.start()

# Ressourcen freigeben
picam2.stop()
cv2.destroyAllWindows()
