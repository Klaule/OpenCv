from picamera2 import Picamera2
import cv2
import numpy as np
from collections import deque
from datetime import datetime
import os
import shutil

# === Globale Parameter ===
FRAMES_PER_SECOND = 10
PRE_RECORD_SECONDS = 7
POST_RECORD_SECONDS = 5
BUFFER_SIZE = PRE_RECORD_SECONDS * FRAMES_PER_SECOND
TEMP_VIDEO_PATH = '/home/klaule/Videos/temp/'
CAT_VIDEO_PATH = '/home/klaule/Videos/cat/'
NO_CAT_VIDEO_PATH = '/home/klaule/Videos/no_cat/'

# Ordner erstellen, falls nicht vorhanden
os.makedirs(TEMP_VIDEO_PATH, exist_ok=True)
os.makedirs(CAT_VIDEO_PATH, exist_ok=True)
os.makedirs(NO_CAT_VIDEO_PATH, exist_ok=True)

buffer = deque(maxlen=BUFFER_SIZE)
recording = False
movement_detected = False
frame_count = 0

# === YOLOv5-Modell laden ===
model_path = "yolov5s.onnx"
net = cv2.dnn.readNetFromONNX(model_path)

# Funktion zur Katzenerkennung mit YOLOv5
def detect_cat(video_path):
    cap = cv2.VideoCapture(video_path)
    cat_detected = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO-DNN Blob erstellen
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward()

        # Ergebnisse verarbeiten
        for detection in outputs[0, 0, :, :]:
            confidence = detection[2]
            if confidence > 0.5:  # Schwellenwert für Erkennung
                class_id = int(detection[1])
                if class_id == 15:  # COCO-Label-ID für 'cat'
                    cat_detected = True
                    break

        if cat_detected:
            break

    cap.release()
    return cat_detected

# === Kameraeinstellungen ===
picam2 = Picamera2()
picam2.set_controls({
    "Sharpness": 1.0,
    "Contrast": 1.0,
    "ExposureTime": 20000,
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
MIN_CONTOUR_AREA = 5000
MOVEMENT_THRESHOLD = 12000

while True:
    try:
        frame = picam2.capture_array()
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

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
                break

        buffer.append(gray_frame.copy())

        if movement_detected and not recording:
            print("Bewegung erkannt! Starte Videoaufnahme...")
            recording = True
            frame_count = 0
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'{TEMP_VIDEO_PATH}video_{timestamp}.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_height, frame_width = gray_frame.shape[:2]
            out = cv2.VideoWriter(filename, fourcc, FRAMES_PER_SECOND, (frame_width, frame_height), isColor=False)

            for buffered_frame in buffer:
                out.write(buffered_frame)
                frame_count += 1

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
                out.release()
                recording = False
                print(f"Video gespeichert: {filename}")

                # Nachträgliche Analyse
                if detect_cat(filename):
                    shutil.move(filename, f'{CAT_VIDEO_PATH}{os.path.basename(filename)}')
                    print(f"Video enthält eine Katze und wurde verschoben.")
                else:
                    shutil.move(filename, f'{NO_CAT_VIDEO_PATH}{os.path.basename(filename)}')
                    print(f"Video enthält keine Katze und wurde verschoben.")

        display_frame_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        cv2.imshow("Bewegungserkennung", display_frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Fehler: {e}")
        picam2.stop()
        picam2.start()

picam2.stop()
cv2.destroyAllWindows()
