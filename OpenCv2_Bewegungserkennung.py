from picamera2 import Picamera2
import cv2
import numpy as np
import time

# Initialisiere Picamera2
picam2 = Picamera2()

# Konfiguration der Kamera mit einer bestimmten Auflösung
camera_config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(camera_config)

# Kamera starten
picam2.start()

# Lese Frames von der Kamera und zeige sie mit OpenCV an
while True:
    # Hole den Frame als NumPy-Array
    frame = picam2.capture_array()

    # Zeige den Frame mit OpenCV an
    cv2.imshow("Kamera Frame", frame)

    # Breche die Schleife ab, wenn 'q' gedrückt wird
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera stoppen und OpenCV-Fenster schließen
picam2.stop()
cv2.destroyAllWindows()
