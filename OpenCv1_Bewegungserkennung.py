from libcamera import CameraManager, FrameBufferAllocator
import cv2
import numpy as np

# Initialisiere die Kamera
camera_manager = CameraManager()
cameras = camera_manager.cameras

if len(cameras) == 0:
    print("Keine Kameras gefunden.")
    exit()

camera = cameras[0]

# Konfiguration der Kamera
camera_config = camera.generate_configuration({'format': 'RGB888', 'size': (640, 480)})
camera.configure(camera_config)

# FrameBufferAllocator initialisieren
allocator = FrameBufferAllocator(camera)
for stream in camera.streams:
    allocator.allocate(stream)

# Kamera starten
camera.start()

# Lese die Frames und zeige sie mit OpenCV an
for request in camera.capture_request():
    frame = request.buffers[0]  # Erhalte den Framebuffer

    # Konvertiere den Frame in ein NumPy-Array
    frame_array = np.frombuffer(frame, dtype=np.uint8).reshape((480, 640, 3))

    # Zeige das Bild an
    cv2.imshow("Kamera", frame_array)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera stoppen und Ressourcen freigeben
camera.stop()
cv2.destroyAllWindows()
