import cv2
import threading


class Webcam:
    def __init__(self, device="/dev/video1", width=1280, height=720, fps=60):

        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)

        # MJPEG Low-Latency Mode
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # (je nach Kamera Linux: 0.25 = manual)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)         # experimentell
        self.cap.set(cv2.CAP_PROP_GAIN, 0)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
        self.cap.set(cv2.CAP_PROP_CONTRAST, 0)

        # Minimize buffer size (tradeoff latency vs frame drops)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print("camera opened:", self.cap.isOpened())

        self._frame = None
        self._lock = threading.Lock()

        self._running = True

        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._running:

            # Drop old frames (low latency)
            self.cap.grab()

            ret, frame = self.cap.retrieve()

            if not ret:
                continue

            # with self._lock:
            # No copy mechanism (low latency)
            self._frame = frame

    def get_frame(self):
        with self._lock:
            if self._frame is None:
                return None
            return self._frame

    def stop(self):
        self._running = False
        self.cap.release()