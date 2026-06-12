import cv2
import threading
import time


class DualViewProcessor:
    def __init__(self, webcam):

        self.webcam = webcam

        self.lock = threading.Lock()

        self.frame_hd = None
        self.frame_small = None

        self.running = True

        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):

        while self.running:

            frame = self.webcam.get_frame()

            if frame is None:
                time.sleep(0.005)
                continue

            hd = frame

            small = cv2.resize(
                frame,
                (568, 320),
                interpolation=cv2.INTER_LINEAR
            )

            with self.lock:
                self.frame_hd = hd
                self.frame_small = small

    def get_hd(self):
        with self.lock:
            if self.frame_hd is None:
                return None
            return self.frame_hd

    def get_small(self):
        with self.lock:
            if self.frame_small is None:
                return None
            return self.frame_small