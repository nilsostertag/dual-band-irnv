import cv2
import threading
import time
from pose_estimation.onnx_detector import ONNXDetector

class DualViewProcessor:
    def __init__(self, webcam):

        self.webcam = webcam

        self.lock = threading.Lock()

        self.frame_hd = None
        self.frame_small = None

        self.detector = ONNXDetector("src/models/yolov8n.onnx")
        self.overlay = None

        self.running = True

        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:

            frame = self.webcam.get_frame()
            if frame is None:
                continue
            hd = frame

            small = cv2.resize(frame, (568, 320))

            boxes = self.detector.infer(small)

            if not boxes:
                boxes = []

            self.latest_boxes = boxes

            small = small = self.draw_boxes(small, boxes, 1.0, 1.0)

            sx = hd.shape[1] / 568
            sy = hd.shape[0] / 320

            hd = self.draw_boxes(hd, boxes, sx, sy)

            with self.lock:
                self.frame_hd = hd
                self.frame_small = small

    def get_hd(self):
        with self.lock:
            if self.frame_hd is None:
                return None
            return self.frame_hd

    def get_small_frame(self):
        if self.small_frame is None:
            return None
        return self.small_frame

    def attach_detector(self, detector, overlay):
        self.detector = detector
        self.overlay = overlay

    def draw_boxes(self, frame, boxes, sx=1.0, sy=1.0):

        if boxes is None:
            return frame

        for x1, y1, x2, y2, conf in boxes:
            x1 = int(x1 * sx)
            y1 = int(y1 * sy)
            x2 = int(x2 * sx)
            y2 = int(y2 * sy)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return frame