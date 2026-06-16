from ultralytics import YOLO
import cv2
import threading
import time
import numpy as np


class YOLODetector:
    def __init__(self):

        self.model = YOLO("yolov8n.pt")

        # wakeup call
        dummy = np.zeros((320, 568, 3), dtype=np.uint8)
        self.model.predict(dummy, verbose=False)

        self.latest_boxes = None
        self.lock = threading.Lock()

        self.running = True

    def run(self, dual_view):

        while self.running:

            frame = dual_view.get_small()

            if frame is None:
                time.sleep(0.005)
                continue

            results = self.model.predict(source=frame, verbose=False)[0]

            boxes = []

            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                boxes.append((x1, y1, x2, y2, cls, conf))

            with self.lock:
                self.latest_boxes = boxes

    def start(self, dual_view):

        import threading
        threading.Thread(
            target=self.run,
            args=(dual_view,),
            daemon=True
        ).start()

    def get_boxes(self):
        with self.lock:
            return self.latest_boxes

    def update(self, frame):

        results = self.model.predict(source=frame, verbose=False)[0]

        boxes = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            boxes.append((x1, y1, x2, y2, cls, conf))

        self.latest_boxes = boxes