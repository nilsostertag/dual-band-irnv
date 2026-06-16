import onnxruntime as ort
import numpy as np
import cv2


class ONNXDetector:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

        # YOLOv8 default
        self.img_size = 320
        self.conf_thres = 0.25

        # shared state
        self.latest_boxes = []

    # -------------------------
    # PREPROCESS
    # -------------------------
    def preprocess(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    # -------------------------
    # INFER
    # -------------------------
    def infer(self, frame):
        if frame is None or not isinstance(frame, np.ndarray):
            return []

        inp = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: inp})
        return self.postprocess(outputs)

    # -------------------------
    # POSTPROCESS (FIXED)
    # -------------------------
    def postprocess(self, outputs):
        if outputs is None or len(outputs) == 0:
            return []

        pred = outputs[0]

        # --- handle YOLOv8 shapes ---
        if pred.ndim == 3:
            pred = pred[0]

        boxes = []

        for p in pred:
            # YOLOv8 format: [x, y, w, h, conf, ...classes]
            conf = p[4]

            if conf < self.conf_thres:
                continue

            x, y, w, h = p[0], p[1], p[2], p[3]

            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2

            boxes.append((x1, y1, x2, y2, float(conf)))

        return boxes
    
    def update(self, frame):
        boxes = self.infer(frame)
        self.last_boxes = boxes
        return boxes

    def get_boxes(self):
        return self.last_boxes