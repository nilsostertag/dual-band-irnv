import cv2


class OverlayRenderer:
    def __init__(self, detector):
        self.detector = detector

    def draw(self, frame, scale_x=1.0, scale_y=1.0):

        boxes = self.detector.get_boxes()

        if not boxes:
            return frame

        for x1, y1, x2, y2, cls, conf in boxes:

            # Skalierung auf HD falls nötig
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{cls}:{conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1)

        return frame
        