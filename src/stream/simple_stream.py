from flask import Flask, Response
import cv2
import time


class SimpleStream:
    def __init__(self, webcam):
        self.webcam = webcam
        self.app = Flask(__name__)

        @self.app.route("/")
        def index():
            return """
            <html>
                <body style="margin:0;">
                    <img src="/stream" style="width:100%;">
                </body>
            </html>
            """

        @self.app.route("/stream")
        def stream():
            print("STREAM ROUTE HIT")
            return Response(
                self._gen(),
                mimetype="multipart/x-mixed-replace; boundary=frame"
            )

    def _gen(self):

        print("GENERATOR STARTED")

        while True:

            frame = self.webcam.get_frame()

            if frame is None:
                time.sleep(0.005)
                continue

            # JPEG encode (tradeoff: quality vs latency)
            ret, jpg = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 75]
            )

            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                jpg.tobytes() +
                b"\r\n"
            )

    def run(self):
        self.app.run(
            host="0.0.0.0",
            port=3030,
            threaded=True,
            debug=False
        )