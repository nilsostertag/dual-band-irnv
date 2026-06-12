import cv2
import time
from flask import Flask, Response


class DualStreamServer:
    def __init__(self, dual_view):

        self.dual_view = dual_view
        self.app = Flask(__name__)

        @self.app.route("/")
        def index():
            return """
            <!doctype html>
                <html>
                <head>
                <style>
                    body {
                        margin: 0;
                        background: #111;
                        display: flex;
                        gap: 10px;
                        padding: 10px;
                        height: 100vh;
                        box-sizing: border-box;
                    }

                    .small {
                        flex: 1;
                        max-width: 30%;
                        object-fit: contain;
                        background: black;
                    }

                    .hd {
                        flex: 2.5;
                        width: 100%;
                        object-fit: contain;
                        background: black;
                    }

                    img {
                        height: 100%;
                    }
                </style>
                </head>

                <body>
                    <img class="small" src="/small">
                    <img class="hd" src="/hd">
                </body>
                </html>
            """

        @self.app.route("/small")
        def small_stream():
            print("SMALL STREAM HIT")
            return Response(
                self._gen_small(),
                mimetype="multipart/x-mixed-replace; boundary=frame"
            )

        @self.app.route("/hd")
        def hd_stream():
            print("HD STREAM HIT")
            return Response(
                self._gen_hd(),
                mimetype="multipart/x-mixed-replace; boundary=frame"
            )

    # -------------------------
    # SMALL STREAM
    # -------------------------
    def _gen_small(self):
        print("SMALL GENERATOR STARTED")

        while True:
            frame = self.dual_view.get_small()

            if frame is None:
                time.sleep(0.005)
                continue

            ret, jpg = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 100]
            )

            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                jpg.tobytes() +
                b"\r\n"
            )

    # -------------------------
    # HD STREAM
    # -------------------------
    def _gen_hd(self):
        print("HD GENERATOR STARTED")

        while True:
            frame = self.dual_view.get_hd()

            if frame is None:
                time.sleep(0.005)
                continue

            ret, jpg = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 100]
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