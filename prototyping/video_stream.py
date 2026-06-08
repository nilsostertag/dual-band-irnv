from flask import Flask, Response
import cv2
import threading

app = Flask(__name__)

# Kamera DIREKT öffnen (wichtig: V4L2 + MJPEG)
camera = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
camera.set(cv2.CAP_PROP_FPS, 60)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

latest_frame = None
lock = threading.Lock()


# Hintergrund-Thread: KEIN blocking im Stream
def capture_loop():
    global latest_frame
    while True:
        ret, frame = camera.read()

        if not ret or frame is None:
            continue

        with lock:
            latest_frame = frame


threading.Thread(target=capture_loop, daemon=True).start()


def generate():
    global latest_frame

    while True:
        with lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        # nur 1x JPEG encode pro Frame
        ret, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

        if not ret:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               jpg.tobytes() +
               b"\r\n")


@app.route("/")
def index():
    return """
    <html>
        <body style="margin:0;">
            <img src="/video" style="width:100%;">
        </body>
    </html>
    """


@app.route("/video")
def video():
    return Response(generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3030, threaded=True)