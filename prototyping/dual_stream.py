from flask import Flask, Response
import cv2
import threading

app = Flask(__name__)

camera = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
camera.set(cv2.CAP_PROP_FPS, 25)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

latest_frame = None
lock = threading.Lock()


def capture_loop():
    global latest_frame
    while True:
        ret, frame = camera.read()
        if not ret:
            continue

        with lock:
            latest_frame = frame


threading.Thread(target=capture_loop, daemon=True).start()


# --- helper: resize once per stream ---
def encode_frame(frame, size, quality=80):
    resized = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
    ret, jpg = cv2.imencode(
        ".jpg",
        resized,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    return jpg.tobytes() if ret else None


def stream_large():
    global latest_frame
    while True:
        with lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        data = encode_frame(frame, (1280, 720), 80)
        if not data:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               data +
               b"\r\n")


def stream_small():
    global latest_frame
    while True:
        with lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        data = encode_frame(frame, (320, 320), 70)
        if not data:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               data +
               b"\r\n")


@app.route("/")
def index():
    return """
    <html>
    <body style="display:flex; gap:10px; background:#111;">

        <div>
            <h3 style="color:white">HD Stream</h3>
            <img src="/large">
        </div>

        <div>
            <h3 style="color:white">Thumb Sync (320x320)</h3>
            <img src="/small">
        </div>

    </body>
    </html>
    """


@app.route("/large")
def large():
    return Response(stream_large(),
        mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/small")
def small():
    return Response(stream_small(),
        mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3030, threaded=True)