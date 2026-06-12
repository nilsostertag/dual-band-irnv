from capture.webcam import Webcam
from capture.dual_view import DualViewProcessor
from stream.dual_stream import DualStreamServer


webcam = Webcam()

dual_view = DualViewProcessor(webcam)

server = DualStreamServer(dual_view)

if __name__ == "__main__":
    server.run()