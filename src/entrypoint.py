from capture.webcam import Webcam
from stream.simple_stream import SimpleStream

webcam = Webcam()

server = SimpleStream(webcam)

if __name__ == "__main__":
    server.run()