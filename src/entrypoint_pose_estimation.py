from capture.dual_view_pose_estimation import DualViewProcessor
from capture.webcam import Webcam
from pose_estimation.onnx_detector import ONNXDetector
from pose_estimation.overlay_renderer import OverlayRenderer
from stream.dual_stream import DualStreamServer

webcam = Webcam()
dual_view = DualViewProcessor(webcam)

detector = ONNXDetector("src/models/yolov8n.onnx")
overlay = OverlayRenderer(detector)

dual_view.attach_detector(detector, overlay)

server = DualStreamServer(dual_view)

if __name__ == "__main__":
    server.run()