from ultralytics import YOLO

# load pretrained nano model
model = YOLO("yolov8n.pt")

# export to ONNX
model.export(
    format="onnx",
    imgsz=320,
    dynamic=False,
    simplify=True,
    opset=12
)

print("Export done → yolov8n.onnx")