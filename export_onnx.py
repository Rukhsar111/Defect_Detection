from ultralytics import YOLO

# Load your custom or pre-trained PyTorch model
model = YOLO("best.pt")  # or best.onnx

# Export the FP32 model to FP32 ONNX format
# model.export(format="onnx", dynamic=True, simplify=True)

# Export the FP32 model to ONNX format to FP16
model.export(format="onnx",half=True)


# Export the FP32 model to INT8 ONNX format
model.export(
    format="onnx",
    int8=True,
    data="configs/data.yaml",   # required for INT8 calibration
    imgsz=640,
)

