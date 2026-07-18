from ultralytics import YOLO
import cv2
import time

# Load model
model = YOLO(r"best.pt")  #      #FP32 PyTorch Format
# model = YOLO(r"exported_models\best_FP32.onnx")    #FP32 ONNX Formet
# model = YOLO(r"exported_models\best_FP16.onnx")  #    #FP16 ONNX Format
# model = YOLO(r"exported_models\best.onnx")  #    #FP16 ONNX Format


results = model.predict(
    source=r"C:\Users\Aatif\Desktop\Artikate_Studio\output_video.mp4",
    stream=True,
    imgsz=640,
    conf=0.5,
    device="cpu",
    verbose=False
)

total_latency = 0
frame_count = 0

for r in results:
    frame_count += 1

    # Get annotated frame (bounding boxes + labels)
    frame = r.plot()

    # Timing
    preprocess = r.speed["preprocess"]
    inference = r.speed["inference"]
    postprocess = r.speed["postprocess"]

    latency = preprocess + inference + postprocess
    total_latency += latency
    fps = 1000 / latency if latency > 0 else 0

    # Draw latency & FPS
    cv2.putText(
        frame,
        f"Latency: {latency:.2f} ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    # Display detections
    y = 90
    if len(r.boxes) == 0:
        cv2.putText(
            frame,
            "No Detections",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
    else:
        for box in r.boxes:
            cls_id = int(box.cls.item())
            cls_name = model.names[cls_id]
            conf = float(box.conf.item())

            text = f"{cls_name}: {conf:.2f}"

            cv2.putText(
                frame,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )
            y += 25

    cv2.imshow("YOLO Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

avg_latency = total_latency / frame_count
print(f"\nProcessed Frames : {frame_count}")
print(f"Average Latency  : {avg_latency:.2f} ms")
print(f"Average FPS      : {1000/avg_latency:.2f}")