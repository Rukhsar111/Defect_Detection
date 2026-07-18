import csv
import cv2
from ultralytics import YOLO

# Load model
model = YOLO("best_FP32.onnx")   # swap to best_INT8.onnx to compare, once verified quantized

VIDEO_PATH = r"C:\Users\Aatif\Desktop\Artikate_Studio\output_video.mp4"
CSV_PATH = "results_FP32.csv"    # rename per model you're testing, e.g. results_INT8.csv

results = model.predict(
    source=VIDEO_PATH,
    stream=True,
    imgsz=640,
    conf=0.5,
    device="cpu",
    verbose=False
)

total_latency = 0
frame_count = 0

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame_idx", "class", "confidence", "x1", "y1", "x2", "y2",
                      "preprocess_ms", "inference_ms", "postprocess_ms", "total_latency_ms"])

    for r in results:
        frame_count += 1

        preprocess = r.speed["preprocess"]
        inference = r.speed["inference"]
        postprocess = r.speed["postprocess"]
        latency = preprocess + inference + postprocess
        total_latency += latency
        fps = 1000 / latency if latency > 0 else 0

        frame = r.plot()
        cv2.putText(frame, f"Latency: {latency:.2f} ms", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if len(r.boxes) == 0:
            writer.writerow([frame_count, "", "", "", "", "", "",
                              f"{preprocess:.2f}", f"{inference:.2f}",
                              f"{postprocess:.2f}", f"{latency:.2f}"])
            cv2.putText(frame, "No Detections", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            y = 90
            for box in r.boxes:
                cls_id = int(box.cls.item())
                cls_name = model.names[cls_id]
                conf = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                writer.writerow([frame_count, cls_name, f"{conf:.4f}",
                                  f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
                                  f"{preprocess:.2f}", f"{inference:.2f}",
                                  f"{postprocess:.2f}", f"{latency:.2f}"])

                cv2.putText(frame, f"{cls_name}: {conf:.2f}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                y += 25

        cv2.imshow("YOLO Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()

avg_latency = total_latency / frame_count
print(f"\nProcessed Frames : {frame_count}")
print(f"Average Latency  : {avg_latency:.2f} ms")
print(f"Average FPS      : {1000/avg_latency:.2f}")
print(f"Per-frame log written to: {CSV_PATH}")