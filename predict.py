# from ultralytics import YOLO

# # Load model
# model = YOLO("best.pt")   # or best.onnx

# # model = YOLO("best.onnx")   # or best.onnx


# results = model.predict(
#     # source=r"C:\Users\Aatif\Desktop\Artikate_Studio\Wood Surface Defects.v1i.yolov8\test\images/",
#     source=r"C:\Users\Aatif\Desktop\Artikate_Studio\output_video.mp4",
#     stream=True,
#     imgsz=640,
#     conf=0.5,
#     device='cpu',      # "cpu" for CPU
#     verbose=False
# )

# total_latency = 0

# for i, r in enumerate(results, 1):
#     preprocess = r.speed["preprocess"]
#     inference = r.speed["inference"]
#     postprocess = r.speed["postprocess"]
#     latency = preprocess + inference + postprocess
#     total_latency += latency

#     print(f"\nImage {i}: {r.path}")
#     print(f"Latency: {latency:.2f} ms "
#           f"(Pre: {preprocess:.2f}, Inf: {inference:.2f}, Post: {postprocess:.2f})")

#     if len(r.boxes) == 0:
#         print("No detections")
#         continue

#     for j, box in enumerate(r.boxes, 1):
#         cls_id = int(box.cls.item())
#         cls_name = model.names[cls_id]
#         conf = float(box.conf.item())
#         x1, y1, x2, y2 = box.xyxy[0].tolist()

#         print(
#             f"Detection {j}: "
#             f"Class={cls_name} "
#             f"Conf={conf:.4f} "
#             f"BBox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
#         )

# avg_latency = total_latency / i
# print("\n==========================")
# print(f"Processed Images : {i}")
# print(f"Average Latency  : {avg_latency:.2f} ms")
# print(f"Average FPS      : {1000/avg_latency:.2f}")




from ultralytics import YOLO
import cv2
import time

# Load model
# model = YOLO("best.pt")  #      #FP32 PyTorch Format
model = YOLO("best_FP32.onnx")    #FP32 ONNX Formet
# model = YOLO("best_FP16.onnx")  #    #FP16 ONNX Format
# model = YOLO("best.onnx")  #    #FP16 ONNX Format


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