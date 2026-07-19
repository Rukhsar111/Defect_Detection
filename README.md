# Defect Detection Pipeline

Single-class ("Defect") detector fine-tuned from YOLOv8, exported to ONNX, run
through ONNX Runtime, and benchmarked FP32 vs FP16.



## 1. Setup

```bash
pip install -r requirements.txt
```

## 2. Training

```bash
python train.py --data config.yaml --model yolov8n.pt \
    --epochs 200 --imgsz 640 --batch 16 --name defect_run1
```

- Base checkpoint: `yolov8n.pt`
- Final epoch / early-stopped at: `200`
- Training hardware: `GPU`


## 3. Export to ONNX

```bash
python export_onnx.py --weights runs/train-3/weights/best.pt \
    --imgsz 640 --simplify
```



## 5. Inference on the held-out video

```bash
python Predict_video_with_log.py --model best.onnx --video holdout_clip.mp4 \
    --output_csv results.csv --output_video annotated.mp4 \
    --providers CPUExecutionProvider
```

Per-frame detections, confidence, and latency are written to `results.csv`.



## Demo Video
# Video 
Pytorch FP32 Model
Processed Frames : 45
Average Latency  : 246.96 ms
Average FPS      : 4.05
![Demo](https://github.com/Rukhsar111/Defect_Detection/blob/main/Outputs/Output_FP32pt.gif)


## FP32_ONNX_Runtime
Processed Frames : 45
Average Latency  : 141.33 ms
Average FPS      : 7.08
![Demo](https://github.com/Rukhsar111/Defect_Detection/blob/main/Outputs/Output_ONNXFP32.gif)


## FP16_ONNX_Runtime
Processed Frames : 45
Average Latency  : 122.49 ms
Average FPS      : 8.16
![Demo](https://github.com/Rukhsar111/Defect_Detection/blob/main/Outputs/Output_ONNXFP16.gif)


## 6. Hardware and measured latency 

| Item | Value |
|---|---|
| CPU / GPU | CPU |
| RAM | 8 |
| OS | Windows |
| |  |
| Input resolution | 640 x 640 |
| Test set | 45 images (also reassembled into a video clip for the per-frame video pass in step 5) |

### PyTorch vs ONNX Runtime FP32 vs FP16 (measured)

| Model | Mean latency | Mean FPS | Notes |
|---|---|---|---|
| PyTorch (`best.pt`) | 246.57 ms | 5 | Baseline, no export |
| ONNX Runtime FP32 (`best_FP32.onnx`) | 140.93 ms | 10 | ~1.32x faster than PyTorch  |
| ONNX Runtime FP16 (`best_FP16.onnx`) | 122.32 ms | 10 | **Slower than FP32**, as expected on CPU: this CPU has no native FP16 compute. |

### FP32 vs INT8 benchmark

Int8 Coversion Not supported in my Hardware.


## 7. Precision / recall on the validation split

```bash
python evaluate.py --weights runs/detect/train-3/weights/best.pt \
    --data config.yaml --conf 0.5 --iou 0.5 --split val
```

| Metric | Value |
|---|---|
| Precision | 0.916 |
| Recall | 0.773 |
| mAP50 | 0.859 |
| mAP50-95 | 0.476|
| Confidence threshold used | 0.5 |
| IoU threshold used | 0.5|

## 8. Worst failure cases on the held-out video

1.1. **Missed detection under glare / uneven exposure**
   The model produced Missed detection on   frames Which  shows strong glare / uneven exposure. 

2.A bright  highlight or lighting hot-spot overlapping the
   defect area.

3.Hairline cracks or small scratches are sometime got flickered.



