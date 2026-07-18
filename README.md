# Defect Detection Pipeline

Single-class ("Defect") detector fine-tuned from YOLOv8, exported to ONNX, run
through ONNX Runtime, and benchmarked FP32 vs INT8.



## 1. Setup

```bash
pip install -r requirements.txt
```

## 2. Training

```bash
python scripts/train.py --data configs/data.yaml --model yolov8n.pt \
    --epochs 100 --imgsz 640 --batch 16 --name defect_run1
```

- Base checkpoint: `yolov8n.pt`
- Final epoch / early-stopped at: `100`
- Training hardware: `GPU`


## 3. Export to ONNX

```bash
python scripts/export_onnx.py --weights runs/detect/defect_run1/weights/best.pt \
    --imgsz 640 --simplify
```



## 5. Inference on the held-out video

```bash
python scripts/infer_video.py --model best.onnx --video holdout_clip.mp4 \
    --output_csv results.csv --output_video annotated.mp4 \
    --providers CPUExecutionProvider
```

Per-frame detections, confidence, and latency are written to `results.csv`.

> Note: this clip was reassembled from the same 45 test images rather than a
> continuously-captured video, so `timestamp_s` in the CSV is synthetic (derived
> from the fixed FPS you encoded it at) rather than a real capture time — treat
> `frame_idx` as the reliable identifier when referencing failure cases below,
> and mention this in the writeup so it's not mistaken for a real video capture.

## 6. Hardware and measured latency (REQUIRED — fill in with real numbers)

| Item | Value |
|---|---|
| CPU / GPU | `[fill in — e.g. Intel i7-12700, 12 cores]` |
| RAM | `[N GB]` |
| OS | `[e.g. Ubuntu 22.04 / Windows 11]` |
| ONNX Runtime execution provider actually used | `CPUExecutionProvider` (confirmed via ORT's own log line — no GPU/TensorRT involved) |
| Input resolution | `[640x640 — confirm]` |
| Test set | 45 images (also reassembled into a video clip for the per-frame video pass in step 5) |

### PyTorch vs ONNX Runtime FP32 vs FP16 (measured)

| Model | Mean latency | Mean FPS | Notes |
|---|---|---|---|
| PyTorch (`best.pt`) | 106.57 ms | 9.38 | Baseline, no export |
| ONNX Runtime FP32 (`best_FP32.onnx`) | 80.93 ms | 12.36 | ~1.32x faster than PyTorch — ORT's graph optimizations (op fusion, constant folding) account for this even before any precision change |
| ONNX Runtime FP16 (`best_FP16.onnx`) | 97.32 ms | 10.27 | **Slower than FP32**, as expected on CPU: this CPU has no native FP16 compute path, so ORT upcasts FP16 weights/activations to FP32 before every op, paying a conversion cost with none of FP16's benefit. FP16 only pays off on hardware with native FP16 support (GPU tensor cores, some ARM/Intel FP16 ISA extensions) — not commodity x86 CPU inference. |

### FP32 vs INT8 benchmark

INT8 tooling (`onnxruntime.quantization.quantize_dynamic`) is available and used here
instead of relying on the FP16 result above, per the assignment's precision order —
FP16 is reported above for completeness since it was already measured, but INT8 is
the real CPU-appropriate comparison:

```bash
python scripts/quantize_onnx.py --input best_FP32.onnx --output best_INT8.onnx
python scripts/benchmark.py --models fp32=best_FP32.onnx int8=best_INT8.onnx \
    --imgsz 640 --runs 200 --warmup 20 --providers CPUExecutionProvider
```

| Model | Mean latency (ms) | Median (ms) | P95 (ms) | FPS |
|---|---|---|---|---|
| FP32 | 80.93 | `[N]` | `[N]` | 12.36 |
| INT8 | `[N — run above]` | `[N]` | `[N]` | `[N]` |

**Confidence these numbers hold on a different machine:** low-to-medium for the
absolute millisecond values — this is a single CPU, single-threaded ORT session, and
absolute latency will shift with core count, clock speed, and whether the target CPU
has AVX2/AVX-512/VNNI (VNNI in particular accelerates the int8 GEMM kernels INT8
quantization relies on, so the FP32→INT8 speedup ratio itself is *more* likely to
transfer to another modern x86 CPU than the raw ms numbers are, but won't transfer to
ARM or older CPUs without VNNI-equivalent instructions).

## 7. Precision / recall on the validation split

```bash
python scripts/evaluate.py --weights runs/detect/defect_run1/weights/best.pt \
    --data configs/data.yaml --conf 0.5 --iou 0.5 --split val
```

| Metric | Value |
|---|---|
| Precision | 0.836 |
| Recall | 0.744 |
| mAP50 | 0.819 |
| mAP50-95 | 0.411 |
| Confidence threshold used | `[0.5]` |
| IoU threshold used | `[0.5]` |

## 8. Worst failure cases on the held-out video

Pick 2–3 concrete frames/timestamps from `results.csv` / `annotated.mp4` where the
model was clearly wrong (missed a real defect, false-positived on a clean region, or
had a confidence collapse), and give your actual hypothesis for each. Example
structure — replace with your real findings:

1. **`[timestamp, e.g. 00:04.2]`** — `[what went wrong, e.g. "missed a hairline
   crack running parallel to a strong specular highlight"]`. Hypothesis:
   `[e.g. "training set has few examples of defects that align with reflective
   glare; model may be learning glare-edge != defect-edge as a shortcut"]`.
2. **`[timestamp]`** — `[...]`. Hypothesis: `[...]`.
3. **`[timestamp]`** — `[...]`. Hypothesis: `[...]`.

## 9. Repo / process notes

This repo's commit history should show, in order: data loading / dataset.yaml setup,
first training run, at least one failed attempt (bad hyperparameters, wrong image
size, whatever actually happened) and the fix, the ONNX export step, and the
benchmarking step. Squashing this into one commit defeats the point — leave the real
sequence in.
