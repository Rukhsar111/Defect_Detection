# Section 1 — Diagnose a Failing CV Pipeline


## Scenario A — Accuracy drops after quantization
A YOLOv8-based defect detector scores 0.91 mAP@0.5 in PyTorch FP32. After exporting to ONNX and quantizing to INT8 for a Jetson Orin deployment, mAP drops to 0.58 on the same validation set. The client's inspection line cannot tolerate this. No architecture or training data changed between the two evaluations.
What would you check first, and in what order?
Name at least three distinct root causes that could each independently produce this exact symptom, and describe a test that would distinguish between them.
Give your fix, including how you'd validate it before redeploying to the client's line.



## What would you check first, and in what order?
From 0.91 map to 0.58 a way too low which is very high. After Quantizing , Accuracy can be impacted  between 2-4% but this number too large. We can check certain things:

I would debug this systematically by isolating where the accuracy is being lost rather than assuming quantization is the problem. Since the architecture and dataset are unchanged, the drop must come from the deployment pipeline.

1. Validate ONNX FP32 against PyTorch - If ONNX already fails, investigate export, preprocessing, or tensor layout.

2. Validate TensorRT FP16 - If FP16 matches PyTorch but INT8 fails, focus on quantization/calibration.

3. Compare preprocessing - Ensure identical resize, color conversion, normalization, and tensor.

4. Compare postprocessing - Verify confidence thresholds, IoU thresholds, NMS implementation, box decoding, and coordinates Mismatching.




## Name at least three distinct root causes that could each independently produce this exact symptom, and describe a test that would distinguish between them.
Root causes:
1. IF there is any preprocessing or postprocessing   ie Resize , NMS Difference, IOU change introduced it  can cause a dip in the  Accuracy.
2. Check If the dataset quality is matching and  not vary Even a color-space mismatch can dramatically reduce mAP.
3. There  could be export engine issue as well like ONNX Export Issues.


## Give your fix, including how you'd validate it before redeploying to the client's line.

Fix
Run a small validation set and compare predictions image-by-image.

I would evaluate the model at every stage using the same validation dataset and compare outputs.
1. check the inference at pytorch FP32ie 0.91map
2. compare it for FP16 it should be  between 0.90-0.91
3. ONNX FP32	Should also be  0.91
4. TensorRT FP16   would be between 	0.90-0.91
5. TensorRT INT8	Should remain close (typically within 1–3%) Not more then that.

Apart from above Ensure:
1. correct tensor layout
2. proper normalization
3. correct channel ordering



# Scenario B — Bounding boxes drift on one camera feed only
A multi-camera PPE-compliance system runs the same model across 12 RTSP feeds. Eleven feeds detect correctly. On one feed, boxes are consistently offset — always shifted, never randomly wrong — and the offset is larger for objects near the frame edges than near the center.


## What does the offset pattern (systematic, edge-dependent) tell you about where in the pipeline the bug likely lives?

If detections are working fine for the 11 feeds and only one camera feed has offset Issue  then this could be a Image Acquisition Problem either the Image Translation occurred or Camera Configuration changed.

1. The issue is almost certainly camera-specific preprocessing, image transformation, or coordinate Mismatch.
2. Error grows with distance from center  this is the specific case  of  lens distortion or  mismatched intrinsics parameter  mapping.
3. A wrong crop offset or ROI misalignment  and Wrong FOV also gives you a  constant pixel shift across the frame.



## List the specific preprocessing or camera-configuration differences you'd check for that one feed.
Specific things to check.

1. Comparing Resolution and aspect ratio  of that particular camera  with  the other 11 feeds. 
2. Lens/FOV — If  this camera a different model, different focal length, or different Lens.
3. Resize Parameter- if  this feed's  resolution being resized with a different scale.
4. Verify camera-specific parameters such as resolution, ROI, lens calibration, and digital zoom settings.



## What's your root-cause hypothesis, and how would you confirm it without physical access to the camera?

Root-cause hypothesis and how to confirm remotely
1. That camera has a different lens/FOV  compared to the other 11.
2. Camera calibration or ROI config entry is misassigned.

How to confirm without physical access?
1. Compare stream  (resolution, codec, FPS, aspect ratio) with other 11  feeds.
2. Check the per-camera config store (calibration matrices, ROI) for that camera's ID — verify it's using its own  saved parameters. 
3. Record a few seconds of the problematic  stream and run it through the same inference pipeline offline.
If the offset persists, the issue is in preprocessing or post processingnot the live stream.


# Scenario C — Model degrades over three months in production, no redeploys
A conveyor-belt object-counting model was deployed at 97% accuracy. Three months later, with zero code or model changes, accuracy has drifted to 84%. The client has not changed the physical setup that anyone reported.
Propose two or three plausible causes for this kind of silent drift with no code changes.
For each, describe what evidence in logs, images, or metrics would confirm or rule it out.
Design a lightweight monitoring signal that would have caught this drift within the first two weeks, not three months.


## Propose two or three plausible causes for this kind of silent drift with no code changes.
Plausible causes would be :
1. There could be  data distribution shift happens if the client has not changed any physical setup. A new variant of data coming to the production line which have similar physical appearance to previous data  But there might be slight change Like color, print, sometimes size. 

2. If the setup not changes their could be physical environmental change occurs usually common in  factory floors which leads to Lightning shift, Vibration is very common in production line that could loose your camera nut  sometimes which leads to change in FOV or Focus over time.


## For each, describe what evidence in logs, images, or metrics would confirm or rule it out.
Evidence to confirm out :

1. Compare the class, size distribution in recent frames against the training distribution  if the model was trained on some other dimension and new product dimension is changed we can check these product statistics.

2. Manually review a sample of recent misclassified or miscounted frames  for consistent failure mode, on which pattern, image type  it failing most which would point to a true distribution shift rather than a hardware issue.


## Design a lightweight monitoring signal that would have caught this drift within the first two weeks, not three months.

1. Monitor the Input Data Distribution and compare it with old one to check any kind deviation occurred  in new data  ie any change in  pattern, Brightness, Contrast and sharpness of image Quality in real time.
2. Track the Model Performance  eg  Model confidence scores over time.
3. Track the Detection count and flag if it deviates from the historical Accuracy.
4. Track object bounding box size, a shift here  flags either a new product variant or a camera focus/zoom change.



# Section 4 — Edge & Air-Gapped Deployment Design 
Client Alpha runs a manufacturing floor with 8 fixed cameras at 1080p, 15fps each, feeding a single on-prem server with one NVIDIA Jetson AGX Orin (64GB) — no internet access, ever. They need per-frame defect detection across all 8 feeds simultaneously, with end-to-end latency under 200ms per frame and zero cloud dependency, including for model updates.
Answer with real numbers where relevant, not just named tools:
What model size/family and precision (FP16/INT8) would you target to fit this latency budget across 8 concurrent streams on one Orin, and what's your reasoning?
Estimate the aggregate throughput you need (frames/second across all feeds) and check it against what your chosen model/precision combination can realistically deliver on that hardware. Show the arithmetic.
Design the retraining loop: how do operators flag false positives/negatives on an air-gapped device, how does that feedback get back into a retraining pipeline with no internet, and how do you validate a new model before it replaces the one running in production?
What's your rollback plan if a newly deployed model regresses on the floor, and how quickly would you detect the regression?
Flag anything you're genuinely unsure about rather than smoothing over it — we'd rather see "I'd need to benchmark X before committing" than a confident number you haven't verified.



## What model size/family and precision (FP16/INT8) would you target to fit this latency budget across 8 concurrent streams on one Orin, and what's your reasoning?
## 1. Model size/precision
I'll try with the smallest Model ie YOLOv8n as its lighter in weight and Accurate.

The Preciion I choose would be INT8  as it  gives 1.5-2x througput over FP16 which can be highly efficient and smooth on 8 Concurrent streams.



## Estimate the aggregate throughput you need (frames/second across all feeds) and check it against what your chosen model/precision combination can realistically deliver on that hardware. Show the arithmetic.
Throughput Estimates:
8 x 15FPS = 120 FPS (15 fps per camera)
Latency budget (200ms total) is Required.
For Capturing Usually  it takes between 5-12ms
for preprocessing On orion It takes between - 2-5ms
For Postprocessing  - 2-5ms

So for each frame the Latency should be between 8-15ms.



## Design the retraining loop: how do operators flag false positives/negatives on an air-gapped device, how does that feedback get back into a retraining pipeline with no internet, and how do you validate a new model before it replaces the one running in production?

We can use Human in the Loop for Evaluation and Retraining. 
1. Prediction & Observation: The production model makes inferences, and user actions or human reviews observe whether the predictions were correct. 
2. Data Logging: Mistakes, false positives, or edge cases are captured and stored Local folder.
3. Trigger: The pipeline initiates retraining based on a set schedule, when a performance metric drops below a specific threshold, or when enough new data has accumulated.
4. Evaluation & Rollout: The retrained model is validated against an sample test dataset to ensure improvement before being safely promoted to production.	




## What's your rollback plan if a newly deployed model regresses on the floor, and how quickly would you detect the regression?
Model Versioning
1. Its a best practice to  keep atleast  Last 2 or 3 Models to the system Using Model Version Control.
2. Model Versioning during deployment would allow for the simple technical solution to stay up and running as the new model is deployed to the public. In the event something goes wrong, the model can be rolled back and the simple technical solution can be pushed forward as the Machine Learning team works to fix the problem.


