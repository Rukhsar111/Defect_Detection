"""
Compute precision/recall (and mAP, via Ultralytics' own val pipeline) on the
provided validation split.

This wraps `model.val()` because Ultralytics already implements correct
IoU-matching, confidence-threshold sweeps, and per-class PR curves -- reimplementing
that by hand is a common source of subtly-wrong metrics. It's fine to depend on the
library for this step; the assignment cares that you report real numbers, not that
you reinvent PR computation from scratch.

Usage:
    python scripts/evaluate.py --weights runs/detect/defect_run1/weights/best.pt \
        --data configs/data.yaml --conf 0.5 --iou 0.5 --split val
"""
import argparse
import json

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate precision/recall on val split")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--split", type=str, default="val", choices=["val", "test"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--output_json", type=str, default="eval_results.json")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)

    metrics = model.val(
        data=args.data,
        conf=args.conf,
        iou=args.iou,
        split=args.split,
        imgsz=args.imgsz,
    )

    # metrics.box.p / .r are per-class arrays; single-class -> take index 0
    precision = float(metrics.box.p[0]) if len(metrics.box.p) else 0.0
    recall = float(metrics.box.r[0]) if len(metrics.box.r) else 0.0
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

    summary = {
        "split": args.split,
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
        "precision": precision,
        "recall": recall,
        "mAP50": map50,
        "mAP50-95": map50_95,
    }

    print(json.dumps(summary, indent=2))
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote summary to {args.output_json}")


if __name__ == "__main__":
    main()
