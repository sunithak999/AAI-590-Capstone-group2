"""
Test module: run YOLOv8 on an MP4 video and display detections.

Example:
python3 test_mp4_yolov8.py \
  --input test_data/output_ds_1.mp4 \
  --output runs/annotated_ds_1.mp4 \
  --conf 0.25
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv8 on an MP4 file and visualize output.")
    parser.add_argument("--model", default="models/yolov8n.pt", help="Path to YOLOv8 model (.pt/.engine/.onnx).")
    parser.add_argument("--input", required=True, help="Path to input .mp4 file (e.g. test_data/output_ds_1.mp4).")
    parser.add_argument("--output", default="", help="Optional output .mp4 path for annotated video (e.g. runs/annotated.mp4).")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--device", default="", help="Device, e.g. 'cpu', '0', '0,1'.")
    parser.add_argument("--no-show", action="store_true", help="Disable display window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    model = YOLO(args.model)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0
    start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(
            source=frame,
            conf=args.conf,
            device=args.device if args.device else None,
            verbose=False,
        )
        annotated = results[0].plot()
        frame_count += 1

        if writer is not None:
            writer.write(annotated)

        if not args.no_show:
            cv2.imshow("YOLOv8 MP4 Test", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    elapsed = max(time.time() - start, 1e-6)
    print(f"Processed {frame_count} frames in {elapsed:.2f}s ({frame_count / elapsed:.2f} FPS).")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
