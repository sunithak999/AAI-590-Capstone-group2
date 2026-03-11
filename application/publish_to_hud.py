"""
Module: AR HUD Data Publisher

Takes parsed detection objects and publishes low-latency JSON payloads
over UDP or WebSocket for AR/HUD consumers.

Can also read an MP4 from runs/ (or test_data/), re-run YOLOv8 inference,
and write HUD-ready JSON files alongside the video or upload to S3.

S3 bucket name is read from .env (S3_BUCKET_NAME).

Example — save HUD JSON files locally:
    python3 application/publish_to_hud.py \\
        --input test_data/output_ds_1.mp4 \\
        --hud-out runs/hud_ds_1/

Example — upload .jsonl to S3 (bucket from .env):
    python3 application/publish_to_hud.py \\
        --input test_data/output_ds_1.mp4 \\
        --s3-key hud/output_ds_1.jsonl

Example — stream live over UDP while processing:
    python3 application/publish_to_hud.py \\
        --input test_data/output_ds_1.mp4 \\
        --udp-host 127.0.0.1 --udp-port 5055
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

try:
    import boto3  # type: ignore
    from botocore.exceptions import BotoCoreError, ClientError  # type: ignore
except ImportError:
    boto3 = None
    BotoCoreError = Exception
    ClientError = Exception

try:
    import websockets  # type: ignore
except ImportError:
    websockets = None


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DetectionPayload:
    class_id: int
    confidence: float
    x_center: float
    y_center: float
    width: float
    height: float
    source_id: int = 0
    frame_num: int = 0
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_message(detections: List[DetectionPayload], sequence: int) -> bytes:
    payload = {
        "schema_version": 1,
        "timestamp_ms": int(time.time() * 1000),
        "sequence": sequence,
        "detections": [d.to_dict() for d in detections],
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


# ---------------------------------------------------------------------------
# Publishers
# ---------------------------------------------------------------------------

class UdpPublisher:
    def __init__(self, host: str = "127.0.0.1", port: int = 5055) -> None:
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, payload: bytes) -> None:
        self.sock.sendto(payload, (self.host, self.port))

    def close(self) -> None:
        self.sock.close()


class WebSocketPublisher:
    def __init__(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        if websockets is None:
            raise RuntimeError(
                "websockets package not installed. Run: pip install websockets"
            )
        self.host = host
        self.port = port
        self.clients: set[Any] = set()
        self._server: Optional[Any] = None

    async def _handler(self, websocket: Any) -> None:
        self.clients.add(websocket)
        try:
            async for _ in websocket:
                pass
        finally:
            self.clients.discard(websocket)

    async def start(self) -> None:
        self._server = await websockets.serve(self._handler, self.host, self.port)

    async def send(self, payload: bytes) -> None:
        if not self.clients:
            return
        message = payload.decode("utf-8")
        await asyncio.gather(
            *(client.send(message) for client in self.clients),
            return_exceptions=True,
        )

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()


class S3Publisher:
    """Collects all frame payloads as JSON Lines and uploads to S3 on flush()."""

    def __init__(self, bucket: str, key: str) -> None:
        if boto3 is None:
            raise RuntimeError("boto3 not installed. Run: pip install boto3")
        self.bucket = bucket
        self.key = key
        self._lines: List[bytes] = []
        region = os.getenv("S3_REGION", "us-east-1")
        access_key = os.getenv("AWS_ACCESS_KEY_ID") or None
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY") or None
        self._client = boto3.client(
            "s3",
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    def collect(self, payload: bytes) -> None:
        self._lines.append(payload)

    def flush(self) -> None:
        body = b"\n".join(self._lines)
        try:
            self._client.put_object(Bucket=self.bucket, Key=self.key, Body=body)
            print(f"Uploaded s3://{self.bucket}/{self.key} ({len(self._lines)} lines)")
        except (BotoCoreError, ClientError) as exc:
            raise RuntimeError(f"S3 upload failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Video → HUD pipeline
# ---------------------------------------------------------------------------

def yolo_results_to_payloads(
    results: Any,
    frame_num: int,
    source_id: int,
    names: Dict[int, str],
) -> List[DetectionPayload]:
    """Convert a single YOLOv8 result to a list of DetectionPayload objects."""
    payloads: List[DetectionPayload] = []
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return payloads

    img_h, img_w = results.orig_shape
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h
        payloads.append(
            DetectionPayload(
                class_id=cls_id,
                confidence=conf,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
                source_id=source_id,
                frame_num=frame_num,
                label=names.get(cls_id, str(cls_id)),
            )
        )
    return payloads


def process_video_to_hud(
    input_path: Path,
    model: YOLO,
    hud_out: Optional[Path] = None,
    s3_publisher: Optional[S3Publisher] = None,
    udp_publisher: Optional[UdpPublisher] = None,
    conf: float = 0.25,
    source_id: int = 0,
) -> None:
    """
    Run YOLOv8 on every frame of input_path, convert detections to HUD
    payloads, and save locally, upload to S3, and/or stream via UDP.
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    if hud_out is not None:
        hud_out.mkdir(parents=True, exist_ok=True)

    frame_num = 0
    sequence = 0
    total_detections = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(source=frame, conf=conf, verbose=False)
        payloads = yolo_results_to_payloads(
            results[0], frame_num, source_id, model.names
        )
        total_detections += len(payloads)
        message = build_message(payloads, sequence)

        if hud_out is not None:
            (hud_out / f"frame_{frame_num:06d}.json").write_bytes(message)

        if s3_publisher is not None:
            s3_publisher.collect(message)

        if udp_publisher is not None:
            udp_publisher.send(message)

        frame_num += 1
        sequence += 1

    cap.release()
    print(
        f"Processed {frame_num} frames | "
        f"{total_detections} total detections | "
        f"source: {input_path.name}"
    )
    if hud_out is not None:
        print(f"HUD JSON files saved to: {hud_out}")

    if s3_publisher is not None:
        s3_publisher.flush()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 on a video and publish HUD-ready detection payloads."
    )
    parser.add_argument(
        "--input", required=True,
        help="Input .mp4 path (e.g. test_data/output_ds_1.mp4).",
    )
    parser.add_argument(
        "--model", default="models/yolov8n.pt",
        help="YOLOv8 model path (default: models/yolov8n.pt).",
    )
    parser.add_argument(
        "--hud-out", default="",
        help="Directory to save per-frame HUD JSON files (e.g. runs/hud_ds_1/).",
    )
    parser.add_argument(
        "--s3-key", default="",
        help="S3 object key for .jsonl upload (e.g. hud/output_ds_1.jsonl). "
             "Bucket is read from S3_BUCKET_NAME in .env.",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Detection confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--source-id", type=int, default=0,
        help="Numeric source/camera ID embedded in each payload.",
    )
    parser.add_argument(
        "--udp-host", default="",
        help="UDP host to stream payloads to (leave empty to disable UDP).",
    )
    parser.add_argument(
        "--udp-port", type=int, default=5055,
        help="UDP port (default: 5055).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    model = YOLO(args.model)
    hud_out = Path(args.hud_out) if args.hud_out else None

    s3_publisher = None
    if args.s3_key:
        bucket = os.getenv("S3_BUCKET_NAME", "")
        if not bucket:
            raise ValueError("S3_BUCKET_NAME is not set in .env")
        s3_publisher = S3Publisher(bucket=bucket, key=args.s3_key)

    udp_publisher = None
    if args.udp_host:
        udp_publisher = UdpPublisher(host=args.udp_host, port=args.udp_port)

    try:
        process_video_to_hud(
            input_path=input_path,
            model=model,
            hud_out=hud_out,
            s3_publisher=s3_publisher,
            udp_publisher=udp_publisher,
            conf=args.conf,
            source_id=args.source_id,
        )
    finally:
        if udp_publisher is not None:
            udp_publisher.close()


if __name__ == "__main__":
    main()
