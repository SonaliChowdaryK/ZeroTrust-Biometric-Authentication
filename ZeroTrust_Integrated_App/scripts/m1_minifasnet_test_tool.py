#!/usr/bin/env python3
"""Standalone M1 test tool for images or an explicitly opened webcam."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integration"))

from biometric_adapters import M1FaceAntiSpoofingAdapter


def report_result(source: str, adapter: M1FaceAntiSpoofingAdapter, image) -> dict:
    started = time.perf_counter()
    result = adapter.impl.module_interface(image)
    elapsed_ms = (time.perf_counter() - started) * 1000
    details = result.get("details", {}) if isinstance(result, dict) else {}
    output = {
        "sample": source,
        "face_detection": (
            "detected" if result.get("status") == "completed"
            else result.get("face_detection", "not confirmed")
        ),
        "status": result.get("status", "error"),
        "prediction": result.get("prediction"),
        "decision": result.get("decision", "UNKNOWN"),
        "live_probability": result.get("live_probability"),
        "spoof_probability": result.get("spoof_probability"),
        "raw_model_output": details.get("raw_model_output"),
        "face_bbox": result.get("face_bbox"),
        "face_detection_confidence": result.get("face_detection_confidence"),
        "model_processing_ms": details.get("processing_ms"),
        "total_processing_ms": round(elapsed_ms, 2),
        "error": result.get("error") or result.get("reason"),
    }
    print(json.dumps(output, indent=2))
    return result


def test_images(paths: list[str], adapter: M1FaceAntiSpoofingAdapter) -> int:
    failures = 0
    for image_path in paths:
        result = report_result(image_path, adapter, image_path)
        if result.get("status") in {"error", "unavailable"}:
            failures += 1
    return failures


def test_webcam(
    adapter: M1FaceAntiSpoofingAdapter,
    camera_index: int,
    sample_limit: int,
) -> int:
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open webcam index {camera_index}.")

    samples = 0
    try:
        print(
            "Webcam ready. Position your face, press ENTER to test a frame, "
            "or type Q then ENTER to quit."
        )
        while samples < sample_limit:
            command = input(f"Capture sample {samples + 1}/{sample_limit}? [ENTER/q]: ")
            if command.strip().lower() in {"q", "quit", "exit"}:
                break
            frame = None
            for _ in range(5):
                success, frame = camera.read()
                if not success:
                    raise RuntimeError("Webcam frame capture failed.")
            samples += 1
            report_result(f"webcam sample {samples}", adapter, frame)
    finally:
        camera.release()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        nargs="+",
        help="One or more image paths; the model is loaded once for all files.",
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Open the webcam interactively; press ENTER to test a frame and Q to exit.",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Maximum webcam samples before exit (default: 20).",
    )
    args = parser.parse_args()
    if not args.images and not args.webcam:
        parser.error("Provide --images PATH [PATH ...] and/or --webcam.")
    if args.samples < 1:
        parser.error("--samples must be at least 1.")

    try:
        adapter = M1FaceAntiSpoofingAdapter()
        print("Loaded the MiniFASNetV2/V1SE fusion and RetinaFace once on CPU.")
        failures = test_images(args.images, adapter) if args.images else 0
        if args.webcam:
            failures += test_webcam(adapter, args.camera_index, args.samples)
        return 1 if failures else 0
    except Exception as exc:
        print(f"M1 test failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
