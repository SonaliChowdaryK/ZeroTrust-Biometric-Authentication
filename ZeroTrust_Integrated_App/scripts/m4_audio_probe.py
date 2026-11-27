#!/usr/bin/env python3
"""Probe AASIST M4 on audio files, optionally comparing original and RNNoise audio."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import soundfile as sf
import torchaudio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integration"))

from audio_enhancement import AudioPreparationError, prepare_audio_for_models
from audio_utils import convert_audio_file
from biometric_adapters import M4VoiceAntiSpoofingAdapter


def _source_audio_info(path: Path) -> dict:
    try:
        info = torchaudio.info(str(path))
    except Exception as exc:
        return {
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
        }

    sample_rate = int(info.sample_rate)
    frames = int(info.num_frames)
    return {
        "status": "read",
        "sample_rate": sample_rate,
        "channels": int(info.num_channels),
        "frames": frames,
        "duration_seconds": frames / sample_rate if sample_rate > 0 else None,
    }


def _wav_info(path: Path) -> dict:
    info = sf.info(str(path))
    return {
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_seconds": float(info.duration),
        "format": info.format,
        "subtype": info.subtype,
    }


def _prediction_report(result: dict) -> dict:
    return {
        "status": result.get("status"),
        "prediction": result.get("prediction"),
        "decision": result.get("decision"),
        "bonafide_probability": result.get("bonafide_probability"),
        "spoof_probability": result.get("spoof_probability"),
        "threshold": result.get("threshold"),
        "raw_model_output": result.get("raw_model_output"),
        "processing_ms": result.get("processing_ms"),
        "preprocessing": result.get("preprocessing")
        or (
            "mono; 16000 Hz; preserve source amplitude; first 64600 samples; "
            "repeat-pad only for valid 4.0-4.0375 second inputs"
        ),
        "error": result.get("error") or result.get("reason"),
    }


def test_audio_file(
    path: Path,
    adapter: M4VoiceAntiSpoofingAdapter,
    compare_denoised: bool,
) -> tuple[dict, bool]:
    result = {
        "input": str(path),
        "input_exists": path.is_file(),
        "input_bytes": path.stat().st_size if path.is_file() else None,
        "source_audio_info": _source_audio_info(path) if path.is_file() else None,
        "decode_status": "not_started",
        "preprocessing_applied": (
            "FFmpeg decode to mono 16000 Hz PCM WAV; AASIST receives the "
            "un-normalized waveform at its official 64600-sample input length"
        ),
    }
    if not path.is_file():
        result.update(
            decode_status="error",
            error="Input audio file does not exist.",
        )
        return result, False

    success = True
    with tempfile.TemporaryDirectory(prefix="m4-audio-probe-") as temp_dir:
        converted_path = Path(temp_dir) / "input_16k_mono.wav"
        conversion_started = time.perf_counter()
        try:
            convert_audio_file(
                path,
                converted_path,
                sample_rate=16000,
                channels=1,
            )
            result["decode_status"] = "decoded"
            result["decode_conversion_ms"] = (
                time.perf_counter() - conversion_started
            ) * 1000
            result["model_input_audio_info"] = _wav_info(converted_path)
        except Exception as exc:
            result.update(
                decode_status="error",
                error=f"{type(exc).__name__}: {exc}",
            )
            return result, False

        try:
            original_result = adapter.verify(converted_path)
            result["original_audio"] = _prediction_report(original_result)
            if original_result.get("status") != "completed":
                success = False
        except Exception as exc:
            result["original_audio"] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            success = False

        if compare_denoised:
            prepared = None
            try:
                prepared = prepare_audio_for_models(converted_path)
                result["rnnoise_preparation"] = prepared.analysis
                denoised_result = adapter.verify(prepared.enhanced_path)
                result["denoised_audio"] = _prediction_report(denoised_result)
                if denoised_result.get("status") != "completed":
                    success = False
            except AudioPreparationError as exc:
                result["rnnoise_preparation"] = exc.report
                result["denoised_audio"] = {
                    "status": "unavailable",
                    "error": str(exc),
                }
                success = False
            except Exception as exc:
                result["denoised_audio"] = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                success = False
            finally:
                if prepared is not None:
                    prepared.cleanup()

    return result, success


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "audio_files",
        nargs="+",
        type=Path,
        help="Audio files to test (WAV, MP3, FLAC, OGG, M4A, AAC, or WEBM).",
    )
    parser.add_argument(
        "--compare-denoised",
        action="store_true",
        help="Also test the RNNoise-enhanced copy; the original is always tested.",
    )
    args = parser.parse_args()

    try:
        adapter = M4VoiceAntiSpoofingAdapter()
    except Exception as exc:
        print(
            json.dumps({
                "status": "error",
                "error": f"Could not load M4: {type(exc).__name__}: {exc}",
            }, indent=2),
            file=sys.stderr,
        )
        return 2

    print(json.dumps({
        "model_loading": {
            "status": "completed",
            "model": "AASIST (ASVspoof 2019 LA)",
            "device": str(adapter.impl.device),
        }
    }, indent=2))

    failures = 0
    for path in args.audio_files:
        result, succeeded = test_audio_file(path, adapter, args.compare_denoised)
        print(json.dumps(result, indent=2, allow_nan=False))
        failures += not succeeded
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
