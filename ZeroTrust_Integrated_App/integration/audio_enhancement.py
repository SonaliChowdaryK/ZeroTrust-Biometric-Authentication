from __future__ import annotations

import shutil
import tempfile
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pyrnnoise import RNNoise
from silero_vad import get_speech_timestamps, load_silero_vad

from audio_utils import convert_audio_file


MODEL_SAMPLE_RATE = 16000
RNNOISE_SAMPLE_RATE = 48000
MIN_RECORDING_SECONDS = 4.0
MIN_SPEECH_SECONDS = 1.0
MIN_DETECTED_SPEECH_SECONDS = 0.25
CLIPPED_SAMPLE_LIMIT = 0.01
_VAD_LOCK = threading.Lock()


class AudioPreparationError(ValueError):
    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report = report


@dataclass
class PreparedAudio:
    original_path: Path
    enhanced_path: Path
    analysis: dict
    workspace: Path

    def cleanup(self) -> None:
        shutil.rmtree(self.workspace, ignore_errors=True)


@lru_cache(maxsize=1)
def _load_vad_model():
    return load_silero_vad()


def _read_mono(path: Path) -> np.ndarray:
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != MODEL_SAMPLE_RATE:
        raise ValueError(f"Expected {MODEL_SAMPLE_RATE} Hz audio, received {sample_rate} Hz.")
    if samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("Audio contains no usable samples.")
    return samples.mean(axis=1, dtype=np.float32)


def _speech_seconds(samples: np.ndarray) -> float:
    model = _load_vad_model()
    with _VAD_LOCK:
        segments = get_speech_timestamps(
            torch.from_numpy(samples),
            model,
            sampling_rate=MODEL_SAMPLE_RATE,
            threshold=0.5,
            min_speech_duration_ms=250,
            speech_pad_ms=0,
            return_seconds=True,
        )
    return sum(segment["end"] - segment["start"] for segment in segments)


def suppress_noise(pcm_48k: np.ndarray) -> tuple[np.ndarray, float]:
    if pcm_48k.dtype != np.int16 or pcm_48k.ndim != 2 or pcm_48k.shape[0] != 1:
        raise ValueError("RNNoise input must be mono 48 kHz int16 PCM.")
    denoiser = RNNoise(sample_rate=RNNOISE_SAMPLE_RATE)
    denoised_frames = []
    speech_probabilities = []
    for probabilities, frame in denoiser.denoise_chunk(pcm_48k, partial=True):
        denoised_frames.append(np.asarray(frame).reshape(-1))
        speech_probabilities.extend(np.asarray(probabilities).reshape(-1))
    if not denoised_frames:
        raise ValueError("RNNoise did not return enhanced audio.")
    return (
        np.concatenate(denoised_frames).astype(np.int16, copy=False),
        float(np.mean(speech_probabilities)) if speech_probabilities else 0.0,
    )


def _error_report(message: str, state: str = "error") -> dict:
    return {
        "state": state,
        "raw_audio": "captured",
        "noise_suppression": "failed",
        "speech_detected": None,
        "speech_seconds": None,
        "audio_quality": "not_assessed",
        "overlap_assessment": "unavailable",
        "ready": False,
        "message": message,
        "m3_audio": "not_run",
        "m4_audio": "not_run",
    }


def prepare_audio_for_models(audio_path) -> PreparedAudio:
    source = Path(str(audio_path))
    workspace = Path(tempfile.mkdtemp(prefix="zt-audio-enhancement-"))
    report = {
        "state": "processing",
        "raw_audio": "captured",
        "noise_suppression": "processing",
        "speech_detected": None,
        "speech_seconds": None,
        "raw_speech_seconds": None,
        "audio_quality": "processing",
        "overlap_assessment": "unavailable",
        "ready": None,
        "message": "Converting and enhancing the captured audio.",
        "m3_audio": "original",
        "m4_audio": "original",
    }

    try:
        if not source.is_file() or source.stat().st_size == 0:
            raise AudioPreparationError(
                "The received audio file is missing or empty. Please record again.",
                _error_report("The received audio file is missing or empty."),
            )

        original_16k = convert_audio_file(
            source,
            workspace / "original_16k.wav",
            sample_rate=MODEL_SAMPLE_RATE,
        )
        original_samples = _read_mono(original_16k)
        duration = len(original_samples) / MODEL_SAMPLE_RATE
        peak = float(np.max(np.abs(original_samples)))
        clipped_fraction = float(np.mean(np.abs(original_samples) >= 0.999))
        report["duration_seconds"] = round(duration, 2)
        report["clipped_fraction"] = round(clipped_fraction, 5)

        if duration < MIN_RECORDING_SECONDS:
            report.update(
                state="error",
                noise_suppression="not_run",
                audio_quality="failed",
                message=f"Recording is too short. Record at least {MIN_RECORDING_SECONDS:.0f} seconds.",
                ready=False,
                m3_audio="not_run",
                m4_audio="not_run",
            )
            raise AudioPreparationError(report["message"], report)
        if peak == 0:
            report.update(
                state="error",
                noise_suppression="not_run",
                speech_detected=False,
                audio_quality="failed",
                message="No speech detected. Please record again.",
                ready=False,
                m3_audio="not_run",
                m4_audio="not_run",
            )
            raise AudioPreparationError(report["message"], report)
        if clipped_fraction > CLIPPED_SAMPLE_LIMIT:
            report.update(
                state="error",
                noise_suppression="not_run",
                audio_quality="failed",
                message="The recording is heavily clipped. Lower the microphone level and record again.",
                ready=False,
                m3_audio="not_run",
                m4_audio="not_run",
            )
            raise AudioPreparationError(report["message"], report)

        raw_speech_seconds = _speech_seconds(original_samples)
        report["raw_speech_seconds"] = round(raw_speech_seconds, 2)

        original_48k = convert_audio_file(
            source,
            workspace / "original_48k.wav",
            sample_rate=RNNOISE_SAMPLE_RATE,
        )
        raw_48k, sample_rate = sf.read(
            original_48k,
            dtype="float32",
            always_2d=True,
        )
        if sample_rate != RNNOISE_SAMPLE_RATE or raw_48k.shape[1] == 0:
            raise ValueError("Audio could not be prepared for RNNoise.")
        mono_48k = raw_48k.mean(axis=1, dtype=np.float32)
        pcm_48k = np.clip(
            np.rint(mono_48k * 32768.0),
            -32768,
            32767,
        ).astype(np.int16)[None, :]

        denoised_48k, _ = suppress_noise(pcm_48k)
        enhanced_48k = workspace / "enhanced_48k.wav"
        sf.write(
            enhanced_48k,
            denoised_48k,
            RNNOISE_SAMPLE_RATE,
            subtype="PCM_16",
        )
        enhanced_16k = convert_audio_file(
            enhanced_48k,
            workspace / "enhanced_16k.wav",
            sample_rate=MODEL_SAMPLE_RATE,
        )
        enhanced_samples = _read_mono(enhanced_16k)
        enhanced_speech_seconds = _speech_seconds(enhanced_samples)

        report.update(
            noise_suppression="completed",
            speech_detected=enhanced_speech_seconds >= MIN_DETECTED_SPEECH_SECONDS,
            speech_seconds=round(enhanced_speech_seconds, 2),
        )

        if raw_speech_seconds < MIN_DETECTED_SPEECH_SECONDS and enhanced_speech_seconds < MIN_DETECTED_SPEECH_SECONDS:
            report.update(
                state="error",
                audio_quality="failed",
                message="No speech detected. Please record again.",
                ready=False,
                m3_audio="not_run",
                m4_audio="not_run",
            )
            raise AudioPreparationError(report["message"], report)
        if raw_speech_seconds < MIN_SPEECH_SECONDS:
            report.update(
                state="error",
                audio_quality="failed",
                message="Recording is too short. Include at least one second of speech and record again.",
                ready=False,
                m3_audio="not_run",
                m4_audio="not_run",
            )
            raise AudioPreparationError(report["message"], report)
        if enhanced_speech_seconds < MIN_SPEECH_SECONDS:
            report.update(
                state="error",
                audio_quality="failed",
                message=(
                    "Speech could not be reliably detected after noise suppression. "
                    "Record again in a quieter environment."
                ),
                ready=False,
                m3_audio="not_run",
                m4_audio="not_run",
            )
            raise AudioPreparationError(report["message"], report)

        report.update(
            state="ready",
            audio_quality="passed",
            message=(
                "Noise suppression completed for speech-quality checks. "
                "M3 and M4 receive the unenhanced recording to preserve their model input characteristics."
            ),
            ready=True,
        )
        return PreparedAudio(
            original_path=original_16k,
            enhanced_path=enhanced_16k,
            analysis=report,
            workspace=workspace,
        )
    except AudioPreparationError:
        shutil.rmtree(workspace, ignore_errors=True)
        raise
    except Exception as exc:
        message = "Audio enhancement failed. Authentication cannot continue reliably."
        failed_report = _error_report(message)
        failed_report["technical_error"] = f"{type(exc).__name__}: {exc}"
        shutil.rmtree(workspace, ignore_errors=True)
        raise AudioPreparationError(message, failed_report) from exc
