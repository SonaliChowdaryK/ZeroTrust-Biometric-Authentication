import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integration"))

from audio_enhancement import (
    AudioPreparationError,
    prepare_audio_for_models,
    suppress_noise,
)


SAMPLE_RATE = 48000


def _write_audio(path, samples, sample_rate=SAMPLE_RATE):
    sf.write(path, np.asarray(samples, dtype=np.float32), sample_rate, subtype="PCM_16")
    return Path(path)


@pytest.mark.parametrize("noise_kind", ["fan", "traffic"])
def test_rnnoise_reduces_controlled_background_noise(noise_kind):
    generator = np.random.default_rng(42)
    time = np.arange(SAMPLE_RATE * 4, dtype=np.float32) / SAMPLE_RATE
    if noise_kind == "fan":
        noise = 0.07 * np.sin(2 * np.pi * 120 * time) + 0.025 * generator.normal(
            size=time.shape
        )
    else:
        noise = 0.035 * generator.normal(size=time.shape) + 0.01 * np.sin(
            2 * np.pi * 70 * time
        )
    pcm = np.clip(np.rint(noise * 32768), -32768, 32767).astype(np.int16)[None, :]

    enhanced, speech_probability = suppress_noise(pcm)

    before_rms = float(np.sqrt(np.mean(np.square(pcm.astype(np.float32)))))
    after_rms = float(np.sqrt(np.mean(np.square(enhanced.astype(np.float32)))))
    assert after_rms < before_rms * 0.25
    assert np.isfinite(speech_probability)
    assert enhanced.dtype == np.int16


def test_silence_or_noise_is_rejected_without_changing_original(tmp_path):
    source = tmp_path / "noise.wav"
    noise = np.random.default_rng(7).normal(0, 0.02, SAMPLE_RATE * 5).astype(np.float32)
    _write_audio(source, noise)
    original_bytes = source.read_bytes()

    with pytest.raises(AudioPreparationError, match="No speech detected") as error:
        prepare_audio_for_models(source)

    assert error.value.report["noise_suppression"] == "completed"
    assert error.value.report["speech_detected"] is False
    assert error.value.report["audio_quality"] == "failed"
    assert source.read_bytes() == original_bytes


def test_recording_shorter_than_m4_window_is_rejected(tmp_path):
    source = tmp_path / "short.wav"
    time = np.arange(SAMPLE_RATE * 3, dtype=np.float32) / SAMPLE_RATE
    _write_audio(source, 0.2 * np.sin(2 * np.pi * 220 * time))

    with pytest.raises(AudioPreparationError, match="Record at least 4 seconds") as error:
        prepare_audio_for_models(source)

    assert error.value.report["noise_suppression"] == "not_run"
    assert error.value.report["audio_quality"] == "failed"
