import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integration"))

from biometric_adapters import M4VoiceAntiSpoofingAdapter, safe_verify
from evidence_validator import validate_module_evidence


M4_SOURCE = ROOT / "models" / "M4" / "m4_voice_antispoofing.py"
SPEC = importlib.util.spec_from_file_location("test_zt_m4", M4_SOURCE)
M4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M4)


class FixedModel:
    def __init__(self, logits):
        self.logits = torch.tensor([logits], dtype=torch.float32)

    def __call__(self, _features):
        return torch.zeros((1, 160)), self.logits


def _make_adapter(logits):
    implementation = M4.M4VoiceAntiSpoofing.__new__(
        M4.M4VoiceAntiSpoofing
    )
    implementation.threshold = 0.5
    implementation.device = torch.device("cpu")
    implementation.model = FixedModel(logits)

    adapter = M4VoiceAntiSpoofingAdapter.__new__(
        M4VoiceAntiSpoofingAdapter
    )
    adapter.impl = implementation
    return adapter


def _patch_preprocessing(monkeypatch):
    monkeypatch.setattr(
        M4,
        "_preprocess_audio",
        lambda _path: (
            torch.zeros((1, M4.NB_SAMPLES), dtype=torch.float32),
            {
                "input_sample_rate": M4.SR,
                "input_channels": 1,
                "input_duration_seconds": M4.NB_SAMPLES / M4.SR,
                "model_sample_rate": M4.SR,
                "model_channels": 1,
                "model_samples": M4.NB_SAMPLES,
                "model_duration_seconds": M4.NB_SAMPLES / M4.SR,
                "preprocessing": "test fixture",
            },
        ),
    )


@pytest.mark.parametrize(
    ("logits", "prediction", "decision"),
    [
        ([-5.0, 5.0], "BONAFIDE", "ACCEPT"),
        ([5.0, -5.0], "SPOOF", "REJECT"),
        ([0.0, 0.0], "SPOOF", "REJECT"),
    ],
)
def test_m4_class_mapping_and_adapter_contract(
    monkeypatch, logits, prediction, decision
):
    _patch_preprocessing(monkeypatch)
    adapter = _make_adapter(logits)

    result = adapter.verify("not-opened-by-fixture.wav")

    assert result["status"] == "completed"
    assert result["prediction"] == prediction
    assert result["decision"] == decision
    assert result["verified"] is (decision == "ACCEPT")
    assert validate_module_evidence(result)["valid"]
    assert result["raw_model_output"]["class_order"] == ["SPOOF", "BONAFIDE"]


@pytest.mark.parametrize(
    "logits",
    [
        [float("nan"), 1.0],
        [float("inf"), 1.0],
    ],
)
def test_non_finite_model_output_is_error_not_accept(monkeypatch, logits):
    _patch_preprocessing(monkeypatch)
    adapter = _make_adapter(logits)

    result = safe_verify(adapter, "test.wav")

    assert result["status"] == "error"
    assert result["decision"] == "UNKNOWN"
    assert result["verified"] is None


def test_wrong_model_output_shape_is_error_not_accept(monkeypatch):
    _patch_preprocessing(monkeypatch)
    adapter = _make_adapter([0.0, 1.0])

    class MalformedModel:
        def __call__(self, _features):
            return torch.zeros((1, 160)), torch.tensor([[0.5]])

    adapter.impl.model = MalformedModel()
    result = safe_verify(adapter, "test.wav")

    assert result["status"] == "error"
    assert result["decision"] == "UNKNOWN"
    assert result["verified"] is None


def test_malformed_completed_adapter_evidence_is_error(monkeypatch):
    _patch_preprocessing(monkeypatch)
    adapter = _make_adapter([0.0, 1.0])

    class MalformedImplementation:
        def predict(self, _path):
            return {
                "status": "completed",
                "decision": "BONAFIDE",
                "prediction": "BONAFIDE",
                "verified": True,
                "bonafide_probability": float("nan"),
                "spoof_probability": 0.9,
                "threshold": 0.5,
                "processing_ms": 1.0,
                "raw_model_output": {},
            }

    adapter.impl = MalformedImplementation()
    result = adapter.verify("test.wav")

    assert result["status"] == "error"
    assert result["decision"] == "UNKNOWN"
    assert result["verified"] is None


def test_adapter_rejects_probabilities_that_do_not_match_logits():
    adapter = _make_adapter([0.0, 1.0])

    class InconsistentImplementation:
        def predict(self, _path):
            return {
                "status": "completed",
                "decision": "ACCEPT",
                "prediction": "BONAFIDE",
                "verified": True,
                "bonafide_probability": 0.9,
                "spoof_probability": 0.1,
                "threshold": 0.5,
                "processing_ms": 1.0,
                "raw_model_output": {
                    "class_order": ["SPOOF", "BONAFIDE"],
                    "logits": [2.0, -2.0],
                    "official_scores": {"SPOOF": 2.0, "BONAFIDE": -2.0},
                    "softmax_scores": {"SPOOF": 0.1, "BONAFIDE": 0.9},
                    "softmax_scores_calibrated": False,
                },
            }

    adapter.impl = InconsistentImplementation()
    result = adapter.verify("test.wav")

    assert result["status"] == "error"
    assert result["decision"] == "UNKNOWN"
    assert result["verified"] is None


def test_missing_input_is_not_run():
    adapter = _make_adapter([0.0, 1.0])

    result = safe_verify(adapter, None)

    assert result["status"] == "not_run"
    assert result["decision"] == "UNKNOWN"
    assert result["verified"] is None


def test_preprocessing_resamples_stereo_and_reports_source_info(tmp_path):
    input_rate = 44100
    duration = 4.25
    time_values = np.arange(int(input_rate * duration), dtype=np.float32) / input_rate
    audio = np.stack(
        (
            0.2 * np.sin(2 * np.pi * 220 * time_values),
            0.1 * np.sin(2 * np.pi * 330 * time_values),
        ),
        axis=1,
    )
    path = tmp_path / "stereo_44100.wav"
    sf.write(path, audio, input_rate, subtype="PCM_16")

    features, metadata = M4._preprocess_audio(path)

    assert tuple(features.shape) == (1, M4.NB_SAMPLES)
    assert torch.isfinite(features).all()
    assert metadata["input_sample_rate"] == input_rate
    assert metadata["input_channels"] == 2
    assert metadata["model_sample_rate"] == M4.SR
    assert metadata["model_channels"] == 1
    assert metadata["model_samples"] == M4.NB_SAMPLES
    assert metadata["input_duration_seconds"] == pytest.approx(duration, abs=1 / input_rate)


def test_short_audio_is_rejected_but_valid_minimum_is_padded_to_official_length():
    with pytest.raises(ValueError, match="at least 4.0 seconds"):
        M4._fit_official_length(torch.zeros(int(M4.SR * 4.0) - 1))

    padded = M4._fit_official_length(torch.ones(int(M4.SR * 4.0)))
    assert padded.numel() == M4.NB_SAMPLES


def test_unverified_checkpoint_is_rejected(tmp_path):
    checkpoint_path = tmp_path / "not_official.pth"
    checkpoint_path.write_bytes(b"not an AASIST checkpoint")

    with pytest.raises(ValueError, match="SHA-256"):
        M4._verify_checkpoint(checkpoint_path)


def test_silent_and_non_finite_audio_are_rejected(monkeypatch):
    monkeypatch.setattr(
        M4.torchaudio,
        "load",
        lambda _path: (torch.zeros((1, M4.NB_SAMPLES)), M4.SR),
    )
    with pytest.raises(ValueError, match="silent or too quiet"):
        M4._preprocess_audio("silence.wav")

    monkeypatch.setattr(
        M4.torchaudio,
        "load",
        lambda _path: (
            torch.full((1, M4.NB_SAMPLES), float("nan")),
            M4.SR,
        ),
    )
    with pytest.raises(ValueError, match="non-finite"):
        M4._preprocess_audio("nan.wav")


def test_real_checkpoint_loads_and_cpu_infers_on_synthetic_tone(tmp_path):
    model = M4.M4VoiceAntiSpoofing(
        str(ROOT / "models" / "M4" / "aasist_2019_la.pth"),
        threshold=0.5,
    )
    sample_rate = M4.SR
    time_values = np.arange(sample_rate * 4, dtype=np.float32) / sample_rate
    tone = 0.2 * np.sin(2 * np.pi * 220 * time_values)
    path = tmp_path / "synthetic_tone.wav"
    sf.write(path, tone.astype(np.float32), sample_rate, subtype="PCM_16")

    result = model.predict(path)

    assert result["status"] == "completed"
    assert result["device"] == "cpu"
    assert result["model_sample_rate"] == sample_rate
    assert result["input_channels"] == 1
    assert result["model_samples"] == M4.NB_SAMPLES
    assert np.isfinite(result["bonafide_probability"])
    assert np.isfinite(result["spoof_probability"])
    assert result["verified"] is (result["decision"] == "BONAFIDE")
    assert result["raw_model_output"]["class_order"] == ["SPOOF", "BONAFIDE"]


def test_incompatible_asvspoof_track_config_is_rejected(monkeypatch, tmp_path):
    config_path = tmp_path / "AASIST.conf"
    config_path.write_text(
        '{"track":"PA","model_config":{"architecture":"AASIST","nb_samp":64600}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(M4, "MODEL_CONFIG_PATH", config_path)

    with pytest.raises(ValueError, match="configuration"):
        M4._load_model_config()
