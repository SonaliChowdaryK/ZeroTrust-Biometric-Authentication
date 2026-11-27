import importlib.util
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integration"))

from biometric_adapters import M1FaceAntiSpoofingAdapter
from evidence_validator import validate_module_evidence


M1_SOURCE = ROOT / "models" / "M1" / "m1_face_antispoofing.py"
SPEC = importlib.util.spec_from_file_location("test_zt_m1", M1_SOURCE)
M1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M1)


class FixedModel:
    def __init__(self, logits):
        self.logits = torch.tensor([logits], dtype=torch.float32)
        self.last_input = None

    def __call__(self, image_tensor):
        self.last_input = image_tensor.detach().clone()
        return self.logits


def make_model(logits, scale=2.7):
    model = M1.FaceAntiSpoofingM1.__new__(M1.FaceAntiSpoofingM1)
    model.device = torch.device("cpu")
    model.models = (
        ("MiniFASNetV2", scale, FixedModel(logits)),
        ("MiniFASNetV1SE", 4.0, FixedModel(logits)),
    )
    model._inference_lock = threading.Lock()
    model._detect_face = lambda image: {
        "bbox": [20, 20, 40, 40],
        "confidence": 0.97,
    }
    return model


@pytest.mark.parametrize(
    ("logits", "prediction", "decision"),
    [
        ([0.0, 5.0, 0.0], "LIVE", "ACCEPT"),
        ([5.0, 0.0, 0.0], "SPOOF", "REJECT"),
        ([0.0, 0.0, 5.0], "SPOOF", "REJECT"),
    ],
)
def test_official_class_mapping_and_contract(logits, prediction, decision):
    model = make_model(logits)
    adapter = M1FaceAntiSpoofingAdapter.__new__(M1FaceAntiSpoofingAdapter)
    adapter.impl = model
    result = adapter.verify(
        np.full((100, 100, 3), (11, 33, 66), dtype=np.uint8)
    )

    assert result["status"] == "completed"
    assert result["prediction"] == prediction
    assert result["decision"] == decision
    assert result["verified"] is (decision == "ACCEPT")
    assert validate_module_evidence(result)["valid"]


def test_model_input_keeps_opencv_bgr_float_pixel_values():
    model = make_model([0.0, 5.0, 0.0])
    model.predict(np.full((100, 100, 3), (11, 33, 66), dtype=np.uint8))

    tensor = model.models[0][2].last_input
    assert tuple(tensor.shape) == (1, 3, 80, 80)
    assert tensor[0, :, 0, 0].tolist() == [11.0, 33.0, 66.0]


def test_no_face_is_unavailable_not_spoof():
    model = make_model([0.0, 5.0, 0.0])
    model._detect_face = lambda image: None

    result = model.module_interface(np.zeros((100, 100, 3), dtype=np.uint8))

    assert result["status"] == "unavailable"
    assert result["decision"] == "UNKNOWN"
    assert result["verified"] is None
    assert result["face_detection"] == "not_detected"


def test_invalid_image_is_error_not_spoof(tmp_path):
    model = make_model([0.0, 5.0, 0.0])

    result = model.module_interface(tmp_path / "missing.png")

    assert result["status"] == "error"
    assert result["decision"] == "UNKNOWN"
    assert result["verified"] is None
    assert "Cannot read face image" in result["error"]


def test_inference_failure_is_error_not_spoof():
    model = make_model([0.0, 5.0, 0.0])

    class BrokenModel:
        def __call__(self, image_tensor):
            raise RuntimeError("deliberate test inference failure")

    model.models = (("MiniFASNetV2", 2.7, BrokenModel()),)
    result = model.module_interface(np.zeros((100, 100, 3), dtype=np.uint8))

    assert result["status"] == "error"
    assert result["decision"] == "UNKNOWN"
    assert result["verified"] is None
    assert result["error_type"] == "RuntimeError"


@pytest.mark.parametrize(
    "logits",
    [
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([[float("nan"), 2.0, 1.0]]),
    ],
)
def test_malformed_model_output_is_error_not_spoof(logits):
    model = make_model([0.0, 5.0, 0.0])

    class MalformedModel:
        def __call__(self, image_tensor):
            return logits

    model.models = (("MiniFASNetV2", 2.7, MalformedModel()),)
    result = model.module_interface(np.zeros((100, 100, 3), dtype=np.uint8))

    assert result["status"] == "error"
    assert result["decision"] == "UNKNOWN"
    assert result["verified"] is None


def test_adapter_preserves_face_not_detected_as_unavailable():
    adapter = M1FaceAntiSpoofingAdapter.__new__(M1FaceAntiSpoofingAdapter)

    class NoFaceImplementation:
        def module_interface(self, image):
            return {
                "module_id": "M1",
                "module_name": "Face Anti-Spoofing",
                "status": "unavailable",
                "decision": "UNKNOWN",
                "verified": None,
                "reason": "Face not detected.",
            }

    adapter.impl = NoFaceImplementation()
    result = adapter.verify("ignored.png")

    assert result["status"] == "unavailable"
    assert result["decision"] == "UNKNOWN"
    assert result["verified"] is None
    assert "Face not detected" in result["reason"]
    assert validate_module_evidence(result)["valid"]


def test_official_checkpoints_and_cpu_forward_load_strictly():
    model = M1.FaceAntiSpoofingM1()

    assert len(model.models) == 2
    with torch.inference_mode():
        for _, _, loaded_model in model.models:
            output = loaded_model(torch.zeros(1, 3, 80, 80))
            assert tuple(output.shape) == (1, 3)
            assert torch.isfinite(output).all()


def test_real_detector_reports_no_face_for_blank_input():
    model = M1.FaceAntiSpoofingM1()
    result = model.module_interface(np.zeros((320, 320, 3), dtype=np.uint8))

    assert result["status"] == "unavailable"
    assert result["decision"] == "UNKNOWN"
    assert result["verified"] is None
    assert result["face_detection"] == "not_detected"


def test_adapter_rejects_malformed_completed_evidence():
    adapter = M1FaceAntiSpoofingAdapter.__new__(M1FaceAntiSpoofingAdapter)

    class MalformedImplementation:
        def module_interface(self, image):
            return {
                "status": "completed",
                "decision": "REJECT",
                "verified": False,
                "confidence": 0.8,
                "details": {
                    "prediction": "LIVE",
                    "probabilities": {
                        "SPOOF_CLASS_0": 0.1,
                        "LIVE": 0.8,
                        "SPOOF_CLASS_2": 0.1,
                    },
                    "live_probability": 0.8,
                    "spoof_probability": 0.2,
                    "face_detection_confidence": 0.97,
                },
            }

    adapter.impl = MalformedImplementation()
    result = adapter.verify("face.png")

    assert result["status"] == "error"
    assert result["decision"] == "UNKNOWN"
    assert result["verified"] is None
