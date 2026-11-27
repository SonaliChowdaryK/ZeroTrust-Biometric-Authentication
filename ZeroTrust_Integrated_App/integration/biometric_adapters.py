from __future__ import annotations
from pathlib import Path
from typing import Any
import importlib.util
import math
import numbers

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

def evidence(module_id, module_name, status, decision="UNKNOWN", verified=None, **extra):
    return {"module_id": module_id, "module_name": module_name,
            "status": status, "decision": decision, "verified": verified, **extra}

def error_evidence(module_id, module_name, exc):
    return evidence(module_id, module_name, "error",
                   error_type=type(exc).__name__, error=str(exc))

def load_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module source: {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

class M1FaceAntiSpoofingAdapter:
    module_id = "M1"
    module_name = "Face Anti-Spoofing"
    def __init__(self):
        source = MODELS / "M1" / "m1_face_antispoofing.py"
        if not source.is_file():
            raise FileNotFoundError("M1 implementation source missing")
        mod = load_from_file("zt_m1", source)
        self.impl = mod.FaceAntiSpoofingM1()

    @staticmethod
    def _valid_score(value):
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            return False
        score = float(value)
        return math.isfinite(score) and 0.0 <= score <= 1.0

    @staticmethod
    def _valid_raw_outputs(raw_model_output):
        if not isinstance(raw_model_output, list) or len(raw_model_output) != 2:
            return False
        if any(not isinstance(item, dict) for item in raw_model_output):
            return False
        model_names = [item.get("model") for item in raw_model_output]
        if any(not isinstance(name, str) for name in model_names):
            return False
        expected_models = {"MiniFASNetV2", "MiniFASNetV1SE"}
        if set(model_names) != expected_models:
            return False
        for item in raw_model_output:
            logits = item.get("logits")
            probabilities = item.get("probabilities")
            if (
                not isinstance(logits, list)
                or len(logits) != 3
                or not all(
                    isinstance(value, numbers.Real)
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in logits
                )
                or not isinstance(probabilities, list)
                or len(probabilities) != 3
                or not all(M1FaceAntiSpoofingAdapter._valid_score(value) for value in probabilities)
                or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5)
            ):
                return False
        return True

    @staticmethod
    def _raw_outputs_match(raw_model_output, probabilities):
        if not M1FaceAntiSpoofingAdapter._valid_raw_outputs(raw_model_output):
            return False
        class_names = ("SPOOF_CLASS_0", "LIVE", "SPOOF_CLASS_2")
        for index, name in enumerate(class_names):
            averaged_probability = sum(
                item["probabilities"][index] for item in raw_model_output
            ) / len(raw_model_output)
            if not math.isclose(
                averaged_probability,
                float(probabilities[name]),
                abs_tol=1e-5,
            ):
                return False
        return True

    def verify(self, image_path):
        raw = self.impl.module_interface(image_path)
        if not isinstance(raw, dict):
            return evidence(
                "M1",
                self.module_name,
                "error",
                error="M1 returned malformed evidence.",
                error_type="ValueError",
            )

        status = raw.get("status")
        if not isinstance(status, str):
            return evidence(
                "M1",
                self.module_name,
                "error",
                error="M1 returned evidence without a valid status.",
                error_type="ValueError",
            )
        if status in {"error", "unavailable", "not_run"}:
            if raw.get("decision") != "UNKNOWN" or raw.get("verified") is not None:
                return evidence(
                    "M1",
                    self.module_name,
                    "error",
                    error="M1 returned inconsistent non-completed evidence.",
                    error_type="ValueError",
                )
            reason = raw.get("reason") or raw.get("error") or "M1 could not verify the face."
            extra = {"reason": str(reason)}
            if raw.get("error"):
                extra["error"] = str(raw["error"])
            if raw.get("error_type"):
                extra["error_type"] = str(raw["error_type"])
            return evidence(
                "M1",
                self.module_name,
                status,
                "UNKNOWN",
                None,
                **extra,
            )

        details = raw.get("details")
        if status != "completed" or not isinstance(details, dict):
            return evidence(
                "M1",
                self.module_name,
                "error",
                error="M1 returned malformed inference output.",
                error_type="ValueError",
            )

        prediction = details.get("prediction")
        expected_decision = {
            "LIVE": "ACCEPT",
            "SPOOF": "REJECT",
        }.get(prediction)
        confidence = raw.get("confidence")
        probabilities = details.get("probabilities")
        class_index = details.get("class_index")
        raw_model_output = details.get("raw_model_output")
        expected_class = None
        if prediction == "LIVE":
            expected_class = "LIVE"
        elif (
            prediction == "SPOOF"
            and isinstance(class_index, int)
            and not isinstance(class_index, bool)
            and class_index in {0, 2}
        ):
            expected_class = f"SPOOF_CLASS_{class_index}"
        if (
            expected_decision is None
            or raw.get("decision") != expected_decision
            or raw.get("prediction") != prediction
            or raw.get("verified") is not (expected_decision == "ACCEPT")
            or not self._valid_score(confidence)
            or not self._valid_score(details.get("confidence"))
            or not math.isclose(
                float(confidence),
                float(details["confidence"]),
                abs_tol=1e-5,
            )
            or not isinstance(class_index, int)
            or isinstance(class_index, bool)
            or class_index not in {0, 1, 2}
            or (prediction == "LIVE") != (class_index == 1)
            or not isinstance(probabilities, dict)
            or set(probabilities) != {
                "SPOOF_CLASS_0",
                "LIVE",
                "SPOOF_CLASS_2",
            }
            or not all(self._valid_score(value) for value in probabilities.values())
            or not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-5)
            or expected_class is None
            or not math.isclose(
                float(confidence),
                float(probabilities[expected_class]),
                abs_tol=1e-5,
            )
            or not self._valid_score(details.get("live_probability"))
            or not math.isclose(
                float(details["live_probability"]),
                float(probabilities["LIVE"]),
                abs_tol=1e-5,
            )
            or not self._valid_score(details.get("spoof_probability"))
            or not math.isclose(
                float(details["spoof_probability"]),
                float(probabilities["SPOOF_CLASS_0"])
                + float(probabilities["SPOOF_CLASS_2"]),
                abs_tol=1e-5,
            )
            or not self._valid_score(details.get("face_detection_confidence"))
            or not self._raw_outputs_match(raw_model_output, probabilities)
        ):
            return evidence(
                "M1",
                self.module_name,
                "error",
                error="M1 returned inconsistent or malformed model evidence.",
                error_type="ValueError",
            )

        return evidence(
            "M1",
            self.module_name,
            "completed",
            expected_decision,
            expected_decision == "ACCEPT",
            confidence=float(confidence),
            prediction=prediction,
            probabilities=probabilities,
            live_probability=details.get("live_probability"),
            spoof_probability=details.get("spoof_probability"),
            face_detection_confidence=details.get("face_detection_confidence"),
            face_bbox=details.get("face_bbox"),
            processing_ms=details.get("processing_ms"),
            raw_model_output=raw_model_output,
            model=details.get("model", "MiniFASNetV2 + MiniFASNetV1SE"),
        )

class M2FaceRecognitionAdapter:
    module_id = "M2"
    module_name = "Face Recognition"
    def __init__(self, template_path=None):
        import numpy as np
        import onnxruntime as ort
        from insightface.app import FaceAnalysis
        folder = MODELS / "M2"
        model_path = folder / "arcface_w600k_r50_fp16.onnx"
        template_path = Path(template_path) if template_path is not None else folder / "M2_face_enrollment_template.npy"
        source = folder / "m2_face_recognition.py"
        for path in (model_path, template_path, source):
            if not path.exists(): raise FileNotFoundError(f"M2 file missing: {path}")
        face_app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"],
                                providers=["CPUExecutionProvider"])
        face_app.prepare(ctx_id=-1, det_size=(640, 640))
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        mod = load_from_file("zt_m2", source)
        self.impl = mod.FaceRecognitionM2(face_app, session, threshold=0.27845)
        template = np.load(template_path).astype(np.float32)
        if template.shape != (512,): raise ValueError(f"M2 template shape unexpected: {template.shape}")
        template /= (np.linalg.norm(template) + 1e-10)
        self.impl.template = template
    def generate_template(self, image_paths):
        if not image_paths:
            raise ValueError("At least one face enrollment image is required.")
        result = self.impl.enroll([str(path) for path in image_paths])
        return self.impl.template, result
    def set_template(self, template_path):
        import numpy as np
        template = np.load(template_path, allow_pickle=False).astype(np.float32)
        if template.shape != (512,):
            raise ValueError(f"M2 template shape unexpected: {template.shape}")
        template /= (np.linalg.norm(template) + 1e-10)
        self.impl.template = template
    def verify(self, image_path):
        raw = self.impl.verify_with_evidence(str(image_path))
        raw_status, raw_decision = raw.get("status"), raw.get("decision")
        if raw_status == "success" and raw_decision in {"ACCEPT", "REJECT"}:
            return evidence("M2", self.module_name, "completed", raw_decision,
                           raw_decision == "ACCEPT", similarity=raw.get("similarity"),
                           threshold=raw.get("threshold", 0.27845),
                           model=raw.get("model", "ArcFace w600k_r50 FP16"),
                           embedding_dimension=raw.get("embedding_dimension", 512),
                           reason=raw.get("reason"))
        if raw_status == "failed" and raw_decision == "REJECT":
            return evidence("M2", self.module_name, "completed", "REJECT", False,
                           similarity=raw.get("similarity"), threshold=0.27845,
                           reason=raw.get("reason", "No valid face detected"))
        return evidence("M2", self.module_name, "error", error=raw.get("reason", "M2 verification failed"))

class M3SpeakerRecognitionAdapter:
    module_id = "M3"
    module_name = "Speaker Recognition"
    def __init__(self, template_path=None):
        folder = MODELS / "M3"
        checkpoint_dir = folder / "pretrained_models" / "spkrec-ecapa-voxceleb"
        template_path = Path(template_path) if template_path is not None else folder / "M3_voice_enrollment_template.pt"
        source = folder / "m3_speaker_recognition.py"
        if not checkpoint_dir.exists() or not template_path.exists() or not source.exists():
            raise FileNotFoundError("M3 implementation, local checkpoint, or template missing")
        mod = load_from_file("zt_m3", source)
        self.impl = mod.M3SpeakerRecognition(model_source=str(checkpoint_dir),
                                            savedir=str(checkpoint_dir),
                                            threshold=0.5235530659556389,
                                            device="cpu")
        self.impl.load_enrollment(str(template_path))
    def generate_template(self, audio_paths):
        if not audio_paths:
            raise ValueError("At least one voice enrollment sample is required.")
        return self.impl.enroll([str(path) for path in audio_paths])
    def set_template(self, template_path):
        self.impl.load_enrollment(str(template_path))
    def verify(self, audio_path):
        raw = self.impl.verify(str(audio_path))
        decision = raw.get("decision")
        if decision not in {"ACCEPT", "REJECT"}:
            return evidence("M3", self.module_name, "error", error="Unexpected M3 decision")
        return evidence("M3", self.module_name, "completed", decision, decision == "ACCEPT",
                       similarity=raw.get("similarity"), threshold=raw.get("threshold", 0.5235530659556389),
                       model=raw.get("model", "SpeechBrain ECAPA-TDNN"),
                       embedding_dimension=raw.get("embedding_dimension", 192))

class M4VoiceAntiSpoofingAdapter:
    module_id = "M4"
    module_name = "Voice Anti-Spoofing"

    def __init__(self):
        folder = MODELS / "M4"
        source = folder / "m4_voice_antispoofing.py"
        ckpt = folder / "aasist_2019_la.pth"
        if not source.is_file() or not ckpt.is_file():
            raise FileNotFoundError("M4 AASIST source/checkpoint missing")
        mod = load_from_file("zt_m4", source)
        self.impl = mod.M4VoiceAntiSpoofing(str(ckpt), threshold=0.5)

    @staticmethod
    def _valid_probability(value):
        return (
            isinstance(value, numbers.Real)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and 0.0 <= float(value) <= 1.0
        )

    @staticmethod
    def _valid_logit_score(value, logits, index):
        return (
            isinstance(value, numbers.Real)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and isinstance(logits, list)
            and len(logits) == 2
            and math.isclose(float(value), float(logits[index]), abs_tol=1e-6)
        )

    @staticmethod
    def _raw_probabilities_match_logits(logits, probabilities):
        if (
            not isinstance(logits, list)
            or len(logits) != 2
            or not isinstance(probabilities, dict)
        ):
            return False
        maximum = max(float(value) for value in logits)
        exponentials = [math.exp(float(value) - maximum) for value in logits]
        denominator = sum(exponentials)
        if not math.isfinite(denominator) or denominator <= 0:
            return False
        expected = [value / denominator for value in exponentials]
        return all(
            M4VoiceAntiSpoofingAdapter._valid_probability(probabilities.get(label))
            and math.isclose(
                float(probabilities[label]),
                expected[index],
                abs_tol=1e-5,
            )
            for index, label in enumerate(("SPOOF", "BONAFIDE"))
        )

    def verify(self, audio_path):
        raw = self.impl.predict(str(audio_path))
        if not isinstance(raw, dict) or raw.get("status") != "completed":
            return evidence(
                "M4",
                self.module_name,
                "error",
                error="M4 returned malformed inference evidence.",
                error_type="ValueError",
            )

        raw_decision = raw.get("prediction")
        threshold = raw.get("threshold")
        bonafide_probability = raw.get("bonafide_probability")
        spoof_probability = raw.get("spoof_probability")
        raw_model_output = raw.get("raw_model_output")
        raw_probabilities = (
            raw_model_output.get("softmax_scores")
            if isinstance(raw_model_output, dict)
            else None
        )
        expected_decision = (
            "SPOOF"
            if self._valid_probability(spoof_probability)
            and self._valid_probability(threshold)
            and spoof_probability >= threshold
            else "BONAFIDE"
            if self._valid_probability(spoof_probability)
            and self._valid_probability(threshold)
            else None
        )

        malformed = (
            not isinstance(raw_decision, str)
            or raw_decision not in {"BONAFIDE", "SPOOF"}
            or raw.get("decision") != raw_decision
            or raw.get("verified") is not (raw_decision == "BONAFIDE")
            or expected_decision != raw_decision
            or not self._valid_probability(bonafide_probability)
            or not self._valid_probability(spoof_probability)
            or not self._valid_probability(threshold)
            or threshold != 0.5
            or not math.isclose(
                float(bonafide_probability) + float(spoof_probability),
                1.0,
                abs_tol=1e-5,
            )
            or not isinstance(raw_model_output, dict)
            or raw_model_output.get("class_order") != ["SPOOF", "BONAFIDE"]
            or raw_model_output.get("softmax_scores_calibrated") is not False
            or not isinstance(raw_model_output.get("logits"), list)
            or len(raw_model_output.get("logits", [])) != 2
            or not all(
                isinstance(value, numbers.Real)
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in raw_model_output.get("logits", [])
            )
            or not isinstance(raw_model_output.get("official_scores"), dict)
            or not self._valid_logit_score(
                raw_model_output.get("official_scores", {}).get("SPOOF"),
                raw_model_output.get("logits", []),
                0,
            )
            or not self._valid_logit_score(
                raw_model_output.get("official_scores", {}).get("BONAFIDE"),
                raw_model_output.get("logits", []),
                1,
            )
            or not isinstance(raw_model_output.get("softmax_scores"), dict)
            or not self._valid_probability(raw_probabilities.get("BONAFIDE"))
            or not self._valid_probability(raw_probabilities.get("SPOOF"))
            or not math.isclose(
                float(raw_probabilities["BONAFIDE"]),
                float(bonafide_probability),
                abs_tol=1e-6,
            )
            or not math.isclose(
                float(raw_probabilities["SPOOF"]),
                float(spoof_probability),
                abs_tol=1e-6,
            )
            or not self._raw_probabilities_match_logits(
                raw_model_output.get("logits"), raw_probabilities
            )
            or not isinstance(raw.get("processing_ms"), numbers.Real)
            or isinstance(raw.get("processing_ms"), bool)
            or not math.isfinite(float(raw.get("processing_ms")))
            or raw.get("processing_ms") < 0
        )
        if malformed:
            return evidence(
                "M4",
                self.module_name,
                "error",
                error="M4 returned inconsistent or malformed model evidence.",
                error_type="ValueError",
            )

        decision = "ACCEPT" if raw_decision == "BONAFIDE" else "REJECT"
        return evidence(
            "M4",
            self.module_name,
            "completed",
            decision,
            decision == "ACCEPT",
            prediction=raw_decision,
            probabilities=raw_model_output["softmax_scores"],
            bonafide_probability=float(bonafide_probability),
            spoof_probability=float(spoof_probability),
            threshold=float(threshold),
            model=raw.get("model", "AASIST (ASVspoof 2019 LA)"),
            processing_ms=float(raw["processing_ms"]),
            raw_model_output=raw_model_output,
        )

def safe_verify(adapter: Any, path):
    try:
        if not path: return evidence(adapter.module_id, adapter.module_name, "not_run", reason="Input not provided")
        return adapter.verify(path)
    except Exception as exc:
        return error_evidence(adapter.module_id, adapter.module_name, exc)
