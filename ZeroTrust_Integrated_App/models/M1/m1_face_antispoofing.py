from __future__ import annotations

import importlib.util
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image


MODEL_DIRECTORY = Path(__file__).resolve().parent
MODEL_VARIANTS = (
    ("MiniFASNetV2", "2.7_80x80_MiniFASNetV2.pth", 2.7),
    ("MiniFASNetV1SE", "4_0_0_80x80_MiniFASNetV1SE.pth", 4.0),
)
DETECTOR_DEFINITION = MODEL_DIRECTORY / "deploy.prototxt"
DETECTOR_CHECKPOINT = MODEL_DIRECTORY / "Widerface-RetinaFace.caffemodel"
MODEL_SOURCE = MODEL_DIRECTORY / "vendor" / "MiniFASNet.py"
MODEL_SOURCE_REVISION = "b6d5f04ad78778917853b25c778acef6d5626d15"
MODEL_NAME = "MiniFASNetV2 + MiniFASNetV1SE official fusion"
INPUT_SIZE = (80, 80)
DETECTOR_CONFIDENCE = 0.6
CLASS_NAMES = ("SPOOF_CLASS_0", "LIVE", "SPOOF_CLASS_2")


class FaceNotDetectedError(ValueError):
    pass


def _load_architecture():
    if not MODEL_SOURCE.is_file():
        raise FileNotFoundError(f"MiniFASNet architecture source missing: {MODEL_SOURCE}")
    spec = importlib.util.spec_from_file_location("zt_official_minifasnet", MODEL_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load MiniFASNet source: {MODEL_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_model(checkpoint_path: Path, architecture_name: str, architecture):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"{architecture_name} checkpoint missing: {checkpoint_path}")

    model_factory = getattr(architecture, architecture_name, None)
    if model_factory is None:
        raise ValueError(f"Unsupported MiniFASNet architecture: {architecture_name}")
    model = model_factory(
        conv6_kernel=((INPUT_SIZE[0] + 15) // 16, (INPUT_SIZE[1] + 15) // 16)
    )
    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(f"{architecture_name} checkpoint is not a non-empty state dictionary.")

    keys = tuple(state_dict)
    has_module_prefix = tuple(key.startswith("module.") for key in keys)
    if any(has_module_prefix) and not all(has_module_prefix):
        raise ValueError(f"{architecture_name} checkpoint has inconsistent module prefixes.")
    if all(has_module_prefix):
        state_dict = {key[7:]: value for key, value in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.to(torch.device("cpu"))
    model.eval()
    return model


def _load_face_detector():
    for path in (DETECTOR_DEFINITION, DETECTOR_CHECKPOINT):
        if not path.is_file():
            raise FileNotFoundError(f"RetinaFace detector asset missing: {path}")
    detector = cv2.dnn.readNetFromCaffe(
        str(DETECTOR_DEFINITION),
        str(DETECTOR_CHECKPOINT),
    )
    if detector.empty():
        raise RuntimeError("RetinaFace detector could not be initialized.")
    return detector


class FaceAntiSpoofingM1:
    """CPU wrapper for MiniVision's official MiniFASNetV2 fusion pipeline."""

    module_id = "M1"
    module_name = "Face Anti-Spoofing"

    def __init__(self, detector=None, models=None):
        self.device = torch.device("cpu")
        self.detector = detector if detector is not None else _load_face_detector()
        if models is None:
            architecture = _load_architecture()
            self.models = tuple(
                (
                    architecture_name,
                    scale,
                    _load_model(
                        MODEL_DIRECTORY / checkpoint_name,
                        architecture_name,
                        architecture,
                    ),
                )
                for architecture_name, checkpoint_name, scale in MODEL_VARIANTS
            )
        else:
            self.models = tuple(models)
        self._inference_lock = threading.Lock()

    @staticmethod
    def _to_bgr(image):
        if isinstance(image, (str, Path)):
            bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Cannot read face image: {image}")
        elif isinstance(image, Image.Image):
            rgb = np.asarray(image.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                raise ValueError("Face image array must use uint8 pixel values.")
            if image.ndim == 2:
                bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.ndim == 3 and image.shape[2] == 3:
                bgr = image
            elif image.ndim == 3 and image.shape[2] == 4:
                bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            else:
                raise ValueError("Face image must be grayscale, BGR, or BGRA.")
        else:
            raise TypeError("Face input must be an image path, PIL image, or uint8 BGR array.")

        if bgr.size == 0 or bgr.shape[0] < 2 or bgr.shape[1] < 2:
            raise ValueError("Face image is empty or too small.")
        return np.ascontiguousarray(bgr)

    def _detect_face(self, bgr):
        height, width = bgr.shape[:2]
        aspect_ratio = width / height
        detection_image = bgr
        if width * height >= 192 * 192:
            detection_image = cv2.resize(
                bgr,
                (
                    int(192 * math.sqrt(aspect_ratio)),
                    int(192 / math.sqrt(aspect_ratio)),
                ),
                interpolation=cv2.INTER_LINEAR,
            )

        blob = cv2.dnn.blobFromImage(
            detection_image,
            scalefactor=1.0,
            mean=(104, 117, 123),
        )
        self.detector.setInput(blob, "data")
        detections = np.asarray(self.detector.forward("detection_out"))
        if detections.size == 0 or detections.size % 7:
            raise ValueError("RetinaFace returned malformed detection output.")
        detections = detections.reshape(-1, 7)
        valid = [
            row for row in detections
            if np.isfinite(row).all() and row[2] >= DETECTOR_CONFIDENCE
        ]
        if not valid:
            return None

        best = max(valid, key=lambda row: float(row[2]))
        left = max(0, min(width - 1, int(best[3] * width)))
        top = max(0, min(height - 1, int(best[4] * height)))
        right = max(0, min(width - 1, int(best[5] * width)))
        bottom = max(0, min(height - 1, int(best[6] * height)))
        if right <= left or bottom <= top:
            return None
        return {
            "bbox": [left, top, right - left + 1, bottom - top + 1],
            "confidence": float(best[2]),
        }

    @staticmethod
    def _expanded_crop(bgr, bbox, scale):
        source_height, source_width = bgr.shape[:2]
        x, y, box_width, box_height = bbox
        if box_width <= 0 or box_height <= 0:
            raise ValueError("RetinaFace returned an invalid face box.")

        scale = min(
            (source_height - 1) / box_height,
            (source_width - 1) / box_width,
            scale,
        )
        new_width = box_width * scale
        new_height = box_height * scale
        center_x = box_width / 2 + x
        center_y = box_height / 2 + y
        left = center_x - new_width / 2
        top = center_y - new_height / 2
        right = center_x + new_width / 2
        bottom = center_y + new_height / 2

        if left < 0:
            right -= left
            left = 0
        if top < 0:
            bottom -= top
            top = 0
        if right > source_width - 1:
            left -= right - source_width + 1
            right = source_width - 1
        if bottom > source_height - 1:
            top -= bottom - source_height + 1
            bottom = source_height - 1

        left = max(0, int(left))
        top = max(0, int(top))
        right = min(source_width - 1, int(right))
        bottom = min(source_height - 1, int(bottom))
        crop = bgr[top:bottom + 1, left:right + 1]
        if crop.size == 0:
            raise ValueError("RetinaFace face crop is empty.")
        return cv2.resize(crop, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)

    def predict(self, image):
        started = time.perf_counter()
        bgr = self._to_bgr(image)

        with self._inference_lock, torch.inference_mode():
            detection = self._detect_face(bgr)
            if detection is None:
                raise FaceNotDetectedError("Face not detected.")

            model_probabilities = []
            raw_model_outputs = []
            for architecture_name, scale, model in self.models:
                crop = self._expanded_crop(bgr, detection["bbox"], scale)
                # The official OpenCV pipeline keeps BGR ordering and converts
                # uint8 pixels to float32 without normalization or /255 scaling.
                tensor = torch.from_numpy(
                    np.ascontiguousarray(crop.transpose(2, 0, 1))
                ).to(dtype=torch.float32).unsqueeze(0)
                logits = model(tensor)
                if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (1, 3):
                    raise ValueError(
                        f"{architecture_name} returned malformed logits; expected shape (1, 3)."
                    )
                if not torch.isfinite(logits).all():
                    raise ValueError(f"{architecture_name} returned non-finite logits.")
                probabilities = functional.softmax(logits[0], dim=0).cpu().tolist()
                if len(probabilities) != len(CLASS_NAMES) or not all(
                    math.isfinite(float(value)) for value in probabilities
                ):
                    raise ValueError(f"{architecture_name} returned malformed probabilities.")
                model_probabilities.append(probabilities)
                raw_model_outputs.append({
                    "model": architecture_name,
                    "logits": logits[0].detach().cpu().tolist(),
                    "probabilities": [float(value) for value in probabilities],
                })

        # This arithmetic mean matches the official test.py ensemble's summed
        # softmax scores and its score division by the two loaded models.
        probabilities = np.asarray(model_probabilities, dtype=np.float64).mean(axis=0)
        class_index = int(np.argmax(probabilities))
        prediction = "LIVE" if class_index == 1 else "SPOOF"
        return {
            "prediction": prediction,
            "class_index": class_index,
            "confidence": float(probabilities[class_index]),
            "probabilities": {
                name: float(probabilities[index])
                for index, name in enumerate(CLASS_NAMES)
            },
            "live_probability": float(probabilities[1]),
            "spoof_probability": float(probabilities[0] + probabilities[2]),
            "raw_model_output": raw_model_outputs,
            "face_bbox": detection["bbox"],
            "face_detection_confidence": detection["confidence"],
            "processing_ms": (time.perf_counter() - started) * 1000,
            "model": MODEL_NAME,
        }

    def module_interface(self, image):
        try:
            result = self.predict(image)
        except FaceNotDetectedError as exc:
            return {
                "module_id": self.module_id,
                "module_name": self.module_name,
                "status": "unavailable",
                "decision": "UNKNOWN",
                "verified": None,
                "reason": str(exc),
                "face_detection": "not_detected",
            }
        except Exception as exc:
            return {
                "module_id": self.module_id,
                "module_name": self.module_name,
                "status": "error",
                "decision": "UNKNOWN",
                "verified": None,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }

        is_live = result["prediction"] == "LIVE"
        return {
            "module_id": self.module_id,
            "module_name": self.module_name,
            "status": "completed",
            "decision": "ACCEPT" if is_live else "REJECT",
            "verified": is_live,
            "confidence": result["confidence"],
            "prediction": result["prediction"],
            "probabilities": result["probabilities"],
            "live_probability": result["live_probability"],
            "spoof_probability": result["spoof_probability"],
            "face_bbox": result["face_bbox"],
            "face_detection_confidence": result["face_detection_confidence"],
            "processing_ms": result["processing_ms"],
            "model": result["model"],
            "details": result,
        }
