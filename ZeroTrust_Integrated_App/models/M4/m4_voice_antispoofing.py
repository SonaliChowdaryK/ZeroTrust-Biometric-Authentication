import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path

import torch
import torchaudio

SR = 16000
MIN_AUDIO_SECONDS = 4.0
NB_SAMPLES = 64600
CLASS_ORDER = ("SPOOF", "BONAFIDE")
CHECKPOINT_SHA256 = (
    "51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0"
)
MODEL_CONFIG_PATH = Path(__file__).parent / "vendor" / "AASIST.conf"
MODEL_SOURCE_PATH = Path(__file__).parent / "vendor" / "AASIST.py"


def _load_model_class():
    if not MODEL_SOURCE_PATH.is_file():
        raise FileNotFoundError(f"AASIST source is missing: {MODEL_SOURCE_PATH}")
    spec = importlib.util.spec_from_file_location("zt_m4_aasist", MODEL_SOURCE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load AASIST source: {MODEL_SOURCE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model


def _load_model_config():
    with MODEL_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    if (
        config.get("track") != "LA"
        or config.get("model_config", {}).get("architecture") != "AASIST"
        or config.get("model_config", {}).get("nb_samp") != NB_SAMPLES
    ):
        raise ValueError("AASIST configuration is missing or incompatible.")
    return config["model_config"]


def _verify_checkpoint(checkpoint_path):
    digest = hashlib.sha256()
    with Path(checkpoint_path).open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != CHECKPOINT_SHA256:
        raise ValueError(
            "AASIST checkpoint SHA-256 does not match the pinned official weights."
        )


def _fit_official_length(wav):
    if wav.ndim != 1 or wav.numel() < int(SR * MIN_AUDIO_SECONDS):
        duration = wav.numel() / SR if wav.ndim == 1 else 0
        raise ValueError(
            f"Audio is too short ({duration:.2f} seconds); "
            f"M4 requires at least {MIN_AUDIO_SECONDS:.1f} seconds."
        )
    if wav.numel() >= NB_SAMPLES:
        return wav[:NB_SAMPLES]
    repeats = math.ceil(NB_SAMPLES / wav.numel())
    return wav.repeat(repeats)[:NB_SAMPLES]


def _preprocess_audio(audio_path):
    wav, original_sample_rate = torchaudio.load(str(audio_path))
    if wav.ndim != 2 or wav.shape[0] < 1 or wav.shape[1] < 1:
        raise ValueError("Audio decoder returned an empty or malformed waveform.")
    if original_sample_rate <= 0:
        raise ValueError(
            f"Audio decoder returned an invalid sample rate: {original_sample_rate}."
        )
    if not torch.isfinite(wav).all():
        raise ValueError("Audio contains non-finite samples.")

    input_channels = int(wav.shape[0])
    input_duration_seconds = float(wav.shape[1] / original_sample_rate)
    if input_duration_seconds < MIN_AUDIO_SECONDS:
        raise ValueError(
            f"Audio is too short ({input_duration_seconds:.2f} seconds); "
            f"M4 requires at least {MIN_AUDIO_SECONDS:.1f} seconds."
        )

    wav = wav.mean(dim=0)
    if original_sample_rate != SR:
        wav = torchaudio.transforms.Resample(original_sample_rate, SR)(wav)
    if not torch.isfinite(wav).all():
        raise ValueError("Resampling produced non-finite audio samples.")

    model_input = _fit_official_length(wav)
    if model_input.abs().max().item() <= 1e-8:
        raise ValueError("Audio is silent or too quiet to assess.")

    metadata = {
        "input_sample_rate": int(original_sample_rate),
        "input_channels": input_channels,
        "input_duration_seconds": input_duration_seconds,
        "model_sample_rate": SR,
        "model_channels": 1,
        "model_samples": NB_SAMPLES,
        "model_duration_seconds": NB_SAMPLES / SR,
        "preprocessing": (
            "mono channel averaging; resample to 16000 Hz; preserve source "
            "amplitude; use the first 64600 samples; repeat-pad only when an "
            "otherwise valid recording is between 4.0 and 4.0375 seconds"
        ),
    }
    return model_input.unsqueeze(0), metadata


def preprocess_audio(audio_path):
    waveform, _ = _preprocess_audio(audio_path)
    return waveform


class M4VoiceAntiSpoofing:
    def __init__(self, checkpoint_path, threshold=0.5):
        if threshold != 0.5:
            raise ValueError("The frozen M4 threshold must remain 0.5.")

        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _verify_checkpoint(checkpoint_path)
        model_config = _load_model_config()
        model_class = _load_model_class()
        self.model = model_class(model_config)

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(checkpoint, dict) or not checkpoint:
            raise ValueError("AASIST checkpoint must be a non-empty state dictionary.")
        self.model.load_state_dict(checkpoint, strict=True)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, audio_path):
        started = time.perf_counter()
        waveform, audio_info = _preprocess_audio(audio_path)
        waveform = waveform.to(self.device)

        with torch.inference_mode():
            output = self.model(waveform)
        if not isinstance(output, tuple) or len(output) != 2:
            raise ValueError("AASIST returned malformed output; expected embedding and logits.")
        embedding, logits = output
        if (
            not isinstance(embedding, torch.Tensor)
            or tuple(embedding.shape) != (1, 160)
            or not isinstance(logits, torch.Tensor)
            or tuple(logits.shape) != (1, 2)
        ):
            raise ValueError("AASIST returned malformed embedding or logits.")
        if not torch.isfinite(embedding).all() or not torch.isfinite(logits).all():
            raise ValueError("AASIST returned non-finite model output.")

        class_scores = torch.softmax(logits[0], dim=0)
        spoof_probability = float(class_scores[0].detach().cpu())
        bonafide_probability = float(class_scores[1].detach().cpu())
        if (
            not math.isfinite(spoof_probability)
            or not math.isfinite(bonafide_probability)
            or not math.isclose(
                spoof_probability + bonafide_probability,
                1.0,
                abs_tol=1e-5,
            )
        ):
            raise ValueError("AASIST returned invalid softmax scores.")

        prediction = (
            "SPOOF"
            if spoof_probability >= self.threshold
            else "BONAFIDE"
        )
        raw_logits = [float(value) for value in logits[0].detach().cpu().tolist()]
        raw_scores = {
            "SPOOF": raw_logits[0],
            "BONAFIDE": raw_logits[1],
        }
        audio_info.update({
            "status": "completed",
            "verified": prediction == "BONAFIDE",
            "decision": prediction,
            "prediction": prediction,
            "bonafide_probability": bonafide_probability,
            "spoof_probability": spoof_probability,
            "threshold": self.threshold,
            "model": "AASIST (ASVspoof 2019 LA)",
            "device": str(self.device),
            "processing_ms": (time.perf_counter() - started) * 1000,
            "raw_model_output": {
                "class_order": list(CLASS_ORDER),
                "logits": raw_logits,
                "official_scores": raw_scores,
                "softmax_scores": {
                    "SPOOF": spoof_probability,
                    "BONAFIDE": bonafide_probability,
                },
                "softmax_scores_calibrated": False,
                "official_score_semantics": (
                    "AASIST evaluation uses the BONAFIDE class logit; higher "
                    "scores support bona fide. Softmax scores are normalized "
                    "model outputs, not calibrated probabilities."
                ),
            },
        })
        return {
            "module_id": "M4",
            "module_name": "Voice Anti-Spoofing",
            **audio_info,
        }
