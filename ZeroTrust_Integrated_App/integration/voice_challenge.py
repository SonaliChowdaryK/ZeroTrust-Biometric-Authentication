from __future__ import annotations

import json
import re
import secrets
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf


MODEL_SAMPLE_RATE = 16000
CHALLENGE_WORD_COUNT = 4
CHALLENGE_LIFETIME_SECONDS = 180
MODEL_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "asr"
    / "vosk-model-small-en-us-0.15"
)
CHALLENGE_WORDS = (
    "blue", "river", "candle", "garden", "silver", "window", "rocket", "planet",
    "yellow", "coffee", "tiger", "forest", "pillow", "basket", "orange", "mountain",
    "pencil", "rabbit", "sunset", "button", "paper", "chair", "bridge", "lemon",
    "market", "camera", "flower", "doctor", "school", "engine", "ticket", "winter",
)
_MODEL_LOCK = threading.Lock()
_MODEL = None


class VoiceChallengeUnavailable(RuntimeError):
    pass


def create_voice_challenge(now=None):
    phrase = " ".join(secrets.SystemRandom().sample(CHALLENGE_WORDS, CHALLENGE_WORD_COUNT))
    return {
        "phrase": phrase,
        "issued_at": time.time() if now is None else float(now),
    }


def is_voice_challenge_valid(challenge, now=None):
    if not isinstance(challenge, dict):
        return False
    phrase = challenge.get("phrase")
    issued_at = challenge.get("issued_at")
    if not isinstance(phrase, str) or not isinstance(issued_at, (int, float)):
        return False
    if len(_tokens(phrase)) != CHALLENGE_WORD_COUNT:
        return False
    current_time = time.time() if now is None else float(now)
    age = current_time - float(issued_at)
    return 0 <= age <= CHALLENGE_LIFETIME_SECONDS


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if not (MODEL_DIRECTORY / "conf" / "model.conf").is_file():
        raise VoiceChallengeUnavailable(
            "The offline English voice-challenge model is missing. "
            "Run python scripts\\download_voice_challenge_model.py."
        )
    try:
        from vosk import Model, SetLogLevel
    except ImportError as exc:
        raise VoiceChallengeUnavailable(
            "The offline voice-challenge package is missing. Install requirements.txt."
        ) from exc

    SetLogLevel(-1)
    with _MODEL_LOCK:
        if _MODEL is None:
            _MODEL = Model(str(MODEL_DIRECTORY))
        return _MODEL


def _tokens(text):
    return re.findall(r"[a-z]+", str(text).lower())


def _contains_challenge(transcript, phrase):
    expected = _tokens(phrase)
    actual = _tokens(transcript)
    if not expected:
        return False
    return any(
        actual[start:start + len(expected)] == expected
        for start in range(len(actual) - len(expected) + 1)
    )


def transcribe_challenge_audio(audio_path):
    model = _load_model()
    try:
        from vosk import KaldiRecognizer
    except ImportError as exc:
        raise VoiceChallengeUnavailable(
            "The offline voice-challenge package is missing. Install requirements.txt."
        ) from exc

    samples, sample_rate = sf.read(audio_path, dtype="int16", always_2d=True)
    if sample_rate != MODEL_SAMPLE_RATE:
        raise ValueError(
            f"Voice challenge audio must be {MODEL_SAMPLE_RATE} Hz; received {sample_rate} Hz."
        )
    if samples.size == 0:
        raise ValueError("Voice challenge audio contains no samples.")
    if samples.shape[1] == 1:
        mono = samples[:, 0]
    else:
        mono = np.rint(samples.astype(np.float32).mean(axis=1)).astype(np.int16)

    recognizer = KaldiRecognizer(model, MODEL_SAMPLE_RATE)
    recognizer.AcceptWaveform(mono.tobytes())
    result = json.loads(recognizer.FinalResult())
    return str(result.get("text", "")).strip()


def verify_voice_challenge(audio_path, phrase, transcriber=None):
    transcript = (
        transcriber(audio_path)
        if transcriber is not None
        else transcribe_challenge_audio(audio_path)
    )
    return {
        "matched": _contains_challenge(transcript, phrase),
        "transcript": transcript,
    }
