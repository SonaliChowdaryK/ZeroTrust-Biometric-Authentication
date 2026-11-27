import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integration"))

import authentication_pipeline
from audio_utils import convert_audio_file
from audio_enhancement import PreparedAudio
from app import audio_input_status, audio_payload_path
import subprocess
import wave
import numpy as np
from pathlib import Path


def test_authentication_requires_user_id():
    with pytest.raises(ValueError, match="User ID"):
        authentication_pipeline.run_authentication("face.jpg", "voice.wav", None)


def test_authentication_rejects_missing_user(tmp_path, monkeypatch):
    monkeypatch.setattr(authentication_pipeline, "_load_m1", lambda: object())
    with pytest.raises(Exception, match="not enrolled"):
        authentication_pipeline.run_authentication(
            str(tmp_path / "missing-face.jpg"),
            str(tmp_path / "missing-audio.wav"),
            "UNKNOWN",
        )


def test_input_validation_rejects_empty_face(tmp_path):
    face = tmp_path / "face.png"
    face.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        authentication_pipeline.validate_input_file(
            face, {".jpg", ".jpeg", ".png"}, "Face"
        )


def test_input_validation_rejects_unsupported_audio(tmp_path):
    audio = tmp_path / "voice.txt"
    audio.write_text("not audio", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported audio"):
        authentication_pipeline.validate_input_file(
            audio, {".wav", ".mp3", ".flac"}, "Audio"
        )


def test_audio_payload_accepts_gradio_filedata(tmp_path):
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"wav")

    class AudioPayload:
        path = str(audio)

    assert audio_payload_path(AudioPayload()) == str(audio)


def test_audio_payload_accepts_nested_filedata():
    assert audio_payload_path({"path": "C:/temp/voice.wav"}) == "C:/temp/voice.wav"


def test_audio_status_confirms_received_wav(tmp_path):
    audio = tmp_path / "voice.wav"
    with wave.open(str(audio), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(np.zeros(16000, dtype=np.int16).tobytes())
    assert "received by the app (1.0 seconds)" in audio_input_status(str(audio))


def test_audio_status_explains_missing_recording():
    assert "No voice sample has reached the app" in audio_input_status(None)


def test_audio_waveform_payload_is_materialized_and_cleaned():
    waveform = (16000, np.zeros(16000, dtype=np.int16))
    status = audio_input_status(waveform)
    assert "received by the app (1.0 seconds)" in status


def test_m4a_is_converted_to_wav(tmp_path):
    wav_path = tmp_path / "voice.wav"
    sample_rate = 16000
    samples = np.sin(2 * np.pi * 220 * np.arange(sample_rate * 2) / sample_rate)
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes((samples * 16000).astype(np.int16).tobytes())

    m4a_path = tmp_path / "voice.m4a"
    converted = subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-i", str(wav_path), "-ac", "1", "-ar", "16000", str(m4a_path)
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if converted.returncode != 0:
        pytest.skip("FFmpeg unavailable for M4A conversion test.")

    normalized = convert_audio_file(
        m4a_path,
        tmp_path / "normalized.wav",
        sample_rate=16000,
    )
    with wave.open(str(normalized), "rb") as wav_file:
        assert wav_file.getframerate() == 16000
        assert wav_file.getnchannels() == 1
        assert wav_file.getnframes() > 0


def test_authentication_preserves_original_audio_for_m3_and_m4(
    tmp_path, monkeypatch
):
    user_root = tmp_path / "enrollments" / "USER1"
    user_root.mkdir(parents=True)
    face_template = user_root / "face.npy"
    voice_template = user_root / "voice.pt"
    face_template.write_bytes(b"face template")
    voice_template.write_bytes(b"voice template")
    monkeypatch.setattr(authentication_pipeline, "ENROLLMENTS_ROOT", tmp_path / "enrollments")
    monkeypatch.setattr(
        authentication_pipeline,
        "get_user",
        lambda user_id: {
            "user_id": "USER1",
            "face_template": "face.npy",
            "voice_template": "voice.pt",
        },
    )

    face = tmp_path / "face.jpg"
    raw_upload = tmp_path / "upload.wav"
    original_audio = tmp_path / "original-16k.wav"
    enhanced_audio = tmp_path / "enhanced-16k.wav"
    for path in (face, raw_upload, original_audio, enhanced_audio):
        path.write_bytes(b"test data")

    observed_paths = {}

    class FakeAdapter:
        def __init__(self, module_id, module_name):
            self.module_id = module_id
            self.module_name = module_name

        def verify(self, path):
            observed_paths[self.module_id] = Path(path)
            return {
                "module_id": self.module_id,
                "module_name": self.module_name,
                "status": "completed",
                "decision": "ACCEPT",
                "verified": True,
            }

    monkeypatch.setattr(
        authentication_pipeline,
        "_load_m1",
        lambda: FakeAdapter("M1", "Face Anti-Spoofing"),
    )
    monkeypatch.setattr(
        authentication_pipeline,
        "_load_m2",
        lambda template_path=None: FakeAdapter("M2", "Face Recognition"),
    )
    monkeypatch.setattr(
        authentication_pipeline,
        "_load_m3",
        lambda template_path=None: FakeAdapter("M3", "Speaker Recognition"),
    )
    monkeypatch.setattr(
        authentication_pipeline,
        "_load_m4",
        lambda: FakeAdapter("M4", "Voice Anti-Spoofing"),
    )
    monkeypatch.setattr(
        authentication_pipeline,
        "run_zero_trust_engine",
        lambda evidence: {"final_decision": "ALLOW", "reason": "test"},
    )
    workspace = tmp_path / "prepared"
    workspace.mkdir()
    prepared = PreparedAudio(
        original_path=original_audio,
        enhanced_path=enhanced_audio,
        analysis={"state": "ready", "ready": True},
        workspace=workspace,
    )

    try:
        authentication_pipeline.run_authentication(
            str(face),
            str(raw_upload),
            "USER1",
            audio_preparation=prepared,
        )
    finally:
        prepared.cleanup()

    assert observed_paths["M3"] == original_audio
    assert observed_paths["M4"] == original_audio
    assert observed_paths["M3"] != enhanced_audio


def test_unsuitable_audio_is_not_sent_to_voice_models(tmp_path, monkeypatch):
    user_root = tmp_path / "enrollments" / "USER1"
    user_root.mkdir(parents=True)
    (user_root / "face.npy").write_bytes(b"face template")
    (user_root / "voice.pt").write_bytes(b"voice template")
    monkeypatch.setattr(authentication_pipeline, "ENROLLMENTS_ROOT", tmp_path / "enrollments")
    monkeypatch.setattr(
        authentication_pipeline,
        "get_user",
        lambda user_id: {
            "user_id": "USER1",
            "face_template": "face.npy",
            "voice_template": "voice.pt",
        },
    )
    monkeypatch.setattr(
        authentication_pipeline,
        "_load_m1",
        lambda: type("FaceAdapter", (), {
            "module_id": "M1",
            "module_name": "Face Anti-Spoofing",
            "verify": lambda self, path: {"module_id": "M1", "status": "completed", "decision": "ACCEPT"},
        })(),
    )
    monkeypatch.setattr(
        authentication_pipeline,
        "_load_m2",
        lambda template_path=None: type("FaceAdapter", (), {
            "module_id": "M2",
            "module_name": "Face Recognition",
            "verify": lambda self, path: {"module_id": "M2", "status": "completed", "decision": "ACCEPT"},
        })(),
    )
    monkeypatch.setattr(
        authentication_pipeline,
        "_load_m3",
        lambda template_path=None: pytest.fail("M3 must not run on unsuitable audio"),
    )
    monkeypatch.setattr(
        authentication_pipeline,
        "_load_m4",
        lambda: pytest.fail("M4 must not run on unsuitable audio"),
    )
    monkeypatch.setattr(
        authentication_pipeline,
        "run_zero_trust_engine",
        lambda evidence: {
            "final_decision": "REQUIRE_ADDITIONAL_VERIFICATION",
            "reason": "Audio was unsuitable.",
        },
    )

    face = tmp_path / "face.jpg"
    face.write_bytes(b"face")
    report = {
        "state": "error",
        "ready": False,
        "message": "No speech detected. Please record again.",
    }

    result = authentication_pipeline.run_authentication(
        str(face),
        audio_preparation=report,
        user_id="USER1",
    )

    voice_results = {
        item["module_id"]: item
        for item in result["module_evidence"]
        if item["module_id"] in {"M3", "M4"}
    }
    assert all(item["status"] == "not_run" for item in voice_results.values())
    assert all("No speech detected" in item["reason"] for item in voice_results.values())


def test_enrollment_uses_unenhanced_audio_for_m3_template(tmp_path, monkeypatch):
    face = tmp_path / "face.jpg"
    audio = tmp_path / "voice.wav"
    original_audio = tmp_path / "original-16k.wav"
    enhanced_audio = tmp_path / "enhanced-16k.wav"
    for path in (face, audio, original_audio, enhanced_audio):
        path.write_bytes(b"test data")

    workspace = tmp_path / "audio-workspace"
    workspace.mkdir()
    prepared = PreparedAudio(
        original_path=original_audio,
        enhanced_path=enhanced_audio,
        analysis={"state": "ready", "ready": True},
        workspace=workspace,
    )
    observed_paths = []

    class FakeM2:
        def generate_template(self, face_paths):
            return np.zeros(512), {"status": "enrolled", "images_used": len(face_paths)}

    class FakeM3:
        def generate_template(self, audio_paths):
            observed_paths.extend(audio_paths)
            return np.zeros(192)

    monkeypatch.setattr(authentication_pipeline, "prepare_audio_for_models", lambda path: prepared)
    monkeypatch.setattr(authentication_pipeline, "M2FaceRecognitionAdapter", FakeM2)
    monkeypatch.setattr(authentication_pipeline, "M3SpeakerRecognitionAdapter", FakeM3)
    monkeypatch.setattr(
        authentication_pipeline,
        "save_enrollment",
        lambda *args, **kwargs: {"user_id": "USER1", "username": "Test"},
    )
    reports = []

    result = authentication_pipeline.enroll_user(
        "USER1",
        "Test",
        str(face),
        str(audio),
        audio_reports=reports,
    )

    assert result["user_id"] == "USER1"
    assert observed_paths == [str(original_audio)]
    assert reports == [{"state": "ready", "ready": True}]
    assert not workspace.exists()
