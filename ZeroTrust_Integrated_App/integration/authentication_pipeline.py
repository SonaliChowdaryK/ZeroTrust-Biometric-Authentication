from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from biometric_adapters import (M1FaceAntiSpoofingAdapter, M2FaceRecognitionAdapter,
    M3SpeakerRecognitionAdapter, M4VoiceAntiSpoofingAdapter, safe_verify)
from central_zero_trust_engine import run_zero_trust_engine
from enrollment_store import get_user, save_enrollment, user_exists
from audio_enhancement import AudioPreparationError, PreparedAudio, prepare_audio_for_models

ENROLLMENTS_ROOT = Path(__file__).resolve().parents[1] / "enrollments"

@lru_cache(maxsize=1)
def _load_m1(): return M1FaceAntiSpoofingAdapter()
@lru_cache(maxsize=16)
def _load_m2(template_path=None): return M2FaceRecognitionAdapter(template_path)
@lru_cache(maxsize=16)
def _load_m3(template_path=None): return M3SpeakerRecognitionAdapter(template_path)
@lru_cache(maxsize=1)
def _load_m4(): return M4VoiceAntiSpoofingAdapter()

def validate_input_file(path, allowed_suffixes, kind):
    if not path:
        raise ValueError(f"{kind} input is required.")
    file_path = Path(str(path))
    if not file_path.exists():
        raise ValueError(f"{kind} path does not exist.")
    if not file_path.is_file():
        raise ValueError(f"{kind} path is not a file.")
    if file_path.stat().st_size == 0:
        raise ValueError(f"{kind} file is empty.")
    if file_path.suffix.lower() not in allowed_suffixes:
        raise ValueError(f"Unsupported {kind.lower()} format: {file_path.suffix or 'unknown'}")
    return file_path

def enroll_user(user_id, username, face_images, audio_samples, allow_update=False, audio_reports=None):
    if not face_images:
        raise ValueError("At least one face enrollment image is required.")
    if not audio_samples:
        raise ValueError("At least one voice enrollment sample is required.")

    face_paths = face_images if isinstance(face_images, (list, tuple)) else [face_images]
    audio_paths = audio_samples if isinstance(audio_samples, (list, tuple)) else [audio_samples]

    for face_path in face_paths:
        validate_input_file(face_path, {".jpg", ".jpeg", ".png", ".bmp", ".webp"}, "Face")
    prepared_audio = []
    try:
        for audio_path in audio_paths:
            validate_input_file(audio_path, {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm"}, "Audio")
            try:
                preparation = prepare_audio_for_models(audio_path)
            except AudioPreparationError as exc:
                if audio_reports is not None:
                    audio_reports.append(exc.report)
                raise
            prepared_audio.append(preparation)
            if audio_reports is not None:
                audio_reports.append(preparation.analysis)

        m2 = M2FaceRecognitionAdapter()
        face_template, face_result = m2.generate_template(face_paths)
        if face_result.get("status") != "enrolled":
            raise ValueError("Face template generation failed.")

        m3 = M3SpeakerRecognitionAdapter()
        voice_template = m3.generate_template(
            [str(preparation.original_path) for preparation in prepared_audio]
        )
        if voice_template is None:
            raise ValueError("Voice template generation failed.")

        return save_enrollment(
            user_id,
            username,
            face_template,
            voice_template,
            face_images_used=face_result.get("images_used", len(face_paths)),
            voice_samples_used=len(audio_paths),
            allow_update=allow_update,
        )
    finally:
        for preparation in prepared_audio:
            preparation.cleanup()


def run_authentication(face_image=None, audio_file=None, user_id=None, audio_preparation=None):
    if not user_id or not str(user_id).strip():
        raise ValueError("A User ID is required for authentication.")
    record = get_user(user_id)
    root = ENROLLMENTS_ROOT / record["user_id"]
    m2_template = root / record["face_template"]
    m3_template = root / record["voice_template"]
    if not m2_template.is_file() or not m3_template.is_file():
        raise FileNotFoundError("The selected user's face or voice template is missing.")

    face_path = None
    prepared_audio = None
    owns_prepared_audio = False
    m3_audio_path = None
    m4_audio_path = None
    audio_analysis = None
    try:
        face_path = validate_input_file(
            face_image,
            {".jpg", ".jpeg", ".png", ".bmp", ".webp"},
            "Face",
        ) if face_image else None
    except Exception as exc:
        face_error = str(exc)
    else:
        face_error = None
    if isinstance(audio_preparation, dict) and not audio_file:
        audio_analysis = audio_preparation
        audio_error = str(
            audio_preparation.get("message")
            or "Audio is unsuitable for verification. Please record again."
        )
    else:
        try:
            validated_audio = validate_input_file(
                audio_file,
                {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm"},
                "Audio",
            ) if audio_file else None
        except Exception as exc:
            audio_error = str(exc)
        else:
            if not validated_audio:
                audio_error = None
            elif isinstance(audio_preparation, PreparedAudio):
                prepared_audio = audio_preparation
                audio_analysis = prepared_audio.analysis
                audio_error = None
            elif isinstance(audio_preparation, dict):
                audio_analysis = audio_preparation
                audio_error = str(
                    audio_preparation.get("message")
                    or "Audio is unsuitable for verification. Please record again."
                )
            else:
                try:
                    prepared_audio = prepare_audio_for_models(validated_audio)
                    owns_prepared_audio = True
                    audio_analysis = prepared_audio.analysis
                    audio_error = None
                except AudioPreparationError as exc:
                    audio_analysis = exc.report
                    audio_error = str(exc)
            if prepared_audio:
                m3_audio_path = str(prepared_audio.original_path)
                m4_audio_path = str(prepared_audio.original_path)

    try:
        adapters, load_errors = {}, {}
        loaders = [("M1", lambda: _load_m1()), ("M2", lambda: _load_m2(str(m2_template) if m2_template else None)),
                   ("M3", lambda: _load_m3(str(m3_template) if m3_template else None)), ("M4", lambda: _load_m4())]
        for mid, loader in loaders:
            if mid in {"M3", "M4"} and not (m3_audio_path and m4_audio_path):
                continue
            try: adapters[mid] = loader()
            except Exception as exc: load_errors[mid] = exc

        specs = [
                ("M1","Face Anti-Spoofing",str(face_path) if face_path else None),
                ("M2","Face Recognition",str(face_path) if face_path else None),
                ("M3","Speaker Recognition",m3_audio_path),
                ("M4","Voice Anti-Spoofing",m4_audio_path)
        ]
        evidence=[]
        for mid,name,path in specs:
            if not path:
                validation_error = face_error if mid in {"M1", "M2"} else audio_error
                evidence.append({"module_id":mid,"module_name":name,"status":"not_run",
                    "decision":"UNKNOWN","verified":None,
                    "reason":validation_error or "Required input not provided"})
            elif mid in load_errors:
                exc=load_errors[mid]
                evidence.append({"module_id":mid,"module_name":name,"status":"error",
                    "decision":"UNKNOWN","verified":None,"error_type":type(exc).__name__,"error":str(exc)})
            else:
                evidence.append(safe_verify(adapters[mid],path))
        central = run_zero_trust_engine(evidence)
        return {
            "module_evidence": evidence,
            "central_result": central,
            "audio_analysis": audio_analysis,
        }
    finally:
        if owns_prepared_audio and prepared_audio:
            prepared_audio.cleanup()
