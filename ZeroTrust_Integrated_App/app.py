import html
import math
import os
import sys
import tempfile
import traceback
import wave
from pathlib import Path

import gradio as gr
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "integration"))

from authentication_pipeline import enroll_user, run_authentication
from enrollment_store import delete_user, list_users
from audio_enhancement import AudioPreparationError, PreparedAudio, prepare_audio_for_models
from voice_challenge import (
    VoiceChallengeUnavailable,
    create_voice_challenge,
    is_voice_challenge_valid,
    verify_voice_challenge,
)


def format_evidence(result):
    evidence = result.get("module_evidence", []) if isinstance(result, dict) else []
    rows = []
    for item in evidence:
        details = []
        for key in (
            "similarity",
            "confidence",
            "bonafide_probability",
            "spoof_probability",
            "threshold",
        ):
            value = item.get(key)
            if value is not None:
                details.append(f"{key}={float(value):.4f}")
        if item.get("prediction"):
            details.append(f"prediction={item['prediction']}")
        if item.get("reason"):
            details.append(str(item["reason"]))
        if item.get("error"):
            details.append(f"ERROR: {item['error']}")
        rows.append([
            item.get("module_id"),
            item.get("module_name"),
            item.get("status"),
            item.get("decision"),
            item.get("verified"),
            "; ".join(details) or "—",
        ])
    return rows


MODULE_DETAILS = {
    "M1": ("Face Anti-Spoofing", "Is this a live face?"),
    "M2": ("Face Recognition", "Does the face match the enrolled user?"),
    "M3": ("Speaker Recognition", "Does the voice match the enrolled user?"),
    "M4": ("Voice Anti-Spoofing", "Is the voice genuine, not replayed or synthetic?"),
}


def render_decision_card(decision="WAITING", reason="Provide biometrics to begin verification."):
    labels = {
        "ALLOW": ("ALLOW", "decision-allow"),
        "DENY": ("DENY", "decision-deny"),
        "REQUIRE_ADDITIONAL_VERIFICATION": (
            "REQUIRE ADDITIONAL VERIFICATION",
            "decision-additional",
        ),
        "PROCESSING": ("VERIFICATION IN PROGRESS", "decision-processing"),
        "WAITING": ("READY FOR VERIFICATION", "decision-waiting"),
    }
    label, tone = labels.get(
        decision,
        labels["REQUIRE_ADDITIONAL_VERIFICATION"],
    )
    return (
        f'<section class="decision-card {tone}" aria-live="polite">'
        '<div class="decision-overline">FINAL AUTHENTICATION DECISION</div>'
        f"<h2>{label}</h2>"
        
        "</section>"
    )


def render_status_banner(state, message):
    labels = {
        "ready": "READY",
        "processing": "PROCESSING",
        "complete": "RUN COMPLETE",
        "warning": "ATTENTION",
    }
    label = labels.get(state, "STATUS")
    return (
        f'<div class="status-banner status-{html.escape(state)}" role="status" aria-live="polite">'
        f'<span class="banner-indicator"></span>'
        f'<strong>{label}</strong>'
        f"<span>{html.escape(str(message))}</span>"
        "</div>"
    )


def render_audio_enhancement(report=None, purpose="verification"):
    report = report if isinstance(report, dict) else {}
    state = report.get("state", "waiting")
    if state not in {"waiting", "processing", "ready", "error"}:
        state = "error"

    def item_state(value):
        if value == "completed" or value == "passed" or value is True:
            return "complete"
        if value == "processing":
            return "processing"
        if value == "failed" or value is False:
            return "failed"
        return "waiting"

    speech_seconds = report.get("speech_seconds")
    if report.get("speech_detected") is True:
        speech_label = (
            f"Detected ({float(speech_seconds):.2f}s)"
            if speech_seconds is not None
            else "Detected"
        )
    elif report.get("speech_detected") is False:
        speech_label = "Not detected"
    else:
        speech_label = "Waiting"

    entries = [
        (
            "Raw audio captured",
            "captured" if report.get("raw_audio") == "captured" else "waiting",
            None,
        ),
        (
            "Noise cancellation (RNNoise)",
            item_state(report.get("noise_suppression")),
            None,
        ),
        (
            "Speech detected (Silero VAD)",
            item_state(report.get("speech_detected")),
            speech_label,
        ),
        (
            "Audio quality",
            item_state(report.get("audio_quality")),
            None,
        ),
        ("Multiple speakers / overlap", "unavailable", None),
        (
            "Ready for enrollment" if purpose == "enrollment" else "Ready for verification",
            item_state(report.get("ready")),
            None,
        ),
    ]
    labels = {
        "complete": "Complete",
        "processing": "Processing",
        "failed": "Needs new recording",
        "waiting": "Waiting",
        "unavailable": "Not assessed",
        "captured": "Received",
    }
    entry_html = "".join(
        '<div class="enhancement-item">'
        f'<span>{html.escape(label)}</span>'
        f'<strong class="enhancement-state {tone}">'
        f'{html.escape(detail or labels[tone])}</strong>'
        "</div>"
        for label, tone, detail in entries
    )
    message = report.get("message") or (
        "Noise suppression is enabled. Record at least 4 seconds of speech in a quiet place."
        if state == "waiting"
        else "Noise suppression and voice activity checks are running."
        if state == "processing"
        else "Multiple-speaker overlap is not reliably assessed; re-record if other voices are audible."
    )
    noise_label = (
        "Enabled - Processing"
        if report.get("noise_suppression") == "processing"
        else "Enabled - Completed"
        if report.get("noise_suppression") == "completed"
        else "Enabled - Failed"
        if report.get("noise_suppression") == "failed"
        else "Enabled - Not run"
        if report.get("noise_suppression") == "not_run"
        else "Enabled - Waiting"
        if state == "waiting"
        else "Enabled"
    )
    return (
        f'<section class="audio-enhancement-card {state}" role="status" aria-live="polite">'
        '<div class="enhancement-heading"><strong>Audio enhancement</strong>'
        f'<span class="noise-cancellation-indicator">{html.escape(noise_label)}</span></div>'
        f'<div class="enhancement-grid">{entry_html}</div>'
        '<p class="overlap-warning">Overlap check unavailable: RNNoise and VAD do not identify '
        "speaker count or overlapping speech. Record alone in a quiet environment.</p>"
        f'<p class="enhancement-message">{html.escape(str(message))}</p>'
        "</section>"
    )


def render_module_card(module_id, evidence=None, detail_override=None):
    name, question = MODULE_DETAILS[module_id]
    evidence = evidence if isinstance(evidence, dict) else {}
    status = str(evidence.get("status", "waiting")).lower()
    decision = str(evidence.get("decision", "UNKNOWN")).upper()
    diagnostic_only = bool(evidence.get("diagnostic_only"))
    if status == "completed" and decision == "ACCEPT":
        tone = "accepted"
    elif status == "completed" and decision == "REJECT":
        tone = "rejected"
    elif status == "error":
        tone = "error"
    elif decision == "REQUIRE_ADDITIONAL_VERIFICATION":
        tone = "additional"
    elif status in {"waiting", "not_run", "unavailable"}:
        tone = "waiting"
    else:
        tone = "processing" if status in {"processing", "in_progress"} else "additional"

    status_label = {
        "waiting": "WAITING",
        "not_run": "NOT RUN",
        "completed": "COMPLETED",
        "error": "ERROR",
        "unavailable": "UNAVAILABLE",
        "processing": "PROCESSING",
        "in_progress": "PROCESSING",
    }.get(status, status.replace("_", " ").upper())
    if (
        module_id == "M1"
        and status == "unavailable"
        and "face not detected" in str(evidence.get("reason", "")).lower()
    ):
        status_label = "FACE NOT DETECTED"
    if module_id == "M4" and status == "not_run":
        reason = str(evidence.get("reason", "")).lower()
        if any(
            indicator in reason
            for indicator in (
                "no speech detected",
                "too short",
                "heavily clipped",
                "could not be reliably detected",
            )
        ):
            status_label = "AUDIO QUALITY INSUFFICIENT"
    if diagnostic_only:
        status_label = "DIAGNOSTIC ONLY" if status == "completed" else "DIAGNOSTIC " + status_label
    if not evidence:
        status_label = "NOT TESTED" if module_id in {"M1", "M4"} else "WAITING"
        decision = ""

    metrics = []
    for key, label in (
        ("similarity", "Similarity"),
        ("confidence", "Confidence"),
        ("live_probability", "LIVE"),
        ("spoof_probability", "SPOOF"),
        ("face_detection_confidence", "Face detection"),
        ("bonafide_probability", "Bonafide"),
        ("spoof_probability", "Spoof"),
        ("threshold", "Threshold"),
    ):
        if module_id == "M4" and label == "SPOOF":
            continue
        if module_id == "M4" and key in {
            "bonafide_probability",
            "spoof_probability",
        }:
            label = (
                "Bonafide score (uncal.)"
                if key == "bonafide_probability"
                else "Spoof score (uncal.)"
            )
        value = evidence.get(key)
        if value is None:
            continue
        try:
            number = float(value)
            display = f"{number:.4f}" if math.isfinite(number) else str(value)
        except (TypeError, ValueError):
            display = str(value)
        metrics.append(
            f'<span class="metric"><span>{label}</span>'
            f"<strong>{html.escape(display)}</strong></span>"
        )
    metric_html = f'<div class="module-metrics">{"".join(metrics)}</div>' if metrics else ""

    detail = detail_override or evidence.get("error") or evidence.get("reason")
    if not detail:
        detail = "Awaiting biometric input." if not evidence else "No additional details."
    decision_html = ""
    if evidence and status not in {"processing", "in_progress", "not_run", "waiting"}:
        decision_label = decision
        if module_id == "M1":
            decision_label = {
                "LIVE": "LIVE DETECTED",
                "SPOOF": "SPOOF DETECTED",
            }.get(str(evidence.get("prediction", "")).upper(), decision)
        elif module_id == "M4":
            decision_label = {
                "BONAFIDE": "GENUINE DETECTED",
                "SPOOF": "SPOOF DETECTED",
            }.get(str(evidence.get("prediction", "")).upper(), decision)
        decision_html = f'<span class="decision-tag">{html.escape(decision_label)}</span>'
    module_detail = ""
    
    if evidence and status not in {"waiting"}:
        detail = detail_override or evidence.get("error") or evidence.get("reason")

      
       
        if detail:
            module_detail = f'<p class="module-detail">{html.escape(str(detail))}</p>'
    return (
        f'<article class="module-card {tone}">'
        '<div class="module-card-top">'
        f'<span class="module-id">{module_id}</span>'
        f'<div class="module-badges"><span class="module-status">{html.escape(status_label)}</span>'
        f"{decision_html}</div>"
        "</div>"
        f'<h3>{html.escape(name)}</h3>'
        f'<p class="module-question">{html.escape(question)}</p>'
        f"{metric_html}"
        f"{module_detail}"
        "</article>"
    )


def authentication_outputs(
    result=None,
    decision="REQUIRE_ADDITIONAL_VERIFICATION",
    reason="Authentication did not complete.",
    raw=None,
    audio_status="Voice processing result is unavailable.",
    audio_enhancement=None,
    processing=False,
):
    final = result.get("central_result", {}) if isinstance(result, dict) else {}
    final_decision = final.get("final_decision", decision)
    final_reason = final.get("reason", reason)
    evidence = result.get("module_evidence", []) if isinstance(result, dict) else []
    voice_challenge = (
        result.get("voice_challenge", {})
        if isinstance(result, dict)
        else {}
    )
    if isinstance(voice_challenge, dict) and voice_challenge.get("status") in {
        "failed",
        "expired",
    }:
        final_reason = (
            f"{final_reason} The fresh voice challenge "
            f"{voice_challenge['status']}; diagnostic M3/M4 results do not count. "
            "Record the current phrase and retry."
        )
    if processing:
        final_decision = "PROCESSING"
        final_reason = "Submitted biometrics are being checked by the verification modules."
    evidence_by_id = {
        item.get("module_id"): item
        for item in evidence
        if isinstance(item, dict) and item.get("module_id") in MODULE_DETAILS
    }
    diagnostic_voice_evidence = (
        result.get("diagnostic_voice_evidence", [])
        if isinstance(result, dict)
        else []
    )
    diagnostic_by_id = {
        item.get("module_id"): item
        for item in diagnostic_voice_evidence
        if isinstance(item, dict) and item.get("module_id") in {"M3", "M4"}
    }
    for module_id in MODULE_DETAILS:
        evidence_by_id.setdefault(
            module_id,
            {
                "status": "not_run",
                "decision": "UNKNOWN",
                "reason": reason if not isinstance(result, dict) else "No module evidence was returned.",
            },
        )
    for module_id, item in diagnostic_by_id.items():
        evidence_by_id[module_id] = {**item, "diagnostic_only": True}
    if processing:
        evidence_by_id = {
            module_id: {
                "status": "processing",
                "decision": "UNKNOWN",
                "reason": "The verification pipeline is running.",
            }
            for module_id in MODULE_DETAILS
        }
    card_html = [
        render_module_card(module_id, evidence_by_id.get(module_id))
        for module_id in MODULE_DETAILS
    ]
    evidence_rows = format_evidence(result) if isinstance(result, dict) else []
    for item in diagnostic_by_id.values():
        diagnostic_row = {**item, "module_name": f"{item.get('module_name', '')} (diagnostic only)"}
        evidence_rows.extend(format_evidence({"module_evidence": [diagnostic_row]}))
    raw_value = result if isinstance(result, dict) else raw
    if audio_enhancement is None and isinstance(result, dict):
        audio_enhancement = result.get("audio_analysis")
    if processing:
        raw_value = {"status": "processing"}
    if raw_value is None:
        raw_value = {"error": reason}
        if processing:
        banner = render_status_banner(
            "processing",
            "Verification is running. Module cards will show their actual outcomes when processing finishes.",
        )
    elif not isinstance(result, dict):
        banner = render_status_banner("warning", reason)
    else:
        # Removed the persistent incomplete-module warning banner.
        # Actual M1-M4 status remains visible in the module cards,
        # and the final decision/reason remains visible below.
        banner = ""
    return (
        banner,
        render_decision_card(final_decision, final_reason),
        *card_html,
        evidence_rows,
        raw_value,
        audio_status,
        render_audio_enhancement(audio_enhancement),
    )


def render_template_status(title, dimension, state="waiting", detail=None):
    labels = {
        "waiting": ("WAITING", "waiting"),
        "complete": ("GENERATED", "accepted"),
        "error": ("NOT COMPLETE", "error"),
    }
    label, tone = labels.get(state, labels["waiting"])
    detail = detail or "Generated after successful enrollment."
    return (
        f'<article class="template-status-card {tone}">'
        f'<span class="template-state">{label}</span>'
        f'<h3>{html.escape(title)}</h3>'
        f'<p class="template-dimension">{html.escape(dimension)}</p>'
        f'<p>{html.escape(detail)}</p>'
        "</article>"
    )


def audio_payload_path(payload):
    if payload is None:
        return None
    if (
        isinstance(payload, tuple)
        and len(payload) == 2
        and isinstance(payload[0], (int, np.integer))
    ):
        sample_rate, samples = payload
        data = np.asarray(samples)
        if data.size == 0:
            return None
        if data.dtype != np.int16:
            if np.issubdtype(data.dtype, np.floating):
                data = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
            else:
                data = np.clip(data, -32768, 32767).astype(np.int16)
        if data.ndim == 1:
            channels = 1
        elif data.ndim == 2:
            channels = data.shape[1]
        else:
            raise ValueError("Audio sample has an unsupported shape.")
        fd, output = tempfile.mkstemp(prefix="zt-audio-callback-", suffix=".wav")
        try:
            os.close(fd)
            with wave.open(output, "wb") as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(int(sample_rate))
                wav_file.writeframes(data.tobytes())
            return output
        except Exception:
            Path(output).unlink(missing_ok=True)
            raise
    if isinstance(payload, (str, Path)):
        value = str(payload).strip()
        return value or None
    if isinstance(payload, dict):
        for key in ("path", "name", "file"):
            value = payload.get(key)
            if isinstance(value, (str, Path)) and str(value).strip():
                return str(value)
    if hasattr(payload, "path"):
        value = payload.path
        if isinstance(value, (str, Path)) and str(value).strip():
            return str(value)
    if hasattr(payload, "name"):
        value = str(payload.name).strip()
        return value or None
    if isinstance(payload, (list, tuple)):
        for item in payload:
            value = audio_payload_path(item)
            if value:
                return value
    return None


def cleanup_audio_payload(path):
    if path and Path(path).name.startswith("zt-audio-callback-"):
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass


def audio_input_status(payload):
    path = audio_payload_path(payload)
    if not path:
        return "No voice sample has reached the app yet. Record and stop, or upload an audio file."

    audio_file = Path(path)
    try:
        if not audio_file.is_file():
            return "The selected voice sample is not available on the server. Please record again or re-upload."
        if audio_file.stat().st_size == 0:
            return "The received voice sample is empty. Please record again or choose another file."
        with wave.open(str(audio_file), "rb") as audio:
            duration = audio.getnframes() / float(audio.getframerate())
        if duration < 1:
            return f"Voice sample received, but it is only {duration:.1f} seconds. Record at least 1 second."
        if duration < 4:
            return (
                f"Voice sample received by the app ({duration:.1f} seconds), "
                "but at least 4 seconds are required."
            )
        return f"✅ Voice sample received by the app ({duration:.1f} seconds). Continue with enrollment or verification."
    except (wave.Error, ZeroDivisionError, OSError):
        size = audio_file.stat().st_size if audio_file.exists() else 0
        return (
            f"Voice file received by the app ({size // 1024} KB). "
            "Its audio format and duration will be checked during enhancement."
        )
    finally:
        cleanup_audio_payload(path)


def audio_capture_update(payload, purpose="verification"):
    status = audio_input_status(payload)
    received = "received" in status.lower()
    report = {
        "state": "waiting",
        "raw_audio": "captured" if received else "waiting",
        "noise_suppression": None,
        "speech_detected": None,
        "audio_quality": "not_assessed",
        "overlap_assessment": "unavailable",
        "ready": None,
        "message": (
            "Audio received. Start verification or enrollment to run enhancement and quality checks."
            if received
            else "Noise suppression is enabled and will run after a usable recording is provided."
        ),
    }
    return status, render_audio_enhancement(report, purpose=purpose)


def voice_processing_status(result):
    if not isinstance(result, dict):
        return "Voice processing result is unavailable."
    evidence = result.get("module_evidence", [])
    voice_modules = {
        item.get("module_id"): item
        for item in evidence
        if isinstance(item, dict) and item.get("module_id") in {"M3", "M4"}
    }
    diagnostic_by_id = {
        item.get("module_id"): item
        for item in result.get("diagnostic_voice_evidence", [])
        if isinstance(item, dict) and item.get("module_id") in {"M3", "M4"}
    }
    lines = ["**Voice module processing:**"]
    for module_id, name in (("M3", "Speaker Recognition"), ("M4", "Voice Anti-Spoofing")):
        diagnostic = diagnostic_by_id.get(module_id)
        if diagnostic:
            status = diagnostic.get("status", "unknown")
            decision = diagnostic.get("decision", "UNKNOWN")
            detail = diagnostic.get("error") or diagnostic.get("reason")
            lines.append(
                f"- {module_id} ({name}, diagnostic only): {status} / {decision} — not used in the final decision"
                + (f"; {detail}" if detail else "")
            )
            continue
        item = voice_modules.get(module_id)
        if not item:
            lines.append(f"- {module_id} ({name}): no evidence returned.")
            continue
        status = item.get("status", "unknown")
        decision = item.get("decision", "UNKNOWN")
        detail = item.get("error") or item.get("reason")
        lines.append(f"- {module_id} ({name}): {status} / {decision}" + (f" — {detail}" if detail else ""))
    challenge = result.get("voice_challenge")
    if isinstance(challenge, dict):
        lines.append(
            "- Fresh spoken challenge: "
            + str(challenge.get("status", "unknown")).replace("_", " ")
        )
        checked_phrase = challenge.get("checked_phrase")
        if checked_phrase:
            lines.append(f"- Phrase checked: `{html.escape(str(checked_phrase))}`")
        transcript = challenge.get("transcript")
        if transcript is not None:
            recognized = "".join(
                character
                for character in str(transcript)
                if character.isalnum() or character.isspace() or character in "'-"
            )
            recognized = " ".join(recognized.split())[:160]
            if recognized:
                lines.append(f"- Offline recognizer heard: `{recognized}`")
            else:
                lines.append("- Offline recognizer returned no words.")
    return "\n".join(lines)


def render_voice_challenge(challenge, state="ready", message=None):
    phrase = challenge.get("phrase", "") if isinstance(challenge, dict) else ""
    labels = {
        "ready": "READY",
        "processing": "CHECKING",
        "passed": "PASSED",
        "failed": "RETRY REQUIRED",
        "error": "UNAVAILABLE",
    }
    label = labels.get(state, "READY")
    message = message or (
        "Say all four words in order within 3 minutes. Checked offline; this blocks fixed old clips, not live voice conversion or replay of this phrase."
    )
    return (
        f'<section class="voice-challenge-card {html.escape(state)}" role="status" aria-live="polite">'
        '<div class="voice-challenge-heading"><strong>Fresh voice challenge</strong>'
        f'<span class="voice-challenge-state">{label}</span></div>'
        f'<p class="voice-challenge-phrase">{html.escape(str(phrase))}</p>'
        f'<p class="voice-challenge-message">{html.escape(str(message))}</p>'
        "</section>"
    )


def issue_voice_challenge():
    challenge = create_voice_challenge()
    return render_voice_challenge(challenge), challenge


def challenge_response(
    outputs,
    challenge,
    state="ready",
    message=None,
    clear_audio=False,
):
    if not is_voice_challenge_valid(challenge):
        challenge = create_voice_challenge()
        state = "ready"
        message = None
    return (
        *outputs,
        render_voice_challenge(challenge, state=state, message=message),
        challenge,
        gr.update(value=None) if clear_audio else gr.update(),
    )


def _voice_challenge_failure_result(
    face_path,
    user_id,
    prepared_audio,
    status,
    message,
    transcript=None,
    checked_phrase=None,
):
    report = dict(prepared_audio.analysis) if prepared_audio else {}
    report.update({
        "state": "error",
        "ready": False,
        "voice_challenge": status,
        "message": message,
    })
    result = run_authentication(
        face_path,
        None,
        user_id=str(user_id).strip(),
        audio_preparation=report,
    )
    result["voice_challenge"] = {"status": status}
    if checked_phrase:
        result["voice_challenge"]["checked_phrase"] = str(checked_phrase)
    if transcript is not None:
        result["voice_challenge"]["transcript"] = str(transcript)[:160]
    return result, report


def _run_voice_diagnostics(user_id, prepared_audio):
    diagnostic_result = run_authentication(
        None,
        str(prepared_audio.original_path),
        user_id=str(user_id).strip(),
        audio_preparation=prepared_audio,
    )
    return [
        {**item, "diagnostic_only": True}
        for item in diagnostic_result.get("module_evidence", [])
        if isinstance(item, dict) and item.get("module_id") in {"M3", "M4"}
    ]


def authenticate(user_id, face_path, audio_path, challenge_state=None):
    if not user_id or not str(user_id).strip():
        yield challenge_response(
            authentication_outputs(
                reason="Enter an enrolled User ID before starting verification.",
                raw={"error": "User ID is required"},
                audio_status=audio_input_status(audio_path),
            ),
            challenge_state,
        )
        return
    if not face_path:
        yield challenge_response(
            authentication_outputs(
                reason="Provide a face image before starting verification.",
                raw={"error": "Face image is missing"},
                audio_status=audio_input_status(audio_path),
            ),
            challenge_state,
        )
        return
    if not is_voice_challenge_valid(challenge_state):
        fresh_challenge = create_voice_challenge()
        yield challenge_response(
            authentication_outputs(
                reason="The voice challenge expired or is missing. Say the new phrase and try again.",
                raw={"error": "Voice challenge expired or missing"},
                audio_status="No voice challenge was checked. Record the new phrase before authenticating.",
            ),
            fresh_challenge,
            clear_audio=True,
        )
        return

    sample_path = None
    prepared_audio = None
    try:
        sample_path = audio_payload_path(audio_path)
        if not sample_path:
            yield challenge_response(
                authentication_outputs(
                    reason="Provide a voice sample before starting verification.",
                    raw={"error": "Audio sample is missing"},
                    audio_status=audio_input_status(None),
                ),
                challenge_state,
            )
            return

        processing_report = {
            "state": "processing",
            "raw_audio": "captured",
            "noise_suppression": "processing",
            "speech_detected": None,
            "audio_quality": "processing",
            "overlap_assessment": "unavailable",
            "ready": None,
            "message": "Noise suppression and speech-quality checks are running.",
        }
        yield challenge_response(
            authentication_outputs(
                processing=True,
                raw={"status": "processing"},
                audio_status="Raw voice sample received. Noise suppression, VAD, and the spoken challenge are processing.",
                audio_enhancement=processing_report,
            ),
            challenge_state,
            state="processing",
            message=(
                "Checking the phrase locally. If it does not match, M3/M4 may run "
                "for diagnostics only and cannot authorize access."
            ),
        )
        try:
            prepared_audio = prepare_audio_for_models(sample_path)
        except AudioPreparationError as exc:
            result = run_authentication(
                face_path,
                None,
                user_id=str(user_id).strip(),
                audio_preparation=exc.report,
            )
            result["voice_challenge"] = {"status": "not_checked"}
            next_challenge = create_voice_challenge()
            yield challenge_response(
                authentication_outputs(
                    result=result,
                    audio_status=voice_processing_status(result),
                    audio_enhancement=exc.report,
                ),
                next_challenge,
                state="failed",
                message="The audio was unsuitable. Use this new phrase with a clear recording.",
                clear_audio=True,
            )
            return

        if not is_voice_challenge_valid(challenge_state):
            challenge_result = {"matched": False}
            challenge_message = "The voice challenge expired. Record again using the new phrase."
            challenge_status = "expired"
        else:
            challenge_result = verify_voice_challenge(
                prepared_audio.original_path,
                challenge_state["phrase"],
            )
            challenge_message = (
                "The spoken phrase did not match. Diagnostic M3/M4 results are separate "
                "and cannot affect the final decision."
            )
            challenge_status = "failed"

        if not challenge_result.get("matched"):
            checked_phrase = (
                challenge_state.get("phrase")
                if isinstance(challenge_state, dict)
                else None
            )
            result, report = _voice_challenge_failure_result(
                face_path,
                user_id,
                prepared_audio,
                challenge_status,
                challenge_message,
                transcript=challenge_result.get("transcript"),
                checked_phrase=checked_phrase,
            )
            if challenge_status == "failed":
                result["diagnostic_voice_evidence"] = _run_voice_diagnostics(
                    user_id,
                    prepared_audio,
                )
            next_challenge = create_voice_challenge()
            if challenge_status == "failed" and checked_phrase:
                transcript = " ".join(
                    str(challenge_result.get("transcript", "")).split()
                )[:120]
                recognized_text = (
                    f'The recognizer heard "{transcript}". '
                    if transcript
                    else "The recognizer returned no words. "
                )
                retry_message = (
                    f'The challenge was "{checked_phrase}", but {recognized_text}'
                    "M3/M4 results are diagnostic only. The previous clip was cleared; "
                    "record the current phrase exactly, in order."
                )
            else:
                retry_message = (
                    "The previous phrase expired. The clip was cleared; "
                    "record this new phrase."
                )
            yield challenge_response(
                authentication_outputs(
                    result=result,
                    audio_status=voice_processing_status(result),
                    audio_enhancement=report,
                ),
                next_challenge,
                state="failed",
                message=retry_message,
                clear_audio=True,
            )
            return

        result = run_authentication(
            face_path,
            sample_path,
            user_id=str(user_id).strip(),
            audio_preparation=prepared_audio,
        )
        result["voice_challenge"] = {"status": "passed"}
        next_challenge = create_voice_challenge()
        yield challenge_response(
            authentication_outputs(
                result=result,
                audio_status=voice_processing_status(result),
                audio_enhancement=prepared_audio.analysis,
            ),
            next_challenge,
            state="passed",
            message="Phrase verified locally. A new phrase is ready for the next authentication attempt.",
            clear_audio=True,
        )
    except VoiceChallengeUnavailable as exc:
        result, report = _voice_challenge_failure_result(
            face_path,
            user_id,
            prepared_audio,
            "unavailable",
            str(exc),
        )
        next_challenge = create_voice_challenge()
        yield challenge_response(
            authentication_outputs(
                result=result,
                audio_status=voice_processing_status(result),
                audio_enhancement=report,
            ),
            next_challenge,
            state="error",
            message=str(exc),
        )
    except Exception as exc:
        error = {
            "exception": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        next_challenge = create_voice_challenge()
        yield challenge_response(
            authentication_outputs(
                reason="The authentication request could not be processed safely.",
                raw=error,
                audio_status=f"Audio was attached, but processing failed: {type(exc).__name__}: {exc}",
                audio_enhancement=(
                    prepared_audio.analysis
                    if prepared_audio
                    else {
                        "state": "error",
                        "raw_audio": "captured",
                        "noise_suppression": "failed",
                        "audio_quality": "not_assessed",
                        "ready": None,
                        "message": "Audio enhancement failed. Authentication cannot continue reliably.",
                    }
                ),
            ),
            next_challenge,
            state="error",
            message=f"Verification could not continue: {type(exc).__name__}. A new phrase is ready.",
        )
    finally:
        if prepared_audio:
            prepared_audio.cleanup()
        cleanup_audio_payload(sample_path)


def normalize_files(files):
    if not files:
        return []
    if isinstance(files, (str, Path)):
        return [str(files)]
    paths = []
    for item in files:
        if isinstance(item, str):
            paths.append(item)
        elif isinstance(item, Path):
            paths.append(str(item))
        elif isinstance(item, dict) and item.get("path"):
            paths.append(str(item["path"]))
        elif hasattr(item, "name"):
            paths.append(str(item.name))
    return paths


def enroll(user_id, username, face_files, audio_files, allow_update):
    audio_reports = []
    try:
        face_paths = normalize_files(face_files)
        audio_paths = normalize_files(audio_files)
        record = enroll_user(
            user_id,
            username,
            face_paths,
            audio_paths,
            allow_update=bool(allow_update),
            audio_reports=audio_reports,
        )
        return (
            (
                '<section class="enrollment-result success">'
                '<span class="result-overline">ENROLLMENT COMPLETE</span>'
                f"<h2>{html.escape(record['username'])}</h2>"
                f'<p>User ID <strong>{html.escape(record["user_id"])}</strong> is enrolled. '
                "Both biometric templates were generated and saved.</p>"
                "</section>"
            ),
            render_audio_enhancement(
                audio_reports[-1] if audio_reports else None,
                purpose="enrollment",
            ),
            render_template_status(
                "Face template",
                "512-dimensional identity embedding",
                state="complete",
                detail="Generated and saved for this enrolled user.",
            ),
            render_template_status(
                "Voice template",
                "192-dimensional speaker embedding",
                state="complete",
                detail="Generated and saved for this enrolled user.",
            ),
            record,
            refresh_user_choices(),
            refresh_users(),
        )
    except Exception as exc:
        error = {
            "exception": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        return (
            (
                '<section class="enrollment-result failure">'
                '<span class="result-overline">ENROLLMENT INCOMPLETE</span>'
                "<h2>Could not enroll this user</h2>"
                f"<p>{html.escape(str(exc))}</p>"
                "</section>"
            ),
            render_audio_enhancement(
                audio_reports[-1] if audio_reports else {
                    "state": "error",
                    "noise_suppression": "not_run",
                    "audio_quality": "not_assessed",
                    "ready": False,
                    "message": str(exc),
                },
                purpose="enrollment",
            ),
            render_template_status(
                "Face template",
                "512-dimensional identity embedding",
                state="error",
                detail="Enrollment was not completed. Review the failure details.",
            ),
            render_template_status(
                "Voice template",
                "192-dimensional speaker embedding",
                state="error",
                detail="Enrollment was not completed. Review the failure details.",
            ),
            error,
            refresh_user_choices(),
            refresh_users(),
        )


def refresh_users():
    users = list_users()
    return [
        [user["user_id"], user["username"], user["status"], user["enrolled_at"], user["updated_at"]]
        for user in users
    ]


def refresh_user_choices():
    return gr.update(choices=[user["user_id"] for user in list_users()])


def delete_selected_user(user_id):
    try:
        record = delete_user(str(user_id).strip())
        return f"## 🗑️ Deleted `{record['user_id']}`", refresh_users(), refresh_user_choices()
    except Exception as exc:
        return f"## ⛔ Delete Failed\n\n{exc}", refresh_users(), refresh_user_choices()


APP_CSS = """
:root {
  --zt-ink: #17243b;
  --zt-muted: #61718a;
  --zt-blue: #2563eb;
  --zt-border: #e2e9f2;
  --zt-canvas: #f4f7fb;
}
body, .gradio-container { background: var(--zt-canvas) !important; color: var(--zt-ink); }
.gradio-container { margin: 0 auto !important; max-width: 1200px !important; padding: 16px 22px 30px !important; }
.gradio-container, button, input, textarea { font-family: Inter, "Segoe UI", Arial, sans-serif !important; }
#app-header { margin: 0 0 12px; }
.zt-header {
  align-items: center; background: #fff; border: 1px solid var(--zt-border);
  border-radius: 13px; box-shadow: 0 3px 12px rgba(27, 52, 89, .045);
  display: flex; justify-content: space-between; padding: 11px 15px;
}
.brand-lockup { align-items: center; display: flex; gap: 12px; }
.brand-mark {
  align-items: center; background: #eaf1ff; border-radius: 10px; color: var(--zt-blue);
  display: flex; font-size: 13px; font-weight: 800; height: 34px; justify-content: center; width: 34px;
}
.brand-name { color: var(--zt-ink); font-size: 14px; font-weight: 750; }
.brand-caption { color: #74839a; font-size: 9px; font-weight: 750; letter-spacing: .1em; }
.header-tag {
  background: #f0f5ff; border: 1px solid #dce8ff; border-radius: 999px;
  color: #315da8; font-size: 10px; font-weight: 650; padding: 6px 9px;
}
.main-layout {
  align-items: start; display: grid !important; gap: 16px;
  grid-template-columns: minmax(390px, .92fr) minmax(490px, 1.08fr);
}
.main-layout > div { min-width: 0 !important; width: auto !important; }
.panel {
  background: #fff !important; border: 1px solid var(--zt-border) !important;
  border-radius: 14px !important; box-shadow: 0 4px 14px rgba(27, 52, 89, .045) !important;
  padding: 16px !important;
}
.panel-heading { align-items: center; display: flex; gap: 10px; margin: 0 0 12px; }
.section-number {
  align-items: center; background: #eaf1ff; border-radius: 50%; color: #2456a6;
  display: inline-flex; flex: 0 0 26px; font-size: 11px; font-weight: 800; height: 26px; justify-content: center;
}
.panel-heading h2 { color: var(--zt-ink); font-size: 17px; letter-spacing: -.02em; margin: 0; }
.panel-heading p { color: var(--zt-muted); font-size: 12px; margin: 3px 0 0; }
.voice-heading { margin-top: 12px; }
.status-banner {
  align-items: center; background: #edf4ff; border: 1px solid #d9e7ff; border-radius: 10px;
  color: #344f77; display: flex; font-size: 12px; gap: 8px; margin: 0 0 13px; padding: 9px 12px;
}
.status-banner strong { color: #285db4; font-size: 10px; letter-spacing: .06em; white-space: nowrap; }
.banner-indicator { background: #4a83df; border-radius: 50%; flex: 0 0 7px; height: 7px; }
.status-processing { background: #edf4ff; border-color: #d9e7ff; }
.status-complete { background: #edf8f2; border-color: #d4ecdd; color: #315c46; }
.status-complete strong { color: #16784e; }
.status-complete .banner-indicator { background: #16a36a; }
.status-warning { background: #fff7e8; border-color: #f3e4c4; color: #765a26; }
.status-warning strong { color: #94620a; }
.status-warning .banner-indicator { background: #d99119; }
.section-label { color: #738198; font-size: 10px; font-weight: 750; letter-spacing: .09em; margin: 13px 0 5px; }
.face-preview .image-container {
  aspect-ratio: 4 / 3; background: #10233b !important; border: 1px solid #263d5a;
  border-radius: 12px; height: 340px !important; max-height: 350px; min-height: 300px;
  overflow: hidden; position: relative;
}
.face-preview .image-container .wrap { background: transparent !important; }
.face-preview .image-container video, .face-preview .image-container img:not(.hide) {
  height: 100% !important; object-fit: contain; width: 100% !important;
}
.face-preview .image-container [title="grant webcam access"] button {
  background: rgba(255, 255, 255, .04) !important;
  border: 1px solid rgba(220, 232, 247, .35) !important;
  color: #eef5ff !important;
}
.face-preview .image-container [title="grant webcam access"] button .wrap { font-size: 0; }
.face-preview .image-container [title="grant webcam access"] button .wrap::after {
  content: "Start Camera"; font-size: 13px;
}
.face-preview .source-selection button[aria-label="Upload file"]::after {
  content: "Upload"; font-size: 10px; margin-left: 4px;
}
.face-preview .source-selection button[aria-label="Capture from camera"]::after {
  content: "Camera"; font-size: 10px; margin-left: 4px;
}
.face-preview button[aria-label="Remove Image"]::after {
  content: "Retake"; font-size: 10px; margin-left: 4px;
}
.camera-tip { color: var(--zt-muted); font-size: 10px; margin: 3px 0 8px; }
.voice-heading { margin-top: 12px; }
.voice-heading p { color: var(--zt-muted); font-size: 11px; margin: -3px 0 5px; }
.face-preview .image-container::before {
  bottom: 13px; color: #d7e3f2; content: "No image captured yet"; font-size: 11px;
  left: 0; pointer-events: none; position: absolute; text-align: center; width: 100%; z-index: 3;
}
.face-preview .image-container::after {
  border: 2px dashed rgba(215, 227, 242, .78); border-radius: 50%; content: "";
  height: 58%; left: 50%; pointer-events: none; position: absolute; top: 46%;
  transform: translate(-50%, -50%); width: 38%; z-index: 2;
}
.face-preview .image-container:has(img:not(.hide), video:not(.hide))::before,
.face-preview .image-container:has(img:not(.hide), video:not(.hide))::after { display: none; }
.face-preview .source-selection { margin-top: 5px; }
.module-stack { gap: 7px !important; }
.module-card, .template-status-card {
  background: #fff; border: 1px solid var(--zt-border); border-radius: 11px;
  box-shadow: 0 2px 7px rgba(27, 52, 89, .025); padding: 10px 12px;
}
.module-card-top { align-items: center; display: flex; justify-content: space-between; }
.module-id {
  align-items: center; background: #eff4ff; border-radius: 7px; color: #285db4;
  display: inline-flex; font-size: 10px; font-weight: 800; height: 22px; justify-content: center; width: 31px;
}
.module-badges { align-items: center; display: flex; gap: 5px; }
.module-status, .decision-tag, .template-state {
  background: #f1f4f8; border-radius: 999px; color: #64748b; font-size: 8px;
  font-weight: 800; letter-spacing: .045em; padding: 4px 7px;
}
.module-card h3 { color: var(--zt-ink); font-size: 13px; margin: 6px 0 2px; }
.module-question { color: var(--zt-muted); font-size: 12px; line-height: 1.35; margin: 0; }
.module-metrics { display: flex; flex-wrap: wrap; gap: 5px; margin: 6px 0 0; }
.metric {
  background: #f7f9fc; border: 1px solid #edf1f6; border-radius: 8px;
  display: inline-flex; flex-direction: column; min-width: 70px; padding: 4px 6px;
}
.metric span { color: #78879b; font-size: 9px; }
.metric strong { color: var(--zt-ink); font-size: 11px; margin-top: 2px; }
.module-detail { color: #738198; font-size: 10px; line-height: 1.35; margin: 5px 0 0; overflow-wrap: anywhere; }
.module-card.accepted, .template-status-card.accepted { border-top: 3px solid #16a36a; }
.module-card.accepted .module-status, .module-card.accepted .decision-tag,
.template-status-card.accepted .template-state { background: #e8f7ef; color: #16784e; }
.module-card.rejected { border-top: 3px solid #dc4c4c; }
.module-card.rejected .module-status, .module-card.rejected .decision-tag { background: #fff0ef; color: #b83232; }
.module-card.error, .template-status-card.error { border-top: 3px solid #e08b28; }
.module-card.error .module-status, .template-status-card.error .template-state { background: #fff4e5; color: #a45b0b; }
.module-card.additional { border-top: 3px solid #e0a126; }
.module-card.additional .module-status, .module-card.additional .decision-tag { background: #fff6dd; color: #93620b; }
.module-card.processing { border-top: 3px solid var(--zt-blue); }
.module-card.processing .module-status { background: #eaf1ff; color: #2456a6; }
.decision-card {
  border: 1px solid transparent; border-radius: 11px; margin: 10px 0 0; padding: 11px 13px;
}
.decision-overline, .result-overline { color: #74839a; font-size: 9px; font-weight: 750; letter-spacing: .1em; }
.decision-card h2 { font-size: 16px; letter-spacing: -.02em; margin: 4px 0; }
.decision-card p { color: #506078; font-size: 11px; line-height: 1.4; margin: 0; }
.decision-allow { background: #edf9f2; border-color: #ccebd8; }
.decision-allow h2 { color: #16784e; }
.decision-deny { background: #fff0ef; border-color: #f3d0ce; }
.decision-deny h2 { color: #b83232; }
.decision-additional { background: #fff7e6; border-color: #f4e3b7; }
.decision-additional h2 { color: #94620a; }
.decision-waiting { background: #f3f6fa; border-color: #e3eaf3; }
.decision-waiting h2 { color: #4a5a72; }
.decision-processing { background: #edf4ff; border-color: #d9e7ff; }
.decision-processing h2 { color: #2456a6; }
.decision-allow .decision-overline { color: #287c55; }
.decision-deny .decision-overline { color: #a64640; }
.decision-additional .decision-overline { color: #966814; }
.enrollment-result {
  border: 1px solid var(--zt-border); border-radius: 14px; margin-bottom: 14px; padding: 18px;
}
.enrollment-result h2 { font-size: 19px; margin: 6px 0; }
.enrollment-result p { color: #506078; font-size: 13px; line-height: 1.5; margin: 0; }
.enrollment-result.success { background: #edf9f2; border-color: #ccebd8; }
.enrollment-result.success h2 { color: #16784e; }
.enrollment-result.failure { background: #fff4e5; border-color: #f2dfbd; }
.enrollment-result.failure h2 { color: #a45b0b; }
.template-status-card { min-height: 110px; }
.template-dimension { color: #45618c !important; font-size: 11px !important; font-weight: 650; }
.template-status-card p { color: var(--zt-muted); font-size: 12px; line-height: 1.45; margin: 6px 0 0; }
.audio-guidance { background: #f3f7ff; border: 1px solid #e0eaff; border-radius: 9px; color: #526782; font-size: 11px; line-height: 1.4; padding: 8px 10px; }
.voice-challenge-card { background: #f3f7ff; border: 1px solid #dce8fb; border-radius: 10px; margin: 8px 0; padding: 10px 12px; }
.voice-challenge-heading { align-items: center; display: flex; justify-content: space-between; }
.voice-challenge-heading strong { color: var(--zt-ink); font-size: 12px; }
.voice-challenge-state { background: #eaf1ff; border-radius: 999px; color: #285db4; font-size: 9px; font-weight: 800; padding: 4px 7px; }
.voice-challenge-phrase { color: #173d79; font-size: 17px; font-weight: 750; letter-spacing: .04em; margin: 7px 0 3px; text-transform: capitalize; }
.voice-challenge-message { color: #526782; font-size: 10px; line-height: 1.4; margin: 0; }
.voice-challenge-card.passed .voice-challenge-state { background: #e8f7ef; color: #16784e; }
.voice-challenge-card.failed .voice-challenge-state, .voice-challenge-card.error .voice-challenge-state { background: #fff0ef; color: #b83232; }
.audio-enhancement-card { background: #fbfdff; border: 1px solid var(--zt-border); border-radius: 10px; margin: 8px 0; padding: 10px; }
.enhancement-heading { align-items: center; display: flex; flex-wrap: wrap; gap: 8px; justify-content: space-between; }
.enhancement-heading > strong { color: var(--zt-ink); font-size: 13px; }
.noise-cancellation-indicator { background: #edf9f2; border: 1px solid #ccebd8; border-radius: 999px; color: #16784e; font-size: 10px; font-weight: 700; padding: 3px 8px; }
.audio-enhancement-card.error .noise-cancellation-indicator { background: #fff4e5; border-color: #f2dfbd; color: #a45b0b; }
.enhancement-grid { display: grid; gap: 6px; grid-template-columns: repeat(2, minmax(0, 1fr)); margin-top: 8px; }
.enhancement-item { align-items: center; background: #fff; border: 1px solid #e7edf5; border-radius: 7px; display: flex; gap: 8px; justify-content: space-between; min-width: 0; padding: 6px 8px; }
.enhancement-item > span { color: #52647d; font-size: 10px; line-height: 1.35; }
.enhancement-state { color: #718096; flex: 0 0 auto; font-size: 9px; text-align: right; }
.enhancement-state.complete { color: #16784e; }
.enhancement-state.processing { color: #2456a6; }
.enhancement-state.failed { color: #a45b0b; }
.overlap-warning { background: #fff8e9; border-radius: 6px; color: #7a5a18; font-size: 10px; line-height: 1.4; margin: 7px 0 0; padding: 6px 8px; }
.enhancement-message { color: #52647d; font-size: 10px; line-height: 1.4; margin: 7px 0 0; }
.privacy-note { border-top: 1px solid var(--zt-border); color: #738198; font-size: 10px; margin-top: 16px; padding-top: 10px; }
.primary-action button { min-height: 40px; }
button.primary {
  background: var(--zt-blue) !important; border: 1px solid var(--zt-blue) !important;
  border-radius: 9px !important; box-shadow: 0 4px 10px rgba(37, 99, 235, .15);
  font-weight: 700 !important;
}
button.primary:hover { background: #1d4ed8 !important; }
input, textarea, [role="combobox"] { border-color: #dce4ef !important; border-radius: 8px !important; }
[role="tablist"] {
  background: transparent !important; border: 0 !important; border-radius: 0 !important;
  gap: 5px; margin: 0 0 10px; padding: 0 !important;
}
[role="tab"] { border: 0 !important; border-radius: 8px !important; color: #60718a !important; font-size: 12px !important; font-weight: 650 !important; }
[role="tab"][aria-selected="true"] { background: #eaf1ff !important; color: #2257aa !important; }
@media (max-width: 980px) {
  .main-layout { grid-template-columns: minmax(0, 1fr) !important; }
}
@media (max-width: 650px) {
  .gradio-container { padding: 12px 10px 24px !important; }
  .panel { padding: 12px !important; }
  .zt-header { padding: 9px 11px; }
  .status-banner { align-items: flex-start; flex-wrap: wrap; }
  .face-preview .image-container { min-height: 250px; }
  .enhancement-grid { grid-template-columns: minmax(0, 1fr); }
}
"""

APP_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="blue",
    neutral_hue="slate",
).set(
    body_background_fill="#f4f7fb",
    body_background_fill_dark="#f4f7fb",
    body_text_color="#17243b",
    body_text_color_dark="#17243b",
    body_text_color_subdued="#61718a",
    body_text_color_subdued_dark="#61718a",
    background_fill_primary="#f4f7fb",
    background_fill_primary_dark="#f4f7fb",
    background_fill_secondary="#ffffff",
    background_fill_secondary_dark="#ffffff",
    border_color_primary="#e2e9f2",
    border_color_primary_dark="#e2e9f2",
    block_background_fill="#ffffff",
    block_background_fill_dark="#ffffff",
    block_border_color="#e2e9f2",
    block_border_color_dark="#e2e9f2",
    block_label_background_fill="#eff4ff",
    block_label_background_fill_dark="#eff4ff",
    block_label_text_color="#52647d",
    block_label_text_color_dark="#52647d",
    block_title_background_fill="#ffffff",
    block_title_background_fill_dark="#ffffff",
    block_title_text_color="#17243b",
    block_title_text_color_dark="#17243b",
    panel_background_fill="#ffffff",
    panel_background_fill_dark="#ffffff",
    panel_border_color="#e2e9f2",
    panel_border_color_dark="#e2e9f2",
    input_background_fill="#ffffff",
    input_background_fill_dark="#ffffff",
    input_border_color="#dce4ef",
    input_border_color_dark="#dce4ef",
    input_placeholder_color="#8997aa",
    input_placeholder_color_dark="#8997aa",
    button_secondary_background_fill="#ffffff",
    button_secondary_background_fill_dark="#ffffff",
    button_secondary_background_fill_hover="#f4f7fb",
    button_secondary_background_fill_hover_dark="#f4f7fb",
    button_secondary_text_color="#41536d",
    button_secondary_text_color_dark="#41536d",
    button_cancel_background_fill="#f4f7fb",
    button_cancel_background_fill_dark="#f4f7fb",
    button_cancel_text_color="#41536d",
    button_cancel_text_color_dark="#41536d",
    table_even_background_fill="#ffffff",
    table_even_background_fill_dark="#ffffff",
    table_odd_background_fill="#f8fafc",
    table_odd_background_fill_dark="#f8fafc",
    table_text_color="#17243b",
    table_text_color_dark="#17243b",
)


with gr.Blocks(
    title="ZeroTrust Biometric Authentication",
    theme=APP_THEME,
    css=APP_CSS,
) as demo:
    gr.HTML(
        """
        <header class="zt-header">
          <div class="brand-lockup">
            <div class="brand-mark">ZT</div>
            <div class="brand-name">Zero Trust Biometric Authentication</div>
          </div>
        </header>
        """,
        elem_id="app-header",
    )

    with gr.Tabs():
        with gr.Tab("Authentication"):
            auth_system_banner = gr.HTML(
                value=render_status_banner(
                    "ready",
                    "Provide an enrolled user, face image, and voice sample to run M1-M4.",
                ),
                elem_id="auth-status-banner",
            )
            with gr.Row(elem_classes=["main-layout"]):
                with gr.Column(scale=1, min_width=0):
                    with gr.Group(elem_classes=["panel", "capture-card"]):
                        gr.HTML(
                            '<div class="panel-heading"><span class="section-number">1</span><div>'
                            "<h2>Face</h2><p>Start the camera or upload an image.</p></div></div>"
                        )
                        auth_face = gr.Image(
                            label="Face camera preview",
                            type="filepath",
                            sources=["webcam", "upload"],
                            height=340,
                            elem_id="auth-face-capture",
                            elem_classes=["face-preview"],
                        )
                        gr.HTML(
                            '<p class="camera-tip">Use Start Camera, Capture, or Upload. Clear/Retake to try again.</p>'
                        )
                        with gr.Row():
                            auth_user_id = gr.Dropdown(
                                label="Enrolled User ID",
                                choices=[user["user_id"] for user in list_users()],
                                allow_custom_value=True,
                                scale=4,
                            )
                            auth_user_id_refresh = gr.Button("Refresh", size="sm", scale=1)
                        gr.HTML(
                            '<div class="voice-heading"><div class="section-label">VOICE SAMPLE</div>'
                            '<p>Record or upload a voice sample.</p></div>'
                        )
                        auth_voice_challenge_state = gr.State()
                        auth_voice_challenge_display = gr.HTML(
                            value=render_voice_challenge(None),
                            elem_classes=["voice-challenge"],
                        )
                        auth_voice_challenge_refresh = gr.Button("New phrase", size="sm")
                        auth_audio = gr.Audio(
                            label="Record or upload voice sample",
                            type="numpy",
                            sources=["microphone", "upload"],
                            format="wav",
                            min_length=1,
                        )
                        auth_audio_status = gr.Markdown(
                            "No voice sample has reached the app yet. Record and stop, or upload an audio file."
                        )
                        auth_audio_enhancement_status = gr.HTML(
                            value=render_audio_enhancement(),
                            elem_classes=["audio-enhancement"],
                        )
                        gr.HTML(
                            '<div class="audio-guidance">If the recording remains at 0:00 in the VS Code preview, '
                            "use Chrome/Edge or upload a file. Speak the displayed challenge words into the microphone; "
                            "record at least 4 seconds and ensure no other voices are audible.</div>"
                        )
                        auth_button = gr.Button(
                            "Authenticate",
                            variant="primary",
                            elem_classes=["primary-action"],
                        )

                with gr.Column(scale=1, min_width=0):
                    with gr.Group(elem_classes=["panel", "progress-card"]):
                        gr.HTML(
                            '<div class="panel-heading"><span class="section-number">2</span><div>'
                            "<h2>Verification progress</h2><p>Each status reflects the module's actual result.</p></div></div>"
                        )
                        with gr.Column(elem_classes=["module-stack"]):
                            auth_module_cards = [
                                gr.HTML(value=render_module_card("M1")),
                                gr.HTML(value=render_module_card("M2")),
                                gr.HTML(value=render_module_card("M3")),
                                gr.HTML(value=render_module_card("M4")),
                            ]
                        auth_decision = gr.HTML(
                            value=render_decision_card(),
                            elem_id="decision-result",
                        )
                        with gr.Accordion("Technical evidence details", open=False):
                            auth_results = gr.Dataframe(
                                headers=["Module", "Module Name", "Status", "Decision", "Verified", "Details"],
                                datatype=["str", "str", "str", "str", "str", "str"],
                                label="Module evidence",
                                interactive=False,
                                wrap=True,
                            )
                            auth_raw = gr.JSON(label="Full evidence and central-engine output")

        with gr.Tab("New User Enrollment"):
            with gr.Row():
                with gr.Column(scale=6):
                    with gr.Group(elem_classes=["panel", "capture-card"]):
                        gr.HTML(
                            '<div class="panel-heading"><h2>Register a new identity</h2>'
                            "<p>Raw face and voice samples are used for template generation, not retained as enrollment records.</p></div>"
                        )
                        with gr.Row():
                            enroll_user_id = gr.Textbox(label="User ID", placeholder="USER001")
                            enroll_name = gr.Textbox(label="Name", placeholder="Full name")
                        gr.HTML('<div class="section-label">FACE ENROLLMENT</div>')
                        enroll_faces = gr.Image(
                            label="Capture or upload face image",
                            type="filepath",
                            sources=["webcam", "upload"],
                            height=230,
                        )
                        gr.HTML('<div class="section-label">VOICE ENROLLMENT</div>')
                        enroll_audio = gr.Audio(
                            label="Record or upload voice sample",
                            type="filepath",
                            sources=["microphone", "upload"],
                            format="wav",
                            min_length=1,
                        )
                        enroll_audio_status = gr.Markdown(
                            "No voice sample has reached the app yet. Record and stop, or upload an audio file."
                        )
                        enroll_audio_enhancement_status = gr.HTML(
                            value=render_audio_enhancement(purpose="enrollment"),
                            elem_classes=["audio-enhancement"],
                        )
                        gr.HTML(
                            '<div class="audio-guidance">If recording stays at 0:00 in the VS Code preview, '
                            "try Chrome or Edge, or upload a WAV, M4A, MP3, FLAC, OGG, AAC, or WebM file. "
                            "Record at least 4 seconds with only one audible speaker.</div>"
                        )
                        allow_update = gr.Checkbox(
                            label="Allow controlled re-enrollment for this User ID",
                            value=False,
                        )
                        enroll_button = gr.Button(
                            "Create enrolled identity",
                            variant="primary",
                            elem_classes=["primary-action"],
                        )

                with gr.Column(scale=6):
                    with gr.Group(elem_classes=["panel", "progress-card"]):
                        gr.HTML(
                            '<div class="panel-heading"><h2>Enrollment progress</h2>'
                            "<p>Template status is confirmed after the enrollment pipeline completes.</p></div>"
                        )
                        enrollment_status = gr.HTML(
                            value='<section class="enrollment-result decision-waiting">'
                            '<span class="result-overline">READY TO ENROLL</span>'
                            "<h2>Awaiting enrollment</h2>"
                            "<p>Provide a User ID, face image, and voice sample to begin.</p></section>"
                        )
                        with gr.Row():
                            enroll_face_template_status = gr.HTML(
                                value=render_template_status(
                                    "Face template",
                                    "512-dimensional identity embedding",
                                    detail="Generated during enrollment.",
                                )
                            )
                            enroll_voice_template_status = gr.HTML(
                                value=render_template_status(
                                    "Voice template",
                                    "192-dimensional speaker embedding",
                                    detail="Generated during enrollment.",
                                )
                            )
                        with gr.Accordion("Enrollment details", open=False):
                            enrollment_raw = gr.JSON(label="Enrollment result")

        with gr.Tab("Registered Users"):
            with gr.Group(elem_classes=["panel"]):
                gr.HTML(
                    '<div class="panel-heading"><h2>Registered identities</h2>'
                    "<p>Manage enrolled user records. Biometric embeddings are never displayed.</p></div>"
                )
                refresh_users_button = gr.Button("Refresh registered users", size="sm")
                user_table = gr.Dataframe(
                    headers=["User ID", "Name", "Status", "Enrolled At", "Updated At"],
                    datatype=["str", "str", "str", "str", "str"],
                    label="Enrolled users",
                    interactive=False,
                    value=refresh_users(),
                )
                with gr.Row():
                    manage_user_id = gr.Textbox(label="User ID to delete")
                    delete_button = gr.Button(
                        "Delete user and templates",
                        variant="stop",
                    )
                delete_status = gr.Markdown()

    enroll_button.click(
        enroll,
        inputs=[enroll_user_id, enroll_name, enroll_faces, enroll_audio, allow_update],
        outputs=[
            enrollment_status,
            enroll_audio_enhancement_status,
            enroll_face_template_status,
            enroll_voice_template_status,
            enrollment_raw,
            auth_user_id,
            user_table,
        ],
    )
    enroll_audio.change(
        lambda payload: audio_capture_update(payload, purpose="enrollment"),
        inputs=[enroll_audio],
        outputs=[enroll_audio_status, enroll_audio_enhancement_status],
    )
    auth_button.click(
        authenticate,
        inputs=[auth_user_id, auth_face, auth_audio, auth_voice_challenge_state],
        outputs=[
            auth_system_banner,
            auth_decision,
            *auth_module_cards,
            auth_results,
            auth_raw,
            auth_audio_status,
            auth_audio_enhancement_status,
            auth_voice_challenge_display,
            auth_voice_challenge_state,
            auth_audio,
        ],
        show_progress="full",
        trigger_mode="once",
    )
    auth_voice_challenge_refresh.click(
        issue_voice_challenge,
        outputs=[auth_voice_challenge_display, auth_voice_challenge_state],
    )
    demo.load(
        issue_voice_challenge,
        outputs=[auth_voice_challenge_display, auth_voice_challenge_state],
    )
    auth_audio.change(
        audio_capture_update,
        inputs=[auth_audio],
        outputs=[auth_audio_status, auth_audio_enhancement_status],
    )
    auth_user_id_refresh.click(
        lambda: gr.update(choices=[user["user_id"] for user in list_users()]),
        outputs=[auth_user_id],
    )
    refresh_users_button.click(refresh_users, outputs=[user_table])
    delete_button.click(
        delete_selected_user,
        inputs=[manage_user_id],
        outputs=[delete_status, user_table, auth_user_id],
    )

    gr.HTML(
        """
        <footer class="privacy-note">
          Prototype only — not production identity security. Raw biometric samples are used for on-demand processing;
          stored templates remain sensitive and should be protected. Model thresholds require broader independent evaluation.
        </footer>
        """
    )


if __name__ == "__main__":
    username = os.environ.get("ZT_APP_USERNAME")
    password = os.environ.get("ZT_APP_PASSWORD")
    if bool(username) != bool(password):
        raise RuntimeError("Set both ZT_APP_USERNAME and ZT_APP_PASSWORD, or neither.")
    auth_pair = (username, password) if username and password else None
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        auth=auth_pair,
    )
