import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integration"))

import app as app_module
from app import authentication_outputs, render_decision_card, render_module_card


def test_module_card_shows_actual_status_decision_and_score():
    card = render_module_card(
        "M2",
        {
            "status": "completed",
            "decision": "ACCEPT",
            "similarity": 0.91234,
        },
    )

    assert "module-card accepted" in card
    assert "COMPLETED" in card
    assert "ACCEPT" in card
    assert "0.9123" in card


def test_module_card_escapes_error_details():
    card = render_module_card(
        "M4",
        {"status": "error", "decision": "UNKNOWN", "error": "<script>alert(1)</script>"},
    )

    assert "module-card error" in card
    assert "<script>" not in card
    assert "&lt;script&gt;" in card


def test_m1_card_shows_actual_liveness_states():
    cases = (
        (None, "NOT TESTED", None),
        ({"status": "processing", "decision": "UNKNOWN"}, "PROCESSING", None),
        (
            {"status": "completed", "decision": "ACCEPT", "prediction": "LIVE"},
            "LIVE DETECTED",
            "LIVE DETECTED",
        ),
        (
            {"status": "completed", "decision": "REJECT", "prediction": "SPOOF"},
            "SPOOF DETECTED",
            "SPOOF DETECTED",
        ),
        (
            {"status": "unavailable", "decision": "UNKNOWN", "reason": "Face not detected."},
            "FACE NOT DETECTED",
            None,
        ),
        ({"status": "error", "decision": "UNKNOWN"}, "ERROR", None),
    )

    for evidence, status_label, decision_label in cases:
        card = render_module_card("M1", evidence)
        assert status_label in card
        if decision_label:
            assert decision_label in card


def test_m4_card_shows_only_actual_audio_decisions():
    assert "NOT TESTED" in render_module_card("M4")
    assert "PROCESSING" in render_module_card(
        "M4", {"status": "processing", "decision": "UNKNOWN"}
    )
    genuine_card = render_module_card(
        "M4",
        {
            "status": "completed",
            "decision": "ACCEPT",
            "prediction": "BONAFIDE",
            "bonafide_probability": 0.9,
            "spoof_probability": 0.1,
        },
    )
    assert "GENUINE DETECTED" in genuine_card
    assert ">Bonafide score (uncal.)</span>" in genuine_card
    assert ">Spoof score (uncal.)</span>" in genuine_card
    assert "AASIST softmax scores are uncalibrated." in genuine_card
    assert "SPOOF DETECTED" in render_module_card(
        "M4",
        {
            "status": "completed",
            "decision": "REJECT",
            "prediction": "SPOOF",
            "bonafide_probability": 0.1,
            "spoof_probability": 0.9,
        },
    )
    assert "ERROR" in render_module_card(
        "M4", {"status": "error", "decision": "UNKNOWN"}
    )
    assert "AUDIO QUALITY INSUFFICIENT" in render_module_card(
        "M4",
        {
            "status": "not_run",
            "decision": "UNKNOWN",
            "reason": "No speech detected. Please record again.",
        },
    )


def test_decision_card_uses_central_decision_tone_and_escapes_reason():
    card = render_decision_card("DENY", "<b>reason</b>")

    assert "decision-deny" in card
    assert "DENY" in card
    assert "&lt;b&gt;reason&lt;/b&gt;" in card


def test_missing_authentication_evidence_displays_not_run_cards():
    outputs = authentication_outputs(
        reason="Audio sample is missing.",
        raw={"error": "Audio sample is missing"},
    )

    assert len(outputs) == 10
    assert "status-warning" in outputs[0]
    assert all("NOT RUN" in card for card in outputs[2:6])
    assert outputs[7] == {"error": "Audio sample is missing"}
    assert "Noise cancellation" in outputs[9]


def test_processing_state_updates_banner_and_module_cards():
    outputs = authentication_outputs(processing=True)

    assert "status-processing" in outputs[0]
    assert "decision-processing" in outputs[1]
    assert all("PROCESSING" in card for card in outputs[2:6])


def test_authentication_emits_processing_then_actual_module_results(monkeypatch, tmp_path):
    evidence = [
        {
            "module_id": module_id,
            "module_name": name,
            "status": "completed",
            "decision": "ACCEPT",
            "verified": True,
            "similarity": 0.9,
        }
        for module_id, (name, _) in app_module.MODULE_DETAILS.items()
    ]
    monkeypatch.setattr(
        app_module,
        "run_authentication",
        lambda *args, **kwargs: {
            "module_evidence": evidence,
            "central_result": {
                "final_decision": "ALLOW",
                "reason": "All modules accepted.",
            },
        },
    )
    prepared_workspace = tmp_path / "audio-workspace"
    prepared_workspace.mkdir(exist_ok=True)
    prepared = app_module.PreparedAudio(
        original_path=Path("original-16k.wav"),
        enhanced_path=Path("enhanced-16k.wav"),
        analysis={
            "state": "ready",
            "raw_audio": "captured",
            "noise_suppression": "completed",
            "speech_detected": True,
            "speech_seconds": 2.4,
            "audio_quality": "passed",
            "overlap_assessment": "unavailable",
            "ready": True,
            "message": "Audio is ready.",
        },
        workspace=prepared_workspace,
    )
    monkeypatch.setattr(
        app_module,
        "prepare_audio_for_models",
        lambda audio_path: prepared,
    )

    challenge_audio_paths = []

    def verify_challenge(audio_path, _phrase):
        challenge_audio_paths.append(Path(audio_path))
        return {"matched": True, "transcript": "expected phrase"}

    monkeypatch.setattr(app_module, "verify_voice_challenge", verify_challenge)
    outputs = list(
        app_module.authenticate(
            "USER1",
            "face.jpg",
            "voice.wav",
            app_module.create_voice_challenge(),
        )
    )

    assert len(outputs) == 2
    assert len(outputs[0]) == 13
    assert "status-processing" in outputs[0][0]
    assert "Enabled - Processing" in outputs[0][9]
    assert all("PROCESSING" in card for card in outputs[0][2:6])
    assert "status-complete" in outputs[1][0]
    assert all("ACCEPT" in card for card in outputs[1][2:6])
    assert "Detected (2.40s)" in outputs[1][9]
    assert "Overlap check unavailable" in outputs[1][9]
    assert "PASSED" in outputs[1][10]
    assert "Fresh spoken challenge: passed" in outputs[1][8]
    assert outputs[1][12]["value"] is None
    assert challenge_audio_paths == [Path("original-16k.wav")]


def test_audio_enhancement_status_escapes_messages_and_reports_real_states():
    status = app_module.render_audio_enhancement({
        "state": "ready",
        "raw_audio": "captured",
        "noise_suppression": "completed",
        "speech_detected": True,
        "speech_seconds": 1.25,
        "audio_quality": "passed",
        "ready": True,
        "message": "<b>ready</b>",
    })

    assert "Enabled - Completed" in status
    assert "Detected (1.25s)" in status
    assert "Ready for verification" in status
    assert "&lt;b&gt;ready&lt;/b&gt;" in status
    assert "<b>ready</b>" not in status


def test_voice_challenge_requires_exact_order_and_is_case_insensitive():
    phrase = "blue river candle garden"

    assert app_module.verify_voice_challenge(
        "unused.wav",
        phrase,
        transcriber=lambda _: "Blue, river, candle, garden.",
    ) == {"matched": True, "transcript": "Blue, river, candle, garden."}
    assert not app_module.verify_voice_challenge(
        "unused.wav",
        phrase,
        transcriber=lambda _: "blue candle river garden",
    )["matched"]


def test_voice_challenge_expires_after_three_minutes():
    challenge = app_module.create_voice_challenge(now=100)

    assert app_module.is_voice_challenge_valid(challenge, now=280)
    assert not app_module.is_voice_challenge_valid(challenge, now=281)


def test_voice_challenge_render_escapes_phrase():
    rendered = app_module.render_voice_challenge(
        {"phrase": "<script>alert"},
    )

    assert "&lt;script&gt;" in rendered
    assert "<script>" not in rendered


def test_voice_processing_status_shows_sanitized_challenge_transcript():
    status = app_module.voice_processing_status({
        "module_evidence": [],
        "voice_challenge": {
            "status": "failed",
            "checked_phrase": "blue river candle garden",
            "transcript": "wrong words **unexpected**",
        },
    })

    assert "Phrase checked: `blue river candle garden`" in status
    assert "Offline recognizer heard: `wrong words unexpected`" in status
    assert "**unexpected**" not in status


def test_voice_processing_status_explains_empty_transcript():
    status = app_module.voice_processing_status({
        "module_evidence": [],
        "voice_challenge": {"status": "failed", "transcript": ""},
    })

    assert "Offline recognizer returned no words." in status


def test_voice_processing_status_displays_apostrophes_without_double_escaping():
    status = app_module.voice_processing_status({
        "module_evidence": [],
        "voice_challenge": {
            "status": "failed",
            "transcript": "didn't understand",
        },
    })

    assert "Offline recognizer heard: `didn't understand`" in status
    assert "&#x27;" not in status


def test_phrase_mismatch_runs_non_authoritative_voice_diagnostics(
    monkeypatch, tmp_path
):
    evidence = [
        {
            "module_id": module_id,
            "module_name": name,
            "status": "not_run" if module_id in {"M3", "M4"} else "completed",
            "decision": "UNKNOWN" if module_id in {"M3", "M4"} else "ACCEPT",
            "verified": None if module_id in {"M3", "M4"} else True,
            "reason": "Voice challenge did not match." if module_id in {"M3", "M4"} else None,
        }
        for module_id, (name, _) in app_module.MODULE_DETAILS.items()
    ]
    diagnostic_evidence = [
        {
            "module_id": "M3",
            "module_name": "Speaker Recognition",
            "status": "completed",
            "decision": "ACCEPT",
            "verified": True,
            "similarity": 0.91,
        },
        {
            "module_id": "M4",
            "module_name": "Voice Anti-Spoofing",
            "status": "completed",
            "decision": "REJECT",
            "verified": False,
            "bonafide_probability": 0.12,
            "spoof_probability": 0.88,
        },
    ]
    calls = []

    def fake_run_authentication(face_path, audio_path, **kwargs):
        calls.append((face_path, audio_path, kwargs))
        if face_path is None:
            return {
                "module_evidence": diagnostic_evidence,
                "central_result": {
                    "final_decision": "ALLOW",
                    "reason": "Diagnostic result must not be used.",
                },
            }
        return {
            "module_evidence": evidence,
            "central_result": {
                "final_decision": "REQUIRE_ADDITIONAL_VERIFICATION",
                "reason": "Voice challenge did not match.",
            },
        }

    workspace = tmp_path / "prepared"
    workspace.mkdir()
    prepared = app_module.PreparedAudio(
        original_path=Path("original-16k.wav"),
        enhanced_path=Path("enhanced-16k.wav"),
        analysis={
            "state": "ready",
            "raw_audio": "captured",
            "noise_suppression": "completed",
            "speech_detected": True,
            "speech_seconds": 2.4,
            "audio_quality": "passed",
            "overlap_assessment": "unavailable",
            "ready": True,
        },
        workspace=workspace,
    )
    monkeypatch.setattr(app_module, "run_authentication", fake_run_authentication)
    monkeypatch.setattr(app_module, "prepare_audio_for_models", lambda _: prepared)
    monkeypatch.setattr(
        app_module,
        "verify_voice_challenge",
        lambda *_: {"matched": False, "transcript": "wrong words"},
    )
    challenge = app_module.create_voice_challenge()

    outputs = list(app_module.authenticate("USER1", "face.jpg", "voice.wav", challenge))

    assert len(outputs) == 2
    assert len(outputs[0]) == 13
    assert len(outputs[1]) == 13
    assert len(calls) == 2
    assert "fresh voice challenge failed" in outputs[1][1].lower()
    assert "diagnostic M3/M4 results do not count" in outputs[1][1]
    assert "Record the current phrase and retry." in outputs[1][1]
    assert calls[0][0] == "face.jpg"
    assert calls[0][1] is None
    assert calls[0][2]["audio_preparation"]["ready"] is False
    assert calls[1][0] is None
    assert calls[1][1] == "original-16k.wav"
    assert calls[1][2]["audio_preparation"] is prepared
    assert "DIAGNOSTIC ONLY" in outputs[1][4]
    assert "ACCEPT" in outputs[1][4]
    assert "DIAGNOSTIC ONLY" in outputs[1][5]
    assert "REJECT" in outputs[1][5]
    assert "recognizer heard" in outputs[1][10].lower()
    assert "wrong words" in outputs[1][10]
    assert "not used in the final decision" in outputs[1][8]
    raw_result = outputs[1][7]
    assert raw_result["central_result"]["final_decision"] == "REQUIRE_ADDITIONAL_VERIFICATION"
    assert all(
        item["status"] == "not_run"
        for item in raw_result["module_evidence"]
        if item["module_id"] in {"M3", "M4"}
    )
    assert {
        item["module_id"]: item["status"]
        for item in raw_result["diagnostic_voice_evidence"]
    } == {"M3": "completed", "M4": "completed"}
    assert outputs[1][11] != challenge
    assert outputs[1][12]["value"] is None
