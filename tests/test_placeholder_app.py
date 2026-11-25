from pathlib import Path

APP_PATH = Path(__file__).resolve().parents[1] / "deployment" / "app.py"
APP_TEXT = APP_PATH.read_text(encoding="utf-8")

def test_placeholder_discloses_not_connected_models():
    expected_modules = (
        "M1 Face Anti-Spoofing: NOT CONNECTED",
        "M2 Face Recognition: NOT CONNECTED",
        "M3 Speaker Recognition: NOT CONNECTED",
        "M4 Voice Anti-Spoofing: NOT CONNECTED",
    )
    for module_status in expected_modules:
        assert module_status in APP_TEXT
    assert "NOT CONNECTED" in APP_TEXT

def test_placeholder_is_fail_safe_and_not_authentication():
    assert "REQUIRE ADDITIONAL VERIFICATION" in APP_TEXT
    assert "not be used for real authentication" in APP_TEXT.lower()

def test_privacy_warning_present():
    assert "Do not upload real biometric data" in APP_TEXT
