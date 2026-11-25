from pathlib import Path

APP_PATH = Path(__file__).resolve().parents[1] / "deployment" / "app.py"
APP_TEXT = APP_PATH.read_text(encoding="utf-8")

def test_placeholder_discloses_not_connected_models():
    for module in ("M1", "M2", "M3", "M4"):
        assert f"{module}:" in APP_TEXT
    assert "NOT CONNECTED" in APP_TEXT

def test_placeholder_is_fail_safe_and_not_authentication():
    assert "REQUIRE ADDITIONAL VERIFICATION" in APP_TEXT
    assert "not be used for real authentication" in APP_TEXT.lower()

def test_privacy_warning_present():
    assert "Do not upload real biometric data" in APP_TEXT
