import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integration"))

from enrollment_store import (
    EnrollmentError,
    delete_user,
    get_user,
    list_users,
    save_enrollment,
)


def template_arrays():
    face = np.ones(512, dtype=np.float32)
    voice = torch.ones(192, dtype=torch.float32)
    return face, voice


def test_successful_enrollment_and_lookup(tmp_path, monkeypatch):
    monkeypatch.setattr("enrollment_store.STORE_DIR", tmp_path)
    face, voice = template_arrays()
    record = save_enrollment("USER001", "Alice", face, voice, 2, 2)
    assert record["user_id"] == "USER001"
    assert get_user("USER001")["username"] == "Alice"
    assert len(list_users()) == 1


def test_duplicate_user_id_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr("enrollment_store.STORE_DIR", tmp_path)
    face, voice = template_arrays()
    save_enrollment("USER001", "Alice", face, voice, 1, 1)
    with pytest.raises(EnrollmentError, match="already enrolled"):
        save_enrollment("USER001", "Alice 2", face, voice, 1, 1)


def test_controlled_reenrollment_updates_record(tmp_path, monkeypatch):
    monkeypatch.setattr("enrollment_store.STORE_DIR", tmp_path)
    face, voice = template_arrays()
    first = save_enrollment("USER001", "Alice", face, voice, 1, 1)
    updated = save_enrollment(
        "USER001", "Alice Updated", face, voice, 3, 3, allow_update=True
    )
    assert updated["username"] == "Alice Updated"
    assert updated["enrolled_at"] == first["enrolled_at"]


def test_delete_removes_templates(tmp_path, monkeypatch):
    monkeypatch.setattr("enrollment_store.STORE_DIR", tmp_path)
    face, voice = template_arrays()
    save_enrollment("USER001", "Alice", face, voice, 1, 1)
    delete_user("USER001")
    with pytest.raises(EnrollmentError, match="not enrolled"):
        get_user("USER001")


def test_missing_template_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr("enrollment_store.STORE_DIR", tmp_path)
    face, voice = template_arrays()
    with pytest.raises(EnrollmentError, match="[Bb]oth face and voice"):
        save_enrollment("USER001", "Alice", face, None, 1, 1)


def test_invalid_user_id_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr("enrollment_store.STORE_DIR", tmp_path)
    face, voice = template_arrays()
    with pytest.raises(EnrollmentError, match="only letters"):
        save_enrollment("../../outside", "Alice", face, voice, 1, 1)
