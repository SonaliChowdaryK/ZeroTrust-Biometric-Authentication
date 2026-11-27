from __future__ import annotations

import json
import os
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
STORE_DIR = ROOT / "enrollments"


class EnrollmentError(Exception):
    pass


def _safe_user_id(user_id: str) -> str:
    if not isinstance(user_id, str):
        raise EnrollmentError("User ID is required.")
    value = user_id.strip()
    if not value:
        raise EnrollmentError("User ID is required.")
    if len(value) > 128:
        raise EnrollmentError("User ID must be 128 characters or fewer.")
    if not all(char.isalnum() or char in "-_" for char in value):
        raise EnrollmentError("User ID may contain only letters, numbers, hyphens, and underscores.")
    return value


def _store_root() -> Path:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(STORE_DIR, 0o700)
    except (OSError, NotImplementedError):
        pass
    return STORE_DIR


def _record_path(user_id: str) -> Path:
    return _store_root() / _safe_user_id(user_id) / "record.json"


def list_users() -> list[dict]:
    users = []
    root = _store_root()
    for record_path in sorted(root.glob("*/record.json")):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            users.append({
                "user_id": record["user_id"],
                "username": record["username"],
                "status": record["status"],
                "enrolled_at": record["enrolled_at"],
                "updated_at": record["updated_at"],
            })
        except (KeyError, json.JSONDecodeError, OSError):
            continue
    return users


def get_user(user_id: str) -> dict:
    path = _record_path(user_id)
    if not path.exists():
        raise EnrollmentError(f"User ID '{_safe_user_id(user_id)}' is not enrolled.")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise EnrollmentError("The enrollment record is unreadable.") from exc
    if record.get("status") != "enrolled":
        raise EnrollmentError("The user enrollment is incomplete.")
    return record


def user_exists(user_id: str) -> bool:
    try:
        path = _record_path(user_id)
        return path.exists()
    except EnrollmentError:
        return False


def save_enrollment(
    user_id: str,
    username: str,
    face_template,
    voice_template,
    face_images_used: int,
    voice_samples_used: int,
    allow_update: bool = False,
) -> dict:
    safe_id = _safe_user_id(user_id)
    safe_name = username.strip() if isinstance(username, str) else ""
    if not safe_name:
        raise EnrollmentError("Username is required.")
    if len(safe_name) > 128:
        raise EnrollmentError("Username must be 128 characters or fewer.")
    if face_template is None or voice_template is None:
        raise EnrollmentError("Both face and voice templates are required.")

    user_dir = _store_root() / safe_id
    exists = user_dir.exists()
    if exists and not allow_update:
        raise EnrollmentError(f"User ID '{safe_id}' is already enrolled.")

    import numpy as np
    import torch

    if hasattr(face_template, "shape") and tuple(face_template.shape) != (512,):
        raise EnrollmentError("Invalid face template dimension.")
    if hasattr(voice_template, "shape") and tuple(voice_template.shape) != (192,):
        raise EnrollmentError("Invalid voice template dimension.")

    face_array = np.asarray(face_template, dtype=np.float32).reshape(-1)
    if face_array.shape != (512,) or not np.isfinite(face_array).all():
        raise EnrollmentError("Invalid face template.")
    face_array /= np.linalg.norm(face_array) + 1e-10

    voice_tensor = voice_template.detach().cpu() if hasattr(voice_template, "detach") else torch.as_tensor(voice_template)
    voice_tensor = torch.as_tensor(voice_tensor, dtype=torch.float32).reshape(-1)
    if tuple(voice_tensor.shape) != (192,) or not torch.isfinite(voice_tensor).all():
        raise EnrollmentError("Invalid voice template.")
    voice_tensor /= torch.linalg.vector_norm(voice_tensor) + 1e-10

    staging = _store_root() / f".{safe_id}.{secrets.token_hex(6)}"
    staging.mkdir(mode=0o700)
    now = datetime.now(timezone.utc).isoformat()
    try:
        np.save(staging / "face_template.npy", face_array, allow_pickle=False)
        torch.save(voice_tensor, staging / "voice_template.pt")
        record = {
            "user_id": safe_id,
            "username": safe_name,
            "status": "enrolled",
            "face_template": "face_template.npy",
            "voice_template": "voice_template.pt",
            "face_template_dimension": 512,
            "voice_template_dimension": 192,
            "face_images_used": int(face_images_used),
            "voice_samples_used": int(voice_samples_used),
            "enrolled_at": get_user(safe_id)["enrolled_at"] if exists else now,
            "updated_at": now,
        }
        (staging / "record.json").write_text(
            json.dumps(record, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        for path in staging.iterdir():
            try:
                os.chmod(path, 0o600)
            except (OSError, NotImplementedError):
                pass

        backup = None
        if exists:
            backup = _store_root() / f".backup.{safe_id}.{secrets.token_hex(6)}"
            user_dir.rename(backup)
        staging.rename(user_dir)
        if backup is not None:
            shutil.rmtree(backup)
        return {
            "user_id": safe_id,
            "username": safe_name,
            "status": "enrolled",
            "enrolled_at": record["enrolled_at"],
            "updated_at": now,
        }
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def delete_user(user_id: str) -> dict:
    safe_id = _safe_user_id(user_id)
    user_dir = _store_root() / safe_id
    if not user_dir.exists():
        raise EnrollmentError(f"User ID '{safe_id}' is not enrolled.")
    record = get_user(safe_id)
    shutil.rmtree(user_dir)
    return record
