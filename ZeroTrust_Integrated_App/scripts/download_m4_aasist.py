#!/usr/bin/env python3
"""Download or verify the pinned official AASIST ASVspoof 2019 LA checkpoint."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "models" / "M4" / "aasist_2019_la.pth"
SOURCE_COMMIT = "a04c9863f63d44471dde8a6abcb3b082b07cd1d1"
DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/clovaai/aasist/"
    f"{SOURCE_COMMIT}/models/weights/AASIST.pth"
)
EXPECTED_SHA256 = (
    "51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    if DESTINATION.is_file():
        actual = sha256_file(DESTINATION)
        if actual != EXPECTED_SHA256:
            raise ValueError(
                f"Existing AASIST checkpoint checksum mismatch: {actual}"
            )
        print(f"Verified existing official AASIST checkpoint: {DESTINATION}")
        return 0

    temporary_path = None
    try:
        with urlopen(DOWNLOAD_URL, timeout=60) as response:
            with tempfile.NamedTemporaryFile(
                dir=DESTINATION.parent,
                prefix=".aasist-download-",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                shutil.copyfileobj(response, temporary_file)

        actual = sha256_file(temporary_path)
        if actual != EXPECTED_SHA256:
            raise ValueError(
                f"Downloaded AASIST checkpoint checksum mismatch: {actual}"
            )
        os.replace(temporary_path, DESTINATION)
        temporary_path = None
        print(f"Downloaded and verified official AASIST checkpoint: {DESTINATION}")
        return 0
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
