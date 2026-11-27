from __future__ import annotations

import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "vosk-model-small-en-us-0.15"
MODEL_URL = f"https://alphacephei.com/vosk/models/{MODEL_NAME}.zip"
MODEL_PARENT = ROOT / "models" / "asr"
MODEL_TARGET = MODEL_PARENT / MODEL_NAME


def main():
    if (MODEL_TARGET / "conf" / "model.conf").is_file():
        print(f"Offline voice-challenge model is ready: {MODEL_TARGET}")
        return
    if MODEL_TARGET.exists():
        raise RuntimeError(f"Incomplete model directory already exists: {MODEL_TARGET}")

    MODEL_PARENT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="zt-vosk-", dir=MODEL_PARENT) as temporary:
        temporary_path = Path(temporary)
        archive_path = temporary_path / f"{MODEL_NAME}.zip"
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
            with archive_path.open("wb") as archive_file:
                shutil.copyfileobj(response, archive_file)

        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                destination = (temporary_path / member.filename).resolve()
                if not destination.is_relative_to(temporary_path.resolve()):
                    raise ValueError("The voice-challenge model archive contains an unsafe path.")
            archive.extractall(temporary_path)

        extracted_model = temporary_path / MODEL_NAME
        if not (extracted_model / "conf" / "model.conf").is_file():
            raise RuntimeError("The downloaded archive does not contain the expected Vosk model.")
        shutil.copytree(extracted_model, MODEL_TARGET)

    print(f"Offline voice-challenge model is ready: {MODEL_TARGET}")


if __name__ == "__main__":
    main()
