#!/usr/bin/env python3
"""Download and SHA-256 verify the official MiniFASNetV2 M1 assets."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
M1_DIRECTORY = ROOT / "models" / "M1"
REVISION = "b6d5f04ad78778917853b25c778acef6d5626d15"
BASE_URL = (
    "https://raw.githubusercontent.com/minivision-ai/"
    f"Silent-Face-Anti-Spoofing/{REVISION}/"
)

ASSETS = {
    "2.7_80x80_MiniFASNetV2.pth": (
        "resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth",
        "a5eb02e1843f19b5386b953cc4c9f011c3f985d0ee2bb9819eea9a142099bec0",
    ),
    "4_0_0_80x80_MiniFASNetV1SE.pth": (
        "resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth",
        "84ee1d37d96894d5e82de5a57df044ef80a58be2b218b5ed7cdfd875ec2f5990",
    ),
    "Widerface-RetinaFace.caffemodel": (
        "resources/detection_model/Widerface-RetinaFace.caffemodel",
        "d08338a2c207df16a9c566f767fea67fb43ba6fff76ce11e938fe3fabefb9402",
    ),
    "deploy.prototxt": (
        "resources/detection_model/deploy.prototxt",
        "9fe2f141b4baee039ed9442da2833e216af40a6ff3e639e7b39258812bcda808",
    ),
    "vendor/MiniFASNet.py": (
        "src/model_lib/MiniFASNet.py",
        "e498c4ec5e1ddfaba62b941a126c19d65aa564999f3309661fe43ee8bf38acd7",
    ),
    "vendor/LICENSE": (
        "LICENSE",
        "daf94bf1dc9cc5700fe5af2c7c0cbd1836e70d509ed78fd0bebef7432edee6fb",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_asset(relative_path: str, source_path: str, expected_hash: str) -> None:
    destination = M1_DIRECTORY / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256(destination) == expected_hash:
        print(f"Verified existing asset: {relative_path}")
        return

    url = BASE_URL + source_path
    temporary_path = None
    try:
        with urlopen(url, timeout=60) as response:
            if response.status != 200:
                raise RuntimeError(f"Download returned HTTP {response.status}: {url}")
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=destination.name + ".",
                suffix=".download",
                delete=False,
            ) as output:
                temporary_path = Path(output.name)
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)

        actual_hash = sha256(temporary_path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"SHA-256 mismatch for {relative_path}: expected "
                f"{expected_hash}, received {actual_hash}"
            )
        os.replace(temporary_path, destination)
        temporary_path = None
        print(f"Downloaded and verified: {relative_path} ({actual_hash})")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    for relative_path, (source_path, expected_hash) in ASSETS.items():
        download_asset(relative_path, source_path, expected_hash)


if __name__ == "__main__":
    main()
