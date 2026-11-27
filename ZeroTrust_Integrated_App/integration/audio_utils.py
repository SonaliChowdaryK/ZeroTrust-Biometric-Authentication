from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def find_ffmpeg() -> str:
    resolved = shutil.which("ffmpeg")
    if resolved:
        return resolved
    try:
        import imageio_ffmpeg
        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled:
            return bundled
    except ImportError:
        pass

    candidates = list(Path("C:/Users").glob("*/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg*/ffmpeg-*-full_build/bin/ffmpeg.exe"))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError("FFmpeg is unavailable. It is required to decode microphone and compressed audio.")


def convert_audio_file(audio_path, output_path, sample_rate, channels=1):
    source = Path(str(audio_path))
    destination = Path(output_path)
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError("Audio file is missing or empty.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        find_ffmpeg(),
        "-nostdin",
        "-y",
        "-v", "error",
        "-i", str(source),
        "-map", "0:a:0",
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        "-vn",
        str(destination),
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        destination.unlink(missing_ok=True)
        message = completed.stderr.strip() or "FFmpeg could not decode the audio."
        raise ValueError(message)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise ValueError("Audio conversion produced an empty file.")
    return destination
