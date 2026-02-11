"""Load, normalize, and resample audio files."""

import tempfile
import os

import librosa
import numpy as np
import soundfile as sf

from app.core.config import settings


def load_audio(file_path: str, sr: int = None) -> tuple[np.ndarray, int]:
    """Load audio file, resample to target SR, convert to mono, peak normalize."""
    sr = sr or settings.sample_rate
    audio, sample_rate = librosa.load(file_path, sr=sr, mono=True)
    # Peak normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio, sample_rate


def save_temp_wav(audio_data: np.ndarray, sr: int = None) -> str:
    """Save audio array to a temporary WAV file. Caller must clean up."""
    sr = sr or settings.sample_rate
    fd, path = tempfile.mkstemp(suffix=".wav")
    try:
        sf.write(path, audio_data, sr)
    except Exception:
        os.close(fd)
        os.unlink(path)
        raise
    os.close(fd)
    return path


async def save_upload_to_temp(upload_file) -> str:
    """Save an UploadFile to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".wav")
    try:
        content = await upload_file.read()
        with os.fdopen(fd, "wb") as f:
            f.write(content)
    except Exception:
        os.close(fd)
        os.unlink(path)
        raise
    return path
