"""Tempo and beat tracking via librosa."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import librosa
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TempoResult:
    tempo: float  # BPM
    beat_times: list[float]  # seconds
    time_signature: str  # e.g., "4/4"
    beat_duration: float  # seconds per beat


def analyze_tempo(audio: np.ndarray, sr: int = 22050) -> TempoResult:
    """Detect tempo and beat positions from audio."""
    tempo_arr, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
    # librosa returns tempo as array in newer versions
    tempo = float(tempo_arr[0]) if hasattr(tempo_arr, '__len__') else float(tempo_arr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    beat_duration = 60.0 / tempo if tempo > 0 else 0.5

    # Default to 4/4 (simplified)
    time_signature = "4/4"

    logger.info("Detected tempo: %.1f BPM, %d beats", tempo, len(beat_times))

    return TempoResult(
        tempo=tempo,
        beat_times=beat_times,
        time_signature=time_signature,
        beat_duration=beat_duration,
    )
