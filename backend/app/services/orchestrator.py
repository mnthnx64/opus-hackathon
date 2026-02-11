"""Pipeline orchestrator: audio → pitch detection → tempo → conversion."""

from __future__ import annotations

import logging
import os

from app.models.music_representation import TranscriptionResult, NotationType, midi_to_base_name
from app.services.audio_processor import load_audio, save_temp_wav
from app.services.pitch_detector import pitch_detector
from app.services.tempo_analyzer import analyze_tempo
from app.services.converters.staff_converter import convert_to_staff
from app.services.converters.hindustani_converter import convert_to_hindustani
from app.services.converters.carnatic_converter import convert_to_carnatic

logger = logging.getLogger(__name__)


def transcribe(
    audio_path: str,
    base_pitch: int = 60,
    notation_type: str = "staff",
) -> TranscriptionResult:
    """Full transcription pipeline: audio file → structured notation."""
    # 1. Load and preprocess audio
    audio, sr = load_audio(audio_path)
    logger.info("Loaded audio: %.2f seconds, sr=%d", len(audio) / sr, sr)

    # 2. Detect pitches
    detection = pitch_detector.predict(audio_path)
    notes = detection.notes
    logger.info("Pitch detection: %d notes found", len(notes))

    if not notes:
        return TranscriptionResult(
            notation_type=NotationType(notation_type),
            base_pitch=base_pitch,
            base_pitch_name=midi_to_base_name(base_pitch),
        )

    # 3. Analyze tempo
    tempo_result = analyze_tempo(audio, sr)
    logger.info("Tempo: %.1f BPM", tempo_result.tempo)

    # 4. Convert to requested notation
    nt = NotationType(notation_type)

    if nt == NotationType.STAFF:
        result = convert_to_staff(
            notes=notes,
            tempo=tempo_result.tempo,
            time_signature=tempo_result.time_signature,
            beat_duration=tempo_result.beat_duration,
        )
    elif nt == NotationType.HINDUSTANI:
        result = convert_to_hindustani(
            notes=notes,
            base_pitch=base_pitch,
            tempo=tempo_result.tempo,
            beat_duration=tempo_result.beat_duration,
        )
    elif nt == NotationType.CARNATIC:
        result = convert_to_carnatic(
            notes=notes,
            base_pitch=base_pitch,
            tempo=tempo_result.tempo,
            beat_duration=tempo_result.beat_duration,
        )
    else:
        raise ValueError(f"Unknown notation type: {notation_type}")

    result.base_pitch = base_pitch
    result.base_pitch_name = midi_to_base_name(base_pitch)
    return result


def transcribe_live_chunk(
    audio_buffer,
    sr: int,
    base_pitch: int = 60,
    notation_type: str = "hindustani",
) -> TranscriptionResult:
    """Transcribe a live audio chunk. Creates temp file for Basic Pitch."""
    temp_path = save_temp_wav(audio_buffer, sr)
    try:
        return transcribe(temp_path, base_pitch, notation_type)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
