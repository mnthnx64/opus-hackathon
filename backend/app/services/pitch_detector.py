"""Basic Pitch wrapper using ONNX/CoreML Model class (NOT deprecated TensorFlow)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from app.models.music_representation import Note, Duration
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class PitchDetectionResult:
    notes: list[Note]
    midi_data: object  # pretty_midi.PrettyMIDI


class PitchDetector:
    def __init__(self):
        self.model = None

    def load_model(self):
        """Load Basic Pitch model at startup. Auto-detects CoreML/ONNX."""
        from basic_pitch.inference import Model
        from basic_pitch import ICASSP_2022_MODEL_PATH

        logger.info("Loading Basic Pitch model from %s", ICASSP_2022_MODEL_PATH)
        self.model = Model(ICASSP_2022_MODEL_PATH)
        logger.info("Basic Pitch model loaded successfully")

    def predict(self, audio_path: str, confidence_threshold: float = None) -> PitchDetectionResult:
        """Run pitch detection on an audio file.

        Returns structured Note objects with MIDI pitches, times, and confidence.
        """
        from basic_pitch.inference import predict

        if self.model is None:
            self.load_model()

        threshold = confidence_threshold or settings.confidence_threshold

        model_output, midi_data, note_events = predict(
            audio_path,
            self.model,
            onset_threshold=0.5,
            frame_threshold=0.3,
            minimum_note_length=58,  # ms
            midi_tempo=120.0,
        )

        notes = []
        for start, end, pitch, velocity, confidence_values in note_events:
            conf = float(np.mean(confidence_values)) if hasattr(confidence_values, '__len__') else float(velocity / 127.0)

            beat_duration = (end - start) * 2.0  # approx at 120 BPM
            duration = Duration.from_beats(beat_duration)

            note = Note(
                midi_pitch=int(pitch),
                start_time=float(start),
                end_time=float(end),
                confidence=conf,
                duration=duration,
            )
            notes.append(note)

        # Sort by start time
        notes.sort(key=lambda n: n.start_time)
        logger.info("Detected %d notes", len(notes))

        return PitchDetectionResult(notes=notes, midi_data=midi_data)


# Singleton
pitch_detector = PitchDetector()
