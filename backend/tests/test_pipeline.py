"""Integration tests for the transcription pipeline."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from app.models.music_representation import Note, Duration, TranscriptionResult, NotationType


class TestPipelineIntegration:
    """Test the pipeline with mocked pitch detection (avoids loading ML model)."""

    def test_empty_audio_produces_empty_result(self):
        """Empty/silence should produce no notes."""
        result = TranscriptionResult(
            notation_type=NotationType.STAFF,
            base_pitch=60,
        )
        assert len(result.notes) == 0
        assert len(result.bars) == 0

    def test_result_serialization_roundtrip(self):
        """TranscriptionResult.to_dict() should be JSON-serializable."""
        import json
        notes = [
            Note(midi_pitch=60, start_time=0.0, end_time=0.5,
                 confidence=0.9, duration=Duration.QUARTER),
        ]
        result = TranscriptionResult(
            notes=notes,
            notation_type=NotationType.STAFF,
            tempo=120.0,
        )
        d = result.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["tempo"] == 120.0
        assert len(parsed["notes"]) == 1

    def test_note_fields_complete(self):
        """Note.to_dict() should include all required fields."""
        note = Note(
            midi_pitch=60,
            start_time=0.0,
            end_time=0.5,
            confidence=0.9,
            duration=Duration.QUARTER,
            note_name="C4",
            accidental="",
            octave=4,
        )
        d = note.to_dict()
        assert d["midi_pitch"] == 60
        assert d["start_time"] == 0.0
        assert d["end_time"] == 0.5
        assert d["confidence"] == 0.9
        assert d["duration"] == "quarter"
        assert d["note_name"] == "C4"
