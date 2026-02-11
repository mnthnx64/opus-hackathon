"""Tests for all three notation converters."""

import pytest
from app.models.music_representation import Note, Duration
from app.services.converters.staff_converter import convert_to_staff
from app.services.converters.hindustani_converter import (
    convert_to_hindustani, SEMITONE_TO_SWARA,
)
from app.services.converters.carnatic_converter import (
    convert_to_carnatic, SEMITONE_TO_CARNATIC,
)


def _make_note(midi_pitch: int, start: float, end: float, conf: float = 0.9) -> Note:
    return Note(
        midi_pitch=midi_pitch,
        start_time=start,
        end_time=end,
        confidence=conf,
        duration=Duration.QUARTER,
    )


# --- Duration quantization ---

class TestDurationQuantization:
    def test_whole_note_threshold(self):
        """WHOLE >= 3.5 beats (bug fix: was 3.0)."""
        assert Duration.from_beats(3.5) == Duration.WHOLE
        assert Duration.from_beats(4.0) == Duration.WHOLE
        # 3.0 should be HALF, not WHOLE
        assert Duration.from_beats(3.0) == Duration.HALF

    def test_half_note(self):
        assert Duration.from_beats(2.0) == Duration.HALF
        assert Duration.from_beats(1.5) == Duration.HALF

    def test_quarter_note(self):
        assert Duration.from_beats(1.0) == Duration.QUARTER
        assert Duration.from_beats(0.75) == Duration.QUARTER

    def test_eighth_note(self):
        assert Duration.from_beats(0.5) == Duration.EIGHTH
        assert Duration.from_beats(0.375) == Duration.EIGHTH

    def test_sixteenth_note(self):
        assert Duration.from_beats(0.25) == Duration.SIXTEENTH
        assert Duration.from_beats(0.1) == Duration.SIXTEENTH


# --- Staff converter ---

class TestStaffConverter:
    def test_basic_conversion(self):
        notes = [_make_note(60, 0.0, 0.5), _make_note(62, 0.5, 1.0)]
        result = convert_to_staff(notes, tempo=120.0, beat_duration=0.5)
        assert result.notation_type.value == "staff"
        assert len(result.notes) >= 2

    def test_note_names(self):
        notes = [_make_note(60, 0.0, 0.5)]  # C4
        result = convert_to_staff(notes, beat_duration=0.5)
        staff_note = result.notes[0]
        assert staff_note.note_name == "C4"
        assert staff_note.octave == 4
        assert staff_note.accidental == ""

    def test_sharp_note(self):
        notes = [_make_note(61, 0.0, 0.5)]  # C#4
        result = convert_to_staff(notes, beat_duration=0.5)
        assert result.notes[0].accidental == "#"
        assert "C#" in result.notes[0].note_name

    def test_rest_insertion(self):
        """Gap between notes should produce a rest."""
        notes = [_make_note(60, 0.0, 0.3), _make_note(62, 0.8, 1.3)]
        result = convert_to_staff(notes, beat_duration=0.5)
        rests = [n for n in result.notes if n.is_rest]
        assert len(rests) >= 1

    def test_bar_grouping(self):
        """Notes should be grouped into bars."""
        notes = [
            _make_note(60, 0.0, 0.5),
            _make_note(62, 0.5, 1.0),
            _make_note(64, 1.0, 1.5),
            _make_note(65, 1.5, 2.0),
            _make_note(67, 2.0, 2.5),  # This should be in bar 2 with 4/4 @ 120BPM
        ]
        result = convert_to_staff(notes, tempo=120.0, beat_duration=0.5)
        assert len(result.bars) >= 1

    def test_empty_notes(self):
        result = convert_to_staff([], beat_duration=0.5)
        assert result.notes == []
        assert result.bars == []


# --- Hindustani converter ---

class TestHindustaniConverter:
    def test_all_12_semitones(self):
        """Every semitone maps to a valid swara."""
        base = 60  # C4 = Sa
        for i in range(12):
            notes = [_make_note(base + i, 0.0, 0.5)]
            result = convert_to_hindustani(notes, base_pitch=base, beat_duration=0.5)
            assert result.notes[0].swara == SEMITONE_TO_SWARA[i][0]
            assert result.notes[0].swara_variant == SEMITONE_TO_SWARA[i][1]

    def test_sa_mapping(self):
        notes = [_make_note(60, 0.0, 0.5)]
        result = convert_to_hindustani(notes, base_pitch=60, beat_duration=0.5)
        assert result.notes[0].swara == "Sa"
        assert result.notes[0].swara_variant == ""

    def test_komal_re(self):
        notes = [_make_note(61, 0.0, 0.5)]
        result = convert_to_hindustani(notes, base_pitch=60, beat_duration=0.5)
        assert result.notes[0].swara == "Re"
        assert result.notes[0].swara_variant == "komal"

    def test_tivra_ma(self):
        notes = [_make_note(66, 0.0, 0.5)]
        result = convert_to_hindustani(notes, base_pitch=60, beat_duration=0.5)
        assert result.notes[0].swara == "Ma"
        assert result.notes[0].swara_variant == "tivra"

    def test_octave_offset(self):
        """Notes above Sa octave should have positive offset."""
        notes = [_make_note(72, 0.0, 0.5)]  # Sa one octave up
        result = convert_to_hindustani(notes, base_pitch=60, beat_duration=0.5)
        assert result.notes[0].octave_offset == 1

    def test_taal_grouping(self):
        notes = [_make_note(60 + i, i * 0.5, (i + 1) * 0.5) for i in range(8)]
        result = convert_to_hindustani(notes, base_pitch=60, beat_duration=0.5)
        assert result.taal_name == "Teentaal"
        assert len(result.bars) >= 1


# --- Carnatic converter ---

class TestCarnaticConverter:
    def test_all_12_semitones(self):
        base = 60
        for i in range(12):
            notes = [_make_note(base + i, 0.0, 0.5)]
            result = convert_to_carnatic(notes, base_pitch=base, beat_duration=0.5)
            assert result.notes[0].swara == SEMITONE_TO_CARNATIC[i][0]
            assert result.notes[0].swara_variant == SEMITONE_TO_CARNATIC[i][1]

    def test_sa_mapping(self):
        notes = [_make_note(60, 0.0, 0.5)]
        result = convert_to_carnatic(notes, base_pitch=60, beat_duration=0.5)
        assert result.notes[0].swara == "Sa"

    def test_ri_variants(self):
        """R1 (semitone 1) and R2 (semitone 2) should have different variants."""
        n1 = [_make_note(61, 0.0, 0.5)]
        n2 = [_make_note(62, 0.0, 0.5)]
        r1 = convert_to_carnatic(n1, base_pitch=60, beat_duration=0.5)
        r2 = convert_to_carnatic(n2, base_pitch=60, beat_duration=0.5)
        assert r1.notes[0].swara_variant == "1"
        assert r2.notes[0].swara_variant == "2"

    def test_silence_insertion(self):
        notes = [_make_note(60, 0.0, 0.3), _make_note(62, 0.8, 1.3)]
        result = convert_to_carnatic(notes, base_pitch=60, beat_duration=0.5)
        silences = [n for n in result.notes if n.is_rest]
        assert len(silences) >= 1
        assert silences[0].swara == ","

    def test_tala_grouping(self):
        notes = [_make_note(60 + i, i * 0.5, (i + 1) * 0.5) for i in range(8)]
        result = convert_to_carnatic(notes, base_pitch=60, beat_duration=0.5)
        assert result.tala_name == "Adi"


# --- Serialization ---

class TestSerialization:
    def test_to_dict(self):
        notes = [_make_note(60, 0.0, 0.5)]
        result = convert_to_staff(notes, beat_duration=0.5)
        d = result.to_dict()
        assert "notes" in d
        assert "bars" in d
        assert "tempo" in d
        assert isinstance(d["notes"], list)
        assert d["notes"][0]["midi_pitch"] == 60
