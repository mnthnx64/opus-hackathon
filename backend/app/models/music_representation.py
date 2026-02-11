"""
Intermediate Music Representation (IMR) — central data model.

All converters produce these dataclasses. Renderers consume them via to_dict().
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class NotationType(str, Enum):
    STAFF = "staff"
    HINDUSTANI = "hindustani"
    CARNATIC = "carnatic"


class Duration(str, Enum):
    WHOLE = "whole"
    HALF = "half"
    QUARTER = "quarter"
    EIGHTH = "eighth"
    SIXTEENTH = "sixteenth"

    @staticmethod
    def from_beats(beats: float) -> "Duration":
        """Quantize beat duration to nearest standard duration.

        Fixed thresholds — WHOLE >= 3.5 (not 3.0) to prevent
        half notes from becoming whole notes.
        """
        if beats >= 3.5:
            return Duration.WHOLE
        elif beats >= 1.5:
            return Duration.HALF
        elif beats >= 0.75:
            return Duration.QUARTER
        elif beats >= 0.375:
            return Duration.EIGHTH
        else:
            return Duration.SIXTEENTH


@dataclass
class Note:
    midi_pitch: int  # 0-127, -1 for rest
    start_time: float  # seconds
    end_time: float  # seconds
    confidence: float = 1.0
    duration: Duration = Duration.QUARTER
    is_rest: bool = False
    # Staff notation fields
    note_name: str = ""  # e.g., "C4", "F#5"
    accidental: str = ""  # "#", "b", ""
    octave: int = 4
    # Indian notation fields
    swara: str = ""  # e.g., "Sa", "Re", "Ga"
    swara_variant: str = ""  # e.g., "komal", "tivra", "R1", "R2"
    octave_offset: int = 0  # -1 = mandra, 0 = madhya, +1 = taar

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = self.duration.value
        return d


@dataclass
class Beat:
    beat_number: int
    notes: list[Note] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "beat_number": self.beat_number,
            "notes": [n.to_dict() for n in self.notes],
        }


@dataclass
class Bar:
    bar_number: int
    beats: list[Beat] = field(default_factory=list)
    time_signature: str = "4/4"

    def to_dict(self) -> dict:
        return {
            "bar_number": self.bar_number,
            "beats": [b.to_dict() for b in self.beats],
            "time_signature": self.time_signature,
        }


@dataclass
class TranscriptionResult:
    notes: list[Note] = field(default_factory=list)
    bars: list[Bar] = field(default_factory=list)
    notation_type: NotationType = NotationType.STAFF
    tempo: float = 120.0
    time_signature: str = "4/4"
    base_pitch: int = 60
    base_pitch_name: str = "C"
    taal_name: str = ""  # Hindustani: "Teentaal", etc.
    tala_name: str = ""  # Carnatic: "Adi", etc.

    def to_dict(self) -> dict:
        return {
            "notes": [n.to_dict() for n in self.notes],
            "bars": [b.to_dict() for b in self.bars],
            "notation_type": self.notation_type.value,
            "tempo": self.tempo,
            "time_signature": self.time_signature,
            "base_pitch": self.base_pitch,
            "base_pitch_name": self.base_pitch_name,
            "taal_name": self.taal_name,
            "tala_name": self.tala_name,
        }


# --- Pitch utility constants ---

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

PITCH_TO_NOTE_NAME = {}
for midi in range(128):
    octave = (midi // 12) - 1
    name = NOTE_NAMES[midi % 12]
    PITCH_TO_NOTE_NAME[midi] = f"{name}{octave}"


def midi_to_note_name(midi: int) -> str:
    if midi < 0 or midi > 127:
        return "Rest"
    return PITCH_TO_NOTE_NAME.get(midi, f"?{midi}")


def midi_to_octave(midi: int) -> int:
    return (midi // 12) - 1


def midi_to_accidental(midi: int) -> str:
    name = NOTE_NAMES[midi % 12]
    if "#" in name:
        return "#"
    return ""


def midi_to_base_name(midi: int) -> str:
    """Return note name without octave, e.g. 'C#'."""
    return NOTE_NAMES[midi % 12]
