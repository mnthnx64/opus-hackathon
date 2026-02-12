"""Random musical phrase generation with weighted intervals and durations."""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class GeneratedNote:
    """A single generated note with synthesis metadata."""
    midi_pitch: int              # MIDI pitch (0-127), -1 for rest
    start_time: float            # seconds
    end_time: float              # seconds
    duration_type: str           # "quarter", "eighth", "half", "sixteenth", "whole"
    ornamentation: str = "plain"
    ornamentation_params: dict = field(default_factory=dict)
    is_rest: bool = False


# Duration name → beats multiplier
DURATION_BEATS: dict[str, float] = {
    "whole": 4.0,
    "half": 2.0,
    "quarter": 1.0,
    "eighth": 0.5,
    "sixteenth": 0.25,
}

# Weighted duration choices
DURATION_CHOICES = ["quarter", "eighth", "half", "sixteenth", "whole"]
DURATION_WEIGHTS = [25, 25, 20, 15, 15]


def generate_phrase(
    num_notes: int,
    base_pitch: int,
    allowed_semitones: list[int],
    tempo: float = 120.0,
    rest_probability: float = 0.1,
    octave_range: tuple[int, int] = (-1, 1),
    duration_weights: list[int] | None = None,
    timing_jitter: float = 0.0,
) -> list[GeneratedNote]:
    """Generate a random musical phrase.

    Args:
        num_notes: Number of notes (including rests) to generate.
        base_pitch: MIDI pitch of Sa/tonic (e.g. 60 for middle C).
        allowed_semitones: Semitone offsets (0-11) to choose from.
        tempo: Beats per minute.
        rest_probability: Chance of a rest instead of a note.
        octave_range: (min_octave_offset, max_octave_offset) relative to base.
        duration_weights: Custom weights for duration choices (5 ints matching
            DURATION_CHOICES). Defaults to DURATION_WEIGHTS if None.
        timing_jitter: Max jitter as fraction of beat duration (0.0-1.0).
            Applied as random ±offset to each note boundary.

    Returns:
        List of GeneratedNote with start/end times calculated from tempo.
    """
    weights = duration_weights if duration_weights is not None else DURATION_WEIGHTS
    beat_duration = 60.0 / tempo  # seconds per beat
    current_time = 0.0
    notes: list[GeneratedNote] = []
    prev_semitone_idx = random.randint(0, len(allowed_semitones) - 1)

    for _ in range(num_notes):
        # Pick duration
        dur_type = random.choices(DURATION_CHOICES, weights=weights, k=1)[0]
        dur_beats = DURATION_BEATS[dur_type]
        dur_seconds = dur_beats * beat_duration

        # Decide rest vs note
        if random.random() < rest_probability:
            notes.append(GeneratedNote(
                midi_pitch=-1,
                start_time=current_time,
                end_time=current_time + dur_seconds,
                duration_type=dur_type,
                is_rest=True,
            ))
            current_time += dur_seconds
            continue

        # Pick semitone — prefer stepwise motion
        semitone_idx = _pick_stepwise(prev_semitone_idx, len(allowed_semitones))
        semitone = allowed_semitones[semitone_idx]
        prev_semitone_idx = semitone_idx

        # Pick octave offset
        octave_offset = random.randint(octave_range[0], octave_range[1])
        midi_pitch = base_pitch + semitone + (octave_offset * 12)
        midi_pitch = max(0, min(127, midi_pitch))

        notes.append(GeneratedNote(
            midi_pitch=midi_pitch,
            start_time=current_time,
            end_time=current_time + dur_seconds,
            duration_type=dur_type,
        ))
        current_time += dur_seconds

    if timing_jitter > 0.0:
        _apply_timing_jitter(notes, beat_duration, timing_jitter)

    return notes


def _pick_stepwise(current_idx: int, num_options: int) -> int:
    """Pick a new index preferring small steps from current_idx.

    Weights: same=1, ±1=4, ±2=2, else=1. Produces more melodic lines.
    """
    weights = []
    for i in range(num_options):
        dist = abs(i - current_idx)
        if dist == 0:
            weights.append(1)
        elif dist == 1:
            weights.append(4)
        elif dist == 2:
            weights.append(2)
        else:
            weights.append(1)
    return random.choices(range(num_options), weights=weights, k=1)[0]


def _apply_timing_jitter(
    notes: list[GeneratedNote],
    beat_duration: float,
    max_jitter: float,
) -> None:
    """Apply micro-timing jitter to note boundaries in-place.

    Each note's start/end is offset by a random amount within
    ±max_jitter * beat_duration. Overlaps are prevented by clamping
    start_time >= previous note's end_time, and start_time >= 0.
    """
    max_offset = max_jitter * beat_duration

    for i, note in enumerate(notes):
        start_offset = random.uniform(-max_offset, max_offset)
        end_offset = random.uniform(-max_offset, max_offset)

        new_start = note.start_time + start_offset
        new_end = note.end_time + end_offset

        # Clamp start >= 0
        new_start = max(0.0, new_start)

        # Clamp start >= previous note's end to prevent overlaps
        if i > 0:
            new_start = max(new_start, notes[i - 1].end_time)

        # Ensure end > start (minimum 1ms)
        new_end = max(new_end, new_start + 0.001)

        note.start_time = round(new_start, 6)
        note.end_time = round(new_end, 6)
