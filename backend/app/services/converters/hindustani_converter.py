"""Hindustani notation converter: MIDI → swaras relative to base pitch."""

from __future__ import annotations

from app.models.music_representation import (
    Note, Beat, Bar, TranscriptionResult, NotationType, Duration,
)

# Semitone offset from Sa → swara name and variant
# 0=Sa, 1=komal Re, 2=Re, 3=komal Ga, 4=Ga, 5=Ma, 6=tivra Ma,
# 7=Pa, 8=komal Dha, 9=Dha, 10=komal Ni, 11=Ni
SEMITONE_TO_SWARA: list[tuple[str, str]] = [
    ("Sa", ""),           # 0
    ("Re", "komal"),      # 1
    ("Re", ""),           # 2
    ("Ga", "komal"),      # 3
    ("Ga", ""),           # 4
    ("Ma", ""),           # 5
    ("Ma", "tivra"),      # 6
    ("Pa", ""),           # 7
    ("Dha", "komal"),     # 8
    ("Dha", ""),          # 9
    ("Ni", "komal"),      # 10
    ("Ni", ""),           # 11
]

# Default taal: Teentaal (16 beats, 4+4+4+4)
TEENTAAL_DIVISIONS = [4, 4, 4, 4]
TEENTAAL_BEATS = 16


def convert_to_hindustani(
    notes: list[Note],
    base_pitch: int = 60,
    tempo: float = 120.0,
    beat_duration: float = 0.5,
    taal_name: str = "Teentaal",
) -> TranscriptionResult:
    """Convert raw notes to Hindustani sargam notation."""
    enriched = []
    for n in notes:
        if n.is_rest:
            enriched.append(Note(
                midi_pitch=-1,
                start_time=n.start_time,
                end_time=n.end_time,
                confidence=n.confidence,
                duration=n.duration,
                is_rest=True,
                swara="-",
                swara_variant="",
                octave_offset=0,
            ))
            continue

        semitone = (n.midi_pitch - base_pitch) % 12
        octave_offset = (n.midi_pitch - base_pitch) // 12
        # Clamp to -2..+2
        octave_offset = max(-2, min(2, octave_offset))

        swara, variant = SEMITONE_TO_SWARA[semitone]

        note = Note(
            midi_pitch=n.midi_pitch,
            start_time=n.start_time,
            end_time=n.end_time,
            confidence=n.confidence,
            duration=n.duration,
            is_rest=False,
            swara=swara,
            swara_variant=variant,
            octave_offset=octave_offset,
        )
        enriched.append(note)

    # Insert rests
    with_rests = _insert_rests_hindustani(enriched, beat_duration)

    # Quantize
    for note in with_rests:
        time_span = note.end_time - note.start_time
        beats = time_span / beat_duration if beat_duration > 0 else 1.0
        note.duration = Duration.from_beats(beats)

    # Group by taal
    bars = _group_by_taal(with_rests, beat_duration, TEENTAAL_BEATS)

    return TranscriptionResult(
        notes=with_rests,
        bars=bars,
        notation_type=NotationType.HINDUSTANI,
        tempo=tempo,
        time_signature="16/4",
        base_pitch=base_pitch,
        taal_name=taal_name,
    )


def _insert_rests_hindustani(notes: list[Note], beat_duration: float) -> list[Note]:
    """Insert rests for gaps > 1/8 beat."""
    if not notes:
        return notes

    min_gap = beat_duration / 8.0
    result = []

    for i, note in enumerate(notes):
        if i > 0:
            prev_end = notes[i - 1].end_time
            gap = note.start_time - prev_end
            if gap > min_gap:
                rest_beats = gap / beat_duration
                rest = Note(
                    midi_pitch=-1,
                    start_time=prev_end,
                    end_time=note.start_time,
                    confidence=1.0,
                    duration=Duration.from_beats(rest_beats),
                    is_rest=True,
                    swara="-",
                )
                result.append(rest)
        result.append(note)

    return result


def _group_by_taal(
    notes: list[Note],
    beat_duration: float,
    beats_per_cycle: int,
) -> list[Bar]:
    """Group notes into taal cycles (bars). Each cycle has beats_per_cycle beats."""
    if not notes:
        return []

    cycle_duration = beat_duration * beats_per_cycle
    bars = []
    current_cycle = 1
    cycle_start = notes[0].start_time if notes else 0.0

    current_beats: list[Beat] = []
    current_beat_num = 1
    beat_start = cycle_start

    for note in notes:
        # Advance cycle if needed
        while note.start_time >= cycle_start + cycle_duration:
            if current_beats:
                bars.append(Bar(
                    bar_number=current_cycle,
                    beats=current_beats,
                    time_signature=f"{beats_per_cycle}/4",
                ))
            current_cycle += 1
            cycle_start += cycle_duration
            current_beats = []
            current_beat_num = 1
            beat_start = cycle_start

        # Advance beat
        while note.start_time >= beat_start + beat_duration:
            current_beat_num += 1
            beat_start += beat_duration

        # Find or create beat
        beat = None
        for b in current_beats:
            if b.beat_number == current_beat_num:
                beat = b
                break
        if beat is None:
            beat = Beat(beat_number=current_beat_num)
            current_beats.append(beat)

        beat.notes.append(note)

    # Close final cycle
    if current_beats:
        bars.append(Bar(
            bar_number=current_cycle,
            beats=current_beats,
            time_signature=f"{beats_per_cycle}/4",
        ))

    return bars
