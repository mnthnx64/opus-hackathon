"""Staff (Western) notation converter: MIDI → note names, accidentals, durations, bars."""

from __future__ import annotations

from app.models.music_representation import (
    Note, Beat, Bar, TranscriptionResult, NotationType, Duration,
    midi_to_note_name, midi_to_octave, midi_to_accidental, midi_to_base_name,
)


def convert_to_staff(
    notes: list[Note],
    tempo: float = 120.0,
    time_signature: str = "4/4",
    beat_duration: float = 0.5,
) -> TranscriptionResult:
    """Convert raw detected notes into staff notation with bars and beats."""
    # Enrich notes with staff-specific fields
    enriched = []
    for n in notes:
        note = Note(
            midi_pitch=n.midi_pitch,
            start_time=n.start_time,
            end_time=n.end_time,
            confidence=n.confidence,
            duration=n.duration,
            is_rest=n.is_rest,
            note_name=midi_to_note_name(n.midi_pitch) if not n.is_rest else "Rest",
            accidental=midi_to_accidental(n.midi_pitch) if not n.is_rest else "",
            octave=midi_to_octave(n.midi_pitch) if not n.is_rest else 0,
        )
        enriched.append(note)

    # Insert rests for gaps > 1/8 beat
    with_rests = _insert_rests(enriched, beat_duration)

    # Quantize durations based on tempo
    quantized = _quantize_durations(with_rests, beat_duration)

    # Group into bars
    beats_per_bar = int(time_signature.split("/")[0])
    bars = _group_into_bars(quantized, beat_duration, beats_per_bar)

    return TranscriptionResult(
        notes=quantized,
        bars=bars,
        notation_type=NotationType.STAFF,
        tempo=tempo,
        time_signature=time_signature,
    )


def _insert_rests(notes: list[Note], beat_duration: float) -> list[Note]:
    """Detect gaps > 1/8 beat between notes and insert rest Note objects."""
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
                    note_name="Rest",
                )
                result.append(rest)
        result.append(note)

    return result


def _quantize_durations(notes: list[Note], beat_duration: float) -> list[Note]:
    """Recalculate durations based on actual time spans and beat_duration."""
    for note in notes:
        time_span = note.end_time - note.start_time
        beats = time_span / beat_duration if beat_duration > 0 else 1.0
        note.duration = Duration.from_beats(beats)
    return notes


def _group_into_bars(
    notes: list[Note],
    beat_duration: float,
    beats_per_bar: int,
) -> list[Bar]:
    """Group notes into bars. Each bar has beats_per_bar beats."""
    if not notes:
        return []

    bar_duration = beat_duration * beats_per_bar
    bars = []
    current_bar_num = 1
    current_bar_start = 0.0

    # Find the start of the first note
    if notes:
        current_bar_start = notes[0].start_time

    current_beats: list[Beat] = []
    current_beat_num = 1
    current_beat_start = current_bar_start

    for note in notes:
        # Check if we've moved past the current bar
        while note.start_time >= current_bar_start + bar_duration:
            # Close current bar
            if current_beats:
                bars.append(Bar(
                    bar_number=current_bar_num,
                    beats=current_beats,
                    time_signature=f"{beats_per_bar}/4",
                ))
            current_bar_num += 1
            current_bar_start += bar_duration
            current_beats = []
            current_beat_num = 1
            current_beat_start = current_bar_start

        # Check if we've moved past the current beat
        while note.start_time >= current_beat_start + beat_duration:
            current_beat_num += 1
            current_beat_start += beat_duration

        # Find or create the beat
        beat = None
        for b in current_beats:
            if b.beat_number == current_beat_num:
                beat = b
                break
        if beat is None:
            beat = Beat(beat_number=current_beat_num)
            current_beats.append(beat)

        beat.notes.append(note)

    # Close final bar
    if current_beats:
        bars.append(Bar(
            bar_number=current_bar_num,
            beats=current_beats,
            time_signature=f"{beats_per_bar}/4",
        ))

    return bars
