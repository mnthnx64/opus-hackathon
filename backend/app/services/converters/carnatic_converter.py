"""Carnatic notation converter: MIDI → Carnatic swaras with R1/R2/R3 variants."""

from __future__ import annotations

from app.models.music_representation import (
    Note, Beat, Bar, TranscriptionResult, NotationType, Duration,
)

# Carnatic semitone mapping with variant numbers
# 0=Sa, 1=R1, 2=R2/G1, 3=R3/G2, 4=G3, 5=M1, 6=M2,
# 7=Pa, 8=D1, 9=D2/N1, 10=D3/N2, 11=N3
SEMITONE_TO_CARNATIC: list[tuple[str, str]] = [
    ("Sa", ""),      # 0
    ("Ri", "1"),     # 1  (Shuddha Rishabham)
    ("Ri", "2"),     # 2  (Chatushruti Rishabham)
    ("Ga", "2"),     # 3  (Sadharana Gandharam)
    ("Ga", "3"),     # 4  (Antara Gandharam)
    ("Ma", "1"),     # 5  (Shuddha Madhyamam)
    ("Ma", "2"),     # 6  (Prati Madhyamam)
    ("Pa", ""),      # 7
    ("Dha", "1"),    # 8  (Shuddha Dhaivatham)
    ("Dha", "2"),    # 9  (Chatushruti Dhaivatham)
    ("Ni", "2"),     # 10 (Kaisiki Nishadham)
    ("Ni", "3"),     # 11 (Kakali Nishadham)
]

# Default tala: Adi tala (8 beats: 4+2+2)
ADI_TALA_BEATS = 8


def convert_to_carnatic(
    notes: list[Note],
    base_pitch: int = 60,
    tempo: float = 120.0,
    beat_duration: float = 0.5,
    tala_name: str = "Adi",
) -> TranscriptionResult:
    """Convert raw notes to Carnatic svara notation."""
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
                swara=",",  # comma = silence in Carnatic
                swara_variant="",
                octave_offset=0,
            ))
            continue

        semitone = (n.midi_pitch - base_pitch) % 12
        octave_offset = (n.midi_pitch - base_pitch) // 12
        octave_offset = max(-2, min(2, octave_offset))

        swara, variant = SEMITONE_TO_CARNATIC[semitone]

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

    # Insert silences
    with_silences = _insert_silences(enriched, beat_duration)

    # Quantize
    for note in with_silences:
        time_span = note.end_time - note.start_time
        beats = time_span / beat_duration if beat_duration > 0 else 1.0
        note.duration = Duration.from_beats(beats)

    # Group by tala
    bars = _group_by_tala(with_silences, beat_duration, ADI_TALA_BEATS)

    return TranscriptionResult(
        notes=with_silences,
        bars=bars,
        notation_type=NotationType.CARNATIC,
        tempo=tempo,
        time_signature=f"{ADI_TALA_BEATS}/4",
        base_pitch=base_pitch,
        tala_name=tala_name,
    )


def _insert_silences(notes: list[Note], beat_duration: float) -> list[Note]:
    """Insert silence markers (comma) for gaps > 1/8 beat."""
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
                    swara=",",
                )
                result.append(rest)
        result.append(note)

    return result


def _group_by_tala(
    notes: list[Note],
    beat_duration: float,
    beats_per_cycle: int,
) -> list[Bar]:
    """Group notes into tala cycles (avartanam)."""
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

        while note.start_time >= beat_start + beat_duration:
            current_beat_num += 1
            beat_start += beat_duration

        beat = None
        for b in current_beats:
            if b.beat_number == current_beat_num:
                beat = b
                break
        if beat is None:
            beat = Beat(beat_number=current_beat_num)
            current_beats.append(beat)

        beat.notes.append(note)

    if current_beats:
        bars.append(Bar(
            bar_number=current_cycle,
            beats=current_beats,
            time_signature=f"{beats_per_cycle}/4",
        ))

    return bars
