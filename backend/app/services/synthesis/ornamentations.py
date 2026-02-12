"""Apply random ornamentations to generated note sequences.

Ten types: plain, 5 gamaka subtypes (kampita, jaru, sphuritam, nokku, odukkal),
staccato, falsetto, meend, andolan.
All use ADSR envelopes and phase-accumulation for smooth pitch bends.
"""

from __future__ import annotations

import random

from app.services.synthesis.note_generator import GeneratedNote

ORNAMENTATION_TYPES = [
    "plain",
    "kampita", "jaru", "sphuritam", "nokku", "odukkal",  # 5 gamaka subtypes
    "staccato", "falsetto", "meend", "andolan",
]


def apply_random_ornamentations(
    notes: list[GeneratedNote],
    probability: float = 0.3,
) -> list[GeneratedNote]:
    """Iterate through notes and randomly assign ornamentations.

    Args:
        notes: List of GeneratedNote (all starting as plain).
        probability: Chance of applying a non-plain ornamentation to each note.

    Returns:
        The same list with ornamentation fields updated in-place.
    """
    for i, note in enumerate(notes):
        if note.is_rest:
            continue

        if random.random() > probability:
            # Keep as plain
            continue

        # Pick a random ornamentation (excluding plain)
        orn_type = random.choice(ORNAMENTATION_TYPES[1:])  # skip "plain"

        # Meend needs a next note to slide to
        if orn_type == "meend":
            next_note = _find_next_pitched(notes, i)
            if next_note is None:
                # Last note or no pitched note follows — fall back to plain
                continue
            note.ornamentation = "meend"
            note.ornamentation_params = {
                "target_pitch": next_note.midi_pitch,
                "slide_start_ratio": round(random.uniform(0.2, 0.6), 2),
            }

        elif orn_type == "kampita":
            note.ornamentation = "kampita"
            note.ornamentation_params = {
                "depth_cents": round(random.uniform(30, 100), 1),
                "rate_hz": round(random.uniform(4.0, 8.0), 1),
                "asymmetry": round(random.uniform(-0.4, 0.4), 2),
            }

        elif orn_type == "jaru":
            note.ornamentation = "jaru"
            note.ornamentation_params = {
                "slide_from_semitones": random.choice([-3, -2, -1, 1, 2, 3]),
                "slide_duration_ratio": round(random.uniform(0.1, 0.25), 2),
            }

        elif orn_type == "sphuritam":
            note.ornamentation = "sphuritam"
            note.ornamentation_params = {
                "grace_semitones": random.choice([-2, -1, 1, 2]),
                "grace_duration_ratio": round(random.uniform(0.05, 0.15), 2),
            }

        elif orn_type == "nokku":
            note.ornamentation = "nokku"
            note.ornamentation_params = {
                "approach_semitones": random.randint(1, 3),
                "approach_duration_ratio": round(random.uniform(0.08, 0.2), 2),
            }

        elif orn_type == "odukkal":
            note.ornamentation = "odukkal"
            note.ornamentation_params = {
                "initial_depth_cents": round(random.uniform(40, 100), 1),
                "decay_rate": round(random.uniform(0.5, 2.0), 2),
                "rate_hz": round(random.uniform(3.0, 6.0), 1),
            }

        elif orn_type == "staccato":
            note.ornamentation = "staccato"
            note.ornamentation_params = {
                "staccato_ratio": round(random.uniform(0.25, 0.5), 2),
            }

        elif orn_type == "falsetto":
            note.ornamentation = "falsetto"
            note.ornamentation_params = {
                "octave_up": random.random() < 0.5,
            }

        elif orn_type == "andolan":
            note.ornamentation = "andolan"
            note.ornamentation_params = {
                "depth_cents": round(random.uniform(10, 30), 1),
                "rate_hz": round(random.uniform(1.5, 3.5), 1),
            }

    return notes


def _find_next_pitched(notes: list[GeneratedNote], current_idx: int) -> GeneratedNote | None:
    """Find the next non-rest note after current_idx."""
    for j in range(current_idx + 1, len(notes)):
        if not notes[j].is_rest:
            return notes[j]
    return None
