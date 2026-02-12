#!/usr/bin/env python
"""Generate synthetic audio-notation pairs for Sangeet pipeline testing.

Usage:
    python generate_dataset.py --num-notes 8 --base-pitch 60 \
        --notes-to-use "Sa Re Ga Ma Pa Dha Ni" --play
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys

import numpy as np
import soundfile as sf

# Allow imports from the backend package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.models.music_representation import Note
from app.services.converters.staff_converter import convert_to_staff
from app.services.converters.hindustani_converter import convert_to_hindustani
from app.services.converters.carnatic_converter import convert_to_carnatic
from app.services.synthesis.sargam_parser import parse_notes_to_use
from app.services.synthesis.note_generator import generate_phrase
from app.services.synthesis.ornamentations import apply_random_ornamentations
from app.services.synthesis.audio_synth import synthesize_phrase


def build_ground_truth_notes(generated_notes) -> list[Note]:
    """Convert GeneratedNote list to IMR Note list using un-ornamented pitches."""
    imr_notes = []
    for gn in generated_notes:
        imr_notes.append(Note(
            midi_pitch=gn.midi_pitch,
            start_time=gn.start_time,
            end_time=gn.end_time,
            confidence=1.0,
            is_rest=gn.is_rest,
        ))
    return imr_notes


def generate_notation(imr_notes: list[Note], base_pitch: int, tempo: float) -> dict:
    """Run all three converters on the ground-truth notes and return combined dict."""
    beat_duration = 60.0 / tempo

    staff_result = convert_to_staff(
        imr_notes, tempo=tempo, beat_duration=beat_duration,
    )
    hindustani_result = convert_to_hindustani(
        imr_notes, base_pitch=base_pitch, tempo=tempo, beat_duration=beat_duration,
    )
    carnatic_result = convert_to_carnatic(
        imr_notes, base_pitch=base_pitch, tempo=tempo, beat_duration=beat_duration,
    )

    return {
        "staff": staff_result.to_dict(),
        "hindustani": hindustani_result.to_dict(),
        "carnatic": carnatic_result.to_dict(),
    }


def build_metadata(
    generated_notes,
    args: argparse.Namespace,
    sample_idx: int,
) -> dict:
    """Build metadata dict recording generation parameters and ornamentations."""
    notes_meta = []
    for gn in generated_notes:
        entry = {
            "midi_pitch": gn.midi_pitch,
            "start_time": round(gn.start_time, 4),
            "end_time": round(gn.end_time, 4),
            "duration_type": gn.duration_type,
            "is_rest": gn.is_rest,
            "ornamentation": gn.ornamentation,
        }
        if gn.ornamentation_params:
            entry["ornamentation_params"] = gn.ornamentation_params
        notes_meta.append(entry)

    return {
        "sample_index": sample_idx,
        "generation_params": {
            "num_notes": args.num_notes,
            "base_pitch": args.base_pitch,
            "notes_to_use": args.notes_to_use,
            "tempo": args.tempo,
            "sample_rate": args.sample_rate,
            "gamaka_probability": args.gamaka_probability,
            "no_gamaka": args.no_gamaka,
            "timing_jitter": args.timing_jitter,
            "duration_weights": args.duration_weights,
        },
        "notes": notes_meta,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic audio-notation pairs for Sangeet testing.",
    )
    parser.add_argument("--num-notes", type=int, required=True, help="Number of notes per phrase")
    parser.add_argument("--base-pitch", type=int, required=True, help="MIDI pitch of Sa/tonic (e.g. 60)")
    parser.add_argument("--notes-to-use", type=str, default=None,
                        help='Space-separated sargam tokens or raga name (e.g. "Sa Re Ga Ma Pa Dha Ni" or "yaman")')
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--output-dir", type=str, default="../dataset", help="Output directory")
    parser.add_argument("--tempo", type=float, default=120.0, help="Tempo in BPM")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Audio sample rate")
    parser.add_argument("--play", action="store_true", help="Play the first generated sample")
    parser.add_argument("--duration-weights", type=str, default=None,
                        help='Comma-separated 5 ints for duration weights, e.g. "25,25,20,15,15"')
    parser.add_argument("--timing-jitter", type=float, default=0.0,
                        help="Max timing jitter as fraction of beat (0.0-1.0)")
    parser.add_argument("--gamaka-probability", type=float, default=0.3,
                        help="Probability of ornamentation per note (0.0-1.0)")
    parser.add_argument("--no-gamaka", action="store_true",
                        help="Disable all ornamentations (plain notes only)")

    args = parser.parse_args()

    # Parse duration weights if provided
    duration_weights = None
    if args.duration_weights:
        duration_weights = [int(x.strip()) for x in args.duration_weights.split(",")]
        if len(duration_weights) != 5:
            parser.error("--duration-weights must have exactly 5 comma-separated integers")
    args.duration_weights_parsed = duration_weights

    # Parse allowed semitones
    allowed_semitones = parse_notes_to_use(args.notes_to_use)
    print(f"Allowed semitones: {allowed_semitones}")

    # Resolve output directory relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.normpath(os.path.join(script_dir, args.output_dir))
    first_wav_path = None

    for i in range(args.num_samples):
        sample_dir = os.path.join(output_dir, f"sample_{i + 1:03d}")
        os.makedirs(sample_dir, exist_ok=True)

        # 1. Generate phrase (un-ornamented)
        notes = generate_phrase(
            num_notes=args.num_notes,
            base_pitch=args.base_pitch,
            allowed_semitones=allowed_semitones,
            tempo=args.tempo,
            duration_weights=args.duration_weights_parsed,
            timing_jitter=args.timing_jitter,
        )

        # 2. Build ground truth from un-ornamented pitches
        imr_notes = build_ground_truth_notes(notes)
        notation = generate_notation(imr_notes, args.base_pitch, args.tempo)

        # 3. Apply ornamentations (modifies notes in-place)
        if not args.no_gamaka:
            apply_random_ornamentations(notes, probability=args.gamaka_probability)

        # 4. Synthesize audio from ornamented notes
        audio = synthesize_phrase(notes, sr=args.sample_rate)

        # 5. Write outputs
        wav_path = os.path.join(sample_dir, "audio.wav")
        sf.write(wav_path, audio, args.sample_rate)

        notation_path = os.path.join(sample_dir, "notation.json")
        with open(notation_path, "w") as f:
            json.dump(notation, f, indent=2)

        metadata = build_metadata(notes, args, i + 1)
        metadata_path = os.path.join(sample_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[{i + 1}/{args.num_samples}] Written to {sample_dir}")

        if i == 0:
            first_wav_path = wav_path

    # Play first sample if requested
    if args.play and first_wav_path:
        system = platform.system()
        if system == "Darwin":
            cmd = ["afplay", first_wav_path]
        elif system == "Linux":
            cmd = ["aplay", first_wav_path]
        else:
            print(f"Playback not supported on {system}")
            return

        print(f"Playing {first_wav_path} ...")
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
