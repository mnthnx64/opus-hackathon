"""Core waveform synthesis: ADSR envelope, harmonic series, FM pitch bends.

Uses phase accumulation (np.cumsum(freq_array) / sr) for smooth frequency
sweeps in gamaka subtypes (kampita, jaru, sphuritam, nokku, odukkal),
meend, and andolan — avoids phase discontinuities.
"""

from __future__ import annotations

import numpy as np

from app.services.synthesis.note_generator import GeneratedNote


def midi_to_freq(midi: int) -> float:
    """Convert MIDI pitch to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def synthesize_phrase(notes: list[GeneratedNote], sr: int = 22050) -> np.ndarray:
    """Render a list of GeneratedNote into a mono audio buffer.

    Args:
        notes: Sequence of notes with ornamentation metadata.
        sr: Sample rate in Hz.

    Returns:
        Normalized float32 audio array.
    """
    if not notes:
        return np.zeros(0, dtype=np.float32)

    total_duration = max(n.end_time for n in notes)
    total_samples = int(total_duration * sr) + sr  # +1s padding
    output = np.zeros(total_samples, dtype=np.float64)

    for note in notes:
        if note.is_rest:
            continue

        start_sample = int(note.start_time * sr)
        num_samples = int((note.end_time - note.start_time) * sr)
        if num_samples <= 0:
            continue

        samples = _synthesize_note(note, num_samples, sr)

        end_sample = start_sample + len(samples)
        if end_sample > len(output):
            end_sample = len(output)
            samples = samples[: end_sample - start_sample]

        output[start_sample:end_sample] += samples

    # Trim trailing padding
    output = output[: int(total_duration * sr) + 1]

    # Normalize to 0.9 peak
    peak = np.max(np.abs(output))
    if peak > 0:
        output = output * (0.9 / peak)

    return output.astype(np.float32)


def _synthesize_note(note: GeneratedNote, num_samples: int, sr: int) -> np.ndarray:
    """Dispatch to the correct synthesis function based on ornamentation."""
    orn = note.ornamentation
    if orn in ("kampita", "gamaka"):  # gamaka is backward-compat alias
        return _synth_kampita(note, num_samples, sr)
    elif orn == "jaru":
        return _synth_jaru(note, num_samples, sr)
    elif orn == "sphuritam":
        return _synth_sphuritam(note, num_samples, sr)
    elif orn == "nokku":
        return _synth_nokku(note, num_samples, sr)
    elif orn == "odukkal":
        return _synth_odukkal(note, num_samples, sr)
    elif orn == "staccato":
        return _synth_staccato(note, num_samples, sr)
    elif orn == "falsetto":
        return _synth_falsetto(note, num_samples, sr)
    elif orn == "meend":
        return _synth_meend(note, num_samples, sr)
    elif orn == "andolan":
        return _synth_andolan(note, num_samples, sr)
    else:
        return _synth_plain(note, num_samples, sr)


# ── ADSR envelope ─────────────────────────────────────────────────────


def _adsr_envelope(
    num_samples: int,
    sr: int,
    attack: float = 0.02,
    decay: float = 0.05,
    sustain_level: float = 0.7,
    release: float = 0.03,
) -> np.ndarray:
    """Generate an ADSR amplitude envelope."""
    attack_samples = int(attack * sr)
    decay_samples = int(decay * sr)
    release_samples = int(release * sr)
    sustain_samples = max(0, num_samples - attack_samples - decay_samples - release_samples)

    envelope = np.concatenate([
        np.linspace(0, 1, attack_samples),
        np.linspace(1, sustain_level, decay_samples),
        np.full(sustain_samples, sustain_level),
        np.linspace(sustain_level, 0, release_samples),
    ])

    # Ensure exact length
    if len(envelope) > num_samples:
        envelope = envelope[:num_samples]
    elif len(envelope) < num_samples:
        envelope = np.pad(envelope, (0, num_samples - len(envelope)))

    return envelope


# ── Phase-accumulated sine ────────────────────────────────────────────


def _phase_accumulated_sine(freq_array: np.ndarray, sr: int) -> np.ndarray:
    """Generate a sine wave from an instantaneous frequency array using phase accumulation.

    This avoids phase discontinuities when the frequency changes over time.
    """
    phase = 2.0 * np.pi * np.cumsum(freq_array) / sr
    return np.sin(phase)


# ── Harmonic series ───────────────────────────────────────────────────


def _harmonic_series(
    freq_array: np.ndarray,
    sr: int,
    harmonics: list[float] | None = None,
) -> np.ndarray:
    """Sum fundamental + harmonics using phase accumulation.

    Args:
        freq_array: Instantaneous frequency at each sample.
        sr: Sample rate.
        harmonics: Relative amplitudes for harmonics [fundamental, 2nd, 3rd, ...].
                   Defaults to [1.0, 0.5, 0.25, 0.12].
    """
    if harmonics is None:
        harmonics = [1.0, 0.5, 0.25, 0.12]

    signal = np.zeros(len(freq_array), dtype=np.float64)
    for i, amp in enumerate(harmonics):
        harmonic_freq = freq_array * (i + 1)
        signal += amp * _phase_accumulated_sine(harmonic_freq, sr)

    return signal


# ── Plain synthesis ───────────────────────────────────────────────────


def _synth_plain(note: GeneratedNote, num_samples: int, sr: int) -> np.ndarray:
    """Standard tone: fundamental + 3 harmonics, ADSR envelope."""
    freq = midi_to_freq(note.midi_pitch)
    freq_array = np.full(num_samples, freq)
    signal = _harmonic_series(freq_array, sr)
    envelope = _adsr_envelope(num_samples, sr)
    return signal * envelope


# ── Kampita synthesis (formerly gamaka) ──────────────────────────────


def _synth_kampita(note: GeneratedNote, num_samples: int, sr: int) -> np.ndarray:
    """Asymmetric pitch oscillation via FM synthesis (phase accumulation).

    Classical kampita gamaka — oscillation around the target note.

    depth_cents: 30-100 cents of pitch deviation
    rate_hz: 4-8 Hz oscillation rate
    asymmetry: ±0.4 — shifts the oscillation waveform
    """
    params = note.ornamentation_params
    base_freq = midi_to_freq(note.midi_pitch)
    depth_cents = params.get("depth_cents", 50.0)
    rate_hz = params.get("rate_hz", 6.0)
    asymmetry = params.get("asymmetry", 0.0)

    t = np.arange(num_samples) / sr

    # Asymmetric oscillation: sin + asymmetry * sin^2
    osc = np.sin(2.0 * np.pi * rate_hz * t)
    osc = osc + asymmetry * (osc ** 2)
    # Normalize to [-1, 1]
    osc_max = max(np.max(np.abs(osc)), 1e-9)
    osc = osc / osc_max

    # Convert cents deviation to frequency ratio
    freq_deviation = base_freq * (2.0 ** (depth_cents * osc / 1200.0) - 1.0)
    freq_array = base_freq + freq_deviation

    signal = _harmonic_series(freq_array, sr)
    envelope = _adsr_envelope(num_samples, sr)
    return signal * envelope


# ── Jaru synthesis (sliding glide approach) ──────────────────────────


def _synth_jaru(note: GeneratedNote, num_samples: int, sr: int) -> np.ndarray:
    """Sliding glide from a neighboring pitch into the target note.

    slide_from_semitones: ±1-3 semitones away
    slide_duration_ratio: 0.1-0.25 of total duration spent sliding
    """
    params = note.ornamentation_params
    slide_semitones = params.get("slide_from_semitones", 2)
    slide_ratio = params.get("slide_duration_ratio", 0.15)

    target_freq = midi_to_freq(note.midi_pitch)
    slide_freq = midi_to_freq(note.midi_pitch + slide_semitones)

    slide_samples = max(1, int(num_samples * slide_ratio))
    hold_samples = num_samples - slide_samples

    freq_array = np.concatenate([
        np.linspace(slide_freq, target_freq, slide_samples),
        np.full(hold_samples, target_freq),
    ])

    signal = _harmonic_series(freq_array, sr)
    envelope = _adsr_envelope(num_samples, sr)
    return signal * envelope


# ── Sphuritam synthesis (grace-note attack) ──────────────────────────


def _synth_sphuritam(note: GeneratedNote, num_samples: int, sr: int) -> np.ndarray:
    """Brief grace note then snap to the main pitch.

    grace_semitones: ±1-2 semitones for the grace note
    grace_duration_ratio: 0.05-0.15 of total duration (min 10ms)
    """
    params = note.ornamentation_params
    grace_semitones = params.get("grace_semitones", 1)
    grace_ratio = params.get("grace_duration_ratio", 0.1)

    grace_freq = midi_to_freq(note.midi_pitch + grace_semitones)
    main_freq = midi_to_freq(note.midi_pitch)

    min_grace_samples = max(1, int(0.01 * sr))  # at least 10ms
    grace_samples = max(min_grace_samples, int(num_samples * grace_ratio))
    grace_samples = min(grace_samples, num_samples - 1)
    main_samples = num_samples - grace_samples

    freq_array = np.concatenate([
        np.full(grace_samples, grace_freq),
        np.full(main_samples, main_freq),
    ])

    signal = _harmonic_series(freq_array, sr)
    envelope = _adsr_envelope(num_samples, sr)
    return signal * envelope


# ── Nokku synthesis (hammer-on from below) ───────────────────────────


def _synth_nokku(note: GeneratedNote, num_samples: int, sr: int) -> np.ndarray:
    """Quick slide up from a lower pitch — like jaru but always from below.

    approach_semitones: 1-3 semitones below
    approach_duration_ratio: 0.08-0.2 of total duration
    """
    params = note.ornamentation_params
    approach_semitones = params.get("approach_semitones", 2)
    approach_ratio = params.get("approach_duration_ratio", 0.12)

    target_freq = midi_to_freq(note.midi_pitch)
    approach_freq = midi_to_freq(note.midi_pitch - approach_semitones)

    approach_samples = max(1, int(num_samples * approach_ratio))
    hold_samples = num_samples - approach_samples

    freq_array = np.concatenate([
        np.linspace(approach_freq, target_freq, approach_samples),
        np.full(hold_samples, target_freq),
    ])

    signal = _harmonic_series(freq_array, sr)
    envelope = _adsr_envelope(num_samples, sr)
    return signal * envelope


# ── Odukkal synthesis (decaying oscillation) ─────────────────────────


def _synth_odukkal(note: GeneratedNote, num_samples: int, sr: int) -> np.ndarray:
    """Oscillation that decays exponentially over time, settling on the target pitch.

    initial_depth_cents: 40-100 cents starting deviation
    decay_rate: 0.5-2.0 — higher means faster decay
    rate_hz: 3-6 Hz oscillation rate
    """
    params = note.ornamentation_params
    base_freq = midi_to_freq(note.midi_pitch)
    initial_depth = params.get("initial_depth_cents", 60.0)
    decay_rate = params.get("decay_rate", 1.0)
    rate_hz = params.get("rate_hz", 4.5)

    t = np.arange(num_samples) / sr
    duration = num_samples / sr

    # Decaying depth envelope
    depth_over_time = initial_depth * np.exp(-decay_rate * t / max(duration, 1e-9) * 5.0)

    osc = np.sin(2.0 * np.pi * rate_hz * t)
    freq_deviation = base_freq * (2.0 ** (depth_over_time * osc / 1200.0) - 1.0)
    freq_array = base_freq + freq_deviation

    signal = _harmonic_series(freq_array, sr)
    envelope = _adsr_envelope(num_samples, sr)
    return signal * envelope


# ── Staccato synthesis ────────────────────────────────────────────────


def _synth_staccato(note: GeneratedNote, num_samples: int, sr: int) -> np.ndarray:
    """Note sounds for 25-50% of duration, rest is silence."""
    params = note.ornamentation_params
    ratio = params.get("staccato_ratio", 0.35)

    sounding_samples = max(1, int(num_samples * ratio))
    freq = midi_to_freq(note.midi_pitch)
    freq_array = np.full(sounding_samples, freq)
    signal = _harmonic_series(freq_array, sr)
    envelope = _adsr_envelope(sounding_samples, sr, release=0.01)

    # Pad with silence to full duration
    full = np.zeros(num_samples, dtype=np.float64)
    full[:sounding_samples] = signal * envelope
    return full


# ── Falsetto synthesis ────────────────────────────────────────────────


def _synth_falsetto(note: GeneratedNote, num_samples: int, sr: int) -> np.ndarray:
    """Thin harmonics (weak overtones), optional octave shift."""
    params = note.ornamentation_params
    octave_up = params.get("octave_up", False)

    pitch = note.midi_pitch + (12 if octave_up else 0)
    pitch = min(127, pitch)
    freq = midi_to_freq(pitch)
    freq_array = np.full(num_samples, freq)

    # Thin harmonics — weak overtones
    signal = _harmonic_series(freq_array, sr, harmonics=[1.0, 0.1, 0.05, 0.02])
    envelope = _adsr_envelope(num_samples, sr, attack=0.03, sustain_level=0.6)
    return signal * envelope


# ── Meend synthesis ───────────────────────────────────────────────────


def _synth_meend(note: GeneratedNote, num_samples: int, sr: int) -> np.ndarray:
    """Smooth pitch slide to the next note.

    slide_start_ratio: fraction of note duration where slide begins (0.2-0.6).
    """
    params = note.ornamentation_params
    target_pitch = params.get("target_pitch", note.midi_pitch)
    slide_start_ratio = params.get("slide_start_ratio", 0.4)

    start_freq = midi_to_freq(note.midi_pitch)
    end_freq = midi_to_freq(target_pitch)

    slide_start = int(num_samples * slide_start_ratio)
    slide_length = num_samples - slide_start

    freq_array = np.full(num_samples, start_freq)
    if slide_length > 0:
        freq_array[slide_start:] = np.linspace(start_freq, end_freq, slide_length)

    signal = _harmonic_series(freq_array, sr)
    envelope = _adsr_envelope(num_samples, sr)
    return signal * envelope


# ── Andolan synthesis ─────────────────────────────────────────────────


def _synth_andolan(note: GeneratedNote, num_samples: int, sr: int) -> np.ndarray:
    """Gentle slow oscillation, subtler than gamaka.

    depth_cents: 10-30 cents
    rate_hz: 1.5-3.5 Hz
    """
    params = note.ornamentation_params
    base_freq = midi_to_freq(note.midi_pitch)
    depth_cents = params.get("depth_cents", 20.0)
    rate_hz = params.get("rate_hz", 2.5)

    t = np.arange(num_samples) / sr
    osc = np.sin(2.0 * np.pi * rate_hz * t)

    freq_deviation = base_freq * (2.0 ** (depth_cents * osc / 1200.0) - 1.0)
    freq_array = base_freq + freq_deviation

    signal = _harmonic_series(freq_array, sr)
    envelope = _adsr_envelope(num_samples, sr)
    return signal * envelope
