# Musical Audio → Notation Web App: Complete Project Plan

## Executive Summary

This document provides a comprehensive, step-by-step project plan for building a web application that translates musical audio into multiple notation systems (Staff, Hindustani Sargam, Carnatic Sargam). The plan is broken into **8 phases with 32 discrete tasks**, each scoped so an AI agent or developer can complete them independently.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    FRONTEND (React/Next.js)              │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌────────┐ │
│  │  Audio    │  │  Live    │  │ Notation  │  │  PDF   │ │
│  │  Upload   │  │  Capture │  │ Renderer  │  │ Export │ │
│  └────┬─────┘  └────┬─────┘  └─────┬─────┘  └───┬────┘ │
│       │              │              │             │      │
│  ┌────▼──────────────▼──────────────▼─────────────▼────┐ │
│  │              State Management (Zustand)              │ │
│  └──────────────────────┬──────────────────────────────┘ │
└─────────────────────────┼───────────────────────────────┘
                          │ REST / WebSocket
┌─────────────────────────▼───────────────────────────────┐
│                   BACKEND (Python FastAPI)                │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │ Audio        │  │ Tempo/Beat   │  │ Notation       │ │
│  │ Transcriber  │  │ Analyzer     │  │ Translator     │ │
│  │ (Basic Pitch │  │ (librosa)    │  │ (MIDI→Sargam/  │ │
│  │  + custom)   │  │              │  │  Staff/Carnatic)│ │
│  └──────────────┘  └──────────────┘  └────────────────┘ │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │ OMR Engine   │  │ PDF          │  │ WebSocket      │ │
│  │ (oemer/      │  │ Generator    │  │ Live Stream    │ │
│  │  custom)     │  │ (ReportLab)  │  │ Handler        │ │
│  └──────────────┘  └──────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **Frontend** | Next.js 14+ (React) + TypeScript | SSR, file-based routing, excellent DX |
| **State** | Zustand | Lightweight, no boilerplate |
| **Audio Capture** | Web Audio API + AudioWorklet | Real-time, low-latency browser audio |
| **Notation Rendering** | VexFlow 5 (staff) + Custom SVG (Indian) | Industry standard for western notation; no library exists for Indian notation rendering |
| **Backend** | Python 3.11+ FastAPI | Async, fast, ML ecosystem compatibility |
| **Pitch Detection** | Spotify's Basic Pitch (ONNX) | Lightweight (~17K params), polyphonic, instrument-agnostic, Apache 2.0 licensed |
| **Tempo/Beat** | librosa | Industry-standard beat tracking & tempo estimation |
| **OMR** | oemer + custom fine-tuned model | Open-source, end-to-end, outputs MusicXML |
| **PDF Export** | ReportLab + SVG embedding | Programmatic PDF generation with vector graphics |
| **Real-time Comm** | WebSockets (FastAPI) | Bidirectional streaming for live mode |
| **ML Training** | PyTorch | Largest research ecosystem, best for custom models |
| **Datasets** | MAESTRO v3, MusicNet, GuitarSet, Slakh2100 | Paired audio+MIDI, multi-instrument coverage |

---

## Phase 1: Project Setup & Core Infrastructure

### Task 1.1: Repository & Development Environment Setup

**Goal:** Mono-repo with frontend and backend, CI/CD, linting, Docker.

**What to build:**
- Initialize a mono-repo structure:
  ```
  /
  ├── frontend/          # Next.js app
  │   ├── src/
  │   │   ├── app/       # App router pages
  │   │   ├── components/
  │   │   ├── lib/       # Utility functions
  │   │   ├── hooks/     # Custom React hooks
  │   │   └── stores/    # Zustand stores
  │   ├── public/
  │   ├── package.json
  │   └── tsconfig.json
  ├── backend/           # FastAPI app
  │   ├── app/
  │   │   ├── api/       # Route handlers
  │   │   ├── core/      # Config, dependencies
  │   │   ├── models/    # Pydantic models
  │   │   ├── services/  # Business logic
  │   │   └── ml/        # ML model loading & inference
  │   ├── tests/
  │   ├── requirements.txt
  │   └── Dockerfile
  ├── ml/                # ML training scripts
  │   ├── data/
  │   ├── models/
  │   ├── training/
  │   └── evaluation/
  ├── shared/            # Shared types/contracts
  ├── docker-compose.yml
  └── README.md
  ```
- Set up Docker Compose with services: `frontend`, `backend`, `redis` (for job queue)
- Configure ESLint, Prettier (frontend), Black, Ruff (backend)
- Set up pre-commit hooks

**Deliverables:** Working dev environment with `docker-compose up` starting all services.

---

### Task 1.2: Backend API Skeleton

**Goal:** FastAPI app with health checks, CORS, file upload endpoint, WebSocket endpoint.

**What to build:**
```python
# backend/app/main.py
from fastapi import FastAPI, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Music Transcription API")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile, base_pitch: str = "C4",
                           notation_type: str = "staff"):
    # Placeholder - will be implemented in Phase 3
    pass

@app.websocket("/ws/live")
async def live_transcription(websocket: WebSocket):
    # Placeholder - will be implemented in Phase 5
    pass

@app.post("/api/translate-notation")
async def translate_notation(file: UploadFile, source_type: str = "staff",
                             target_type: str = "hindustani"):
    # Placeholder - will be implemented in Phase 7
    pass
```

**Key endpoints to stub:**
1. `POST /api/transcribe` — upload audio, get notation
2. `POST /api/translate-notation` — upload notation image, get other notation
3. `GET /api/export/pdf` — export notation as PDF
4. `WS /ws/live` — live audio streaming
5. `GET /api/config/pitches` — available base pitches
6. `GET /api/config/notations` — available notation types

**Deliverables:** Running FastAPI with Swagger docs at `/docs`.

---

### Task 1.3: Frontend Shell & Routing

**Goal:** Next.js app with page routing, layout, and placeholder UI.

**What to build:**
- Pages: `/` (home/upload), `/live` (live mode), `/results` (notation display), `/translate` (notation translation), `/editor` (interactive notation editor & playback)
- Shared layout with navigation header
- Zustand store skeleton:
  ```typescript
  // stores/transcriptionStore.ts
  interface TranscriptionState {
    audioFile: File | null;
    basePitch: string;        // e.g., "C4", "D4"
    notationType: 'staff' | 'hindustani' | 'carnatic';
    transcriptionResult: TranscriptionResult | null;
    isProcessing: boolean;
    confidence: number[];
    tempo: number | null;
    setAudioFile: (file: File) => void;
    setBasePitch: (pitch: string) => void;
    setNotationType: (type: string) => void;
  }
  ```
- Base pitch selector component (dropdown of all 12 chromatic pitches × octaves 2-6)
- Notation type selector component (Staff, Hindustani, Carnatic)

**Deliverables:** Navigable frontend shell with controls wired to state.

---

## Phase 2: Audio Processing Pipeline

### Task 2.1: Audio File Upload & Preprocessing

**Goal:** Accept audio uploads (WAV, MP3, FLAC, OGG, M4A), validate, normalize, and convert to the format required by the ML model.

**What to build (backend):**
```python
# backend/app/services/audio_processor.py
import librosa
import numpy as np
import soundfile as sf

class AudioProcessor:
    SUPPORTED_FORMATS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    TARGET_SR = 22050  # Basic Pitch expects 22050 Hz

    def validate_and_load(self, file_path: str) -> tuple[np.ndarray, int]:
        """Load audio, convert to mono, resample to 22050 Hz."""
        y, sr = librosa.load(file_path, sr=self.TARGET_SR, mono=True)
        return y, sr

    def normalize_audio(self, y: np.ndarray) -> np.ndarray:
        """Peak normalize to prevent clipping issues."""
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak
        return y

    def detect_silence(self, y: np.ndarray, sr: int,
                       threshold_db: float = -40) -> list[tuple[float, float]]:
        """Detect silence regions. Returns list of (start_sec, end_sec)."""
        intervals = librosa.effects.split(y, top_db=abs(threshold_db))
        # Invert to get silence regions
        silences = []
        if intervals[0][0] > 0:
            silences.append((0.0, intervals[0][0] / sr))
        for i in range(len(intervals) - 1):
            silences.append((intervals[i][1] / sr, intervals[i+1][0] / sr))
        return silences
```

**Frontend upload component:**
- Drag-and-drop zone with file type validation
- Audio waveform preview using WaveSurfer.js or custom canvas
- Upload progress bar
- File size limit: 50MB

**Deliverables:** Audio upload → preprocessing pipeline working end-to-end.

---

### Task 2.2: Tempo & Beat Detection

**Goal:** Detect BPM and beat positions using librosa. Report confidence. Handle "no tempo detected" case.

**What to build:**
```python
# backend/app/services/tempo_analyzer.py
import librosa
import numpy as np

class TempoAnalyzer:
    def analyze(self, y: np.ndarray, sr: int) -> dict:
        """
        Returns:
        {
            "tempo_detected": bool,
            "bpm": float | None,
            "confidence": float,   # 0.0-1.0
            "beat_times": list[float],  # seconds
            "downbeat_times": list[float],
            "time_signature": str | None  # "4/4", "3/4", etc.
        }
        """
        # Onset strength envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)

        # Tempo estimation with prior
        tempo = librosa.feature.tempo(
            onset_envelope=onset_env, sr=sr, aggregate=None
        )

        # Beat tracking
        bpm, beat_frames = librosa.beat.beat_track(
            y=y, sr=sr, onset_envelope=onset_env
        )
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        # Confidence: strength of autocorrelation at detected tempo
        ac = librosa.autocorrelate(onset_env, max_size=len(onset_env))
        confidence = self._compute_tempo_confidence(ac, bpm, sr)

        # If confidence < threshold, report "can't detect"
        CONFIDENCE_THRESHOLD = 0.3
        if confidence < CONFIDENCE_THRESHOLD:
            return {
                "tempo_detected": False,
                "bpm": None,
                "confidence": confidence,
                "beat_times": [],
                "downbeat_times": [],
                "time_signature": None
            }

        # Time signature estimation (heuristic)
        time_sig = self._estimate_time_signature(onset_env, beat_frames, sr)

        return {
            "tempo_detected": True,
            "bpm": float(bpm),
            "confidence": float(confidence),
            "beat_times": beat_times.tolist(),
            "downbeat_times": beat_times[::4].tolist(),  # rough downbeats
            "time_signature": time_sig
        }

    def _compute_tempo_confidence(self, ac, bpm, sr):
        """Compute how strong the periodic signal is at the detected BPM."""
        if bpm == 0:
            return 0.0
        period_frames = int(60.0 / bpm * sr / 512)  # hop_length=512
        if period_frames < len(ac):
            peak = ac[period_frames]
            return float(min(peak / (ac[0] + 1e-8), 1.0))
        return 0.0

    def _estimate_time_signature(self, onset_env, beat_frames, sr):
        """Heuristic time signature estimation."""
        # This is a simplified heuristic. For production,
        # use a dedicated time signature classifier.
        if len(beat_frames) < 8:
            return None
        # Analyze accent patterns in onset strength at beat positions
        beat_strengths = onset_env[beat_frames[:min(32, len(beat_frames))]]
        # Simple heuristic: if every 3rd beat is stronger → 3/4
        # if every 4th → 4/4
        pattern_3 = np.mean(beat_strengths[::3]) / (np.mean(beat_strengths) + 1e-8)
        pattern_4 = np.mean(beat_strengths[::4]) / (np.mean(beat_strengths) + 1e-8)
        if pattern_3 > pattern_4 * 1.2:
            return "3/4"
        return "4/4"
```

**Deliverables:** Tempo analysis service returning structured data with confidence.

---

### Task 2.3: Pitch Detection with Basic Pitch

**Goal:** Integrate Spotify's Basic Pitch for polyphonic note detection from audio. Extract MIDI note events with onset, offset, pitch, velocity, and confidence.

**What to build:**
```python
# backend/app/services/pitch_detector.py
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
import numpy as np

class PitchDetector:
    def __init__(self):
        # Load model once at startup
        import tensorflow as tf
        self.model = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))

    def detect(self, audio_path: str,
               onset_threshold: float = 0.5,
               frame_threshold: float = 0.3,
               min_note_length_ms: float = 58.0,
               min_freq: float = None,
               max_freq: float = None) -> dict:
        """
        Returns:
        {
            "notes": [
                {
                    "start_time": float,   # seconds
                    "end_time": float,     # seconds
                    "midi_pitch": int,     # 0-127
                    "frequency": float,    # Hz
                    "velocity": int,       # 0-127
                    "confidence": float,   # 0.0-1.0
                    "pitch_bend": list     # pitch bend data if available
                }
            ],
            "midi_data": PrettyMIDI,  # full MIDI object
            "raw_posteriors": np.ndarray  # frame-level confidence matrix
        }
        """
        model_output, midi_data, note_events = predict(
            audio_path,
            self.model,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=min_note_length_ms,
            minimum_frequency=min_freq,
            maximum_frequency=max_freq,
        )

        notes = []
        for onset, offset, pitch, velocity, pitch_bends in note_events:
            freq = librosa.midi_to_hz(pitch)
            # Confidence from model output posteriors
            frame_start = int(onset * 22050 / 256)
            frame_end = int(offset * 22050 / 256)
            if frame_end <= model_output['note'].shape[0]:
                conf = float(np.mean(
                    model_output['note'][frame_start:frame_end, int(pitch) - 21]
                ))
            else:
                conf = float(velocity / 127.0)

            notes.append({
                "start_time": float(onset),
                "end_time": float(offset),
                "midi_pitch": int(pitch),
                "frequency": float(freq),
                "velocity": int(velocity),
                "confidence": round(conf, 3),
                "pitch_bend": pitch_bends if pitch_bends else []
            })

        return {
            "notes": notes,
            "midi_data": midi_data,
            "raw_posteriors": model_output
        }
```

**Installation:** `pip install basic-pitch[tf]`

**Deliverables:** Pitch detection service returning structured note events with confidence scores.

---

## Phase 3: Core Notation Translation Engine

### Task 3.1: Internal Music Representation (IMR)

**Goal:** Define a universal internal data structure that can represent notes in any notation system (Staff, Hindustani, Carnatic). This is the central data model everything converts to and from.

**What to build:**
```python
# backend/app/models/music_representation.py
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class NotationType(str, Enum):
    STAFF = "staff"
    HINDUSTANI = "hindustani"
    CARNATIC = "carnatic"

class NoteValue(str, Enum):
    """Duration values"""
    WHOLE = "whole"          # 4 beats
    HALF = "half"            # 2 beats
    QUARTER = "quarter"      # 1 beat
    EIGHTH = "eighth"        # 0.5 beats
    SIXTEENTH = "sixteenth"  # 0.25 beats

@dataclass
class Note:
    midi_pitch: int               # Absolute pitch (MIDI number)
    frequency: float              # Hz
    start_time: float             # Seconds from start
    end_time: float               # Seconds
    duration_beats: float         # Duration in beats (if tempo detected)
    note_value: Optional[NoteValue]  # Quantized duration
    velocity: int                 # 0-127
    confidence: float             # 0.0-1.0
    is_rest: bool = False
    is_tied: bool = False         # Tied to next note
    dotted: bool = False          # Dotted note (1.5x duration)
    octave: int = 4              # Octave number

    # Western staff properties
    staff_name: Optional[str] = None    # "C", "D", "E", etc.
    accidental: Optional[str] = None    # "sharp", "flat", "natural"

    # Indian notation properties
    swara: Optional[str] = None         # "Sa", "Re", "Ga", etc.
    swara_variant: Optional[str] = None # For Carnatic: "R1", "R2", "R3"
    octave_marker: Optional[str] = None # "lower", "middle", "upper"

@dataclass
class Beat:
    beat_number: int
    notes: list[Note] = field(default_factory=list)

@dataclass
class Bar:
    bar_number: int
    beats: list[Beat] = field(default_factory=list)
    time_signature: str = "4/4"

@dataclass
class TranscriptionResult:
    bars: list[Bar] = field(default_factory=list)
    ungrouped_notes: list[Note] = field(default_factory=list)  # If no tempo
    tempo_detected: bool = False
    bpm: Optional[float] = None
    time_signature: Optional[str] = None
    key_signature: Optional[str] = None
    base_pitch: str = "C4"       # User-selected tonic
    notation_type: NotationType = NotationType.STAFF
    confidence_mean: float = 0.0
```

**Deliverables:** Data model module importable by all services.

---

### Task 3.2: MIDI-to-Staff Notation Converter

**Goal:** Convert detected MIDI note events into western staff notation data (note names, accidentals, durations, rests, bars).

**What to build:**
```python
# backend/app/services/converters/staff_converter.py
import librosa

class StaffConverter:
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Enharmonic preferences based on key
    SHARP_KEYS = {'C', 'G', 'D', 'A', 'E', 'B', 'F#'}
    FLAT_KEYS = {'F', 'Bb', 'Eb', 'Ab', 'Db', 'Gb'}

    def convert(self, notes: list[dict], tempo_data: dict,
                base_pitch: str = "C4") -> TranscriptionResult:
        """Convert raw note events to staff notation."""
        result = TranscriptionResult(notation_type=NotationType.STAFF)
        result.base_pitch = base_pitch
        result.bpm = tempo_data.get("bpm")
        result.tempo_detected = tempo_data.get("tempo_detected", False)
        result.time_signature = tempo_data.get("time_signature", "4/4")

        # Convert each note event to Note objects
        note_objects = []
        for n in notes:
            note_obj = self._midi_to_staff_note(n, tempo_data)
            note_objects.append(note_obj)

        # Insert rests between notes
        note_objects = self._insert_rests(note_objects, tempo_data)

        if result.tempo_detected:
            # Group into bars and beats
            result.bars = self._group_into_bars(
                note_objects, tempo_data
            )
        else:
            result.ungrouped_notes = note_objects

        return result

    def _midi_to_staff_note(self, note_event: dict,
                            tempo_data: dict) -> Note:
        """Convert a single MIDI note event to a staff Note."""
        midi_pitch = note_event["midi_pitch"]
        note_name = self.NOTE_NAMES[midi_pitch % 12]
        octave = (midi_pitch // 12) - 1

        accidental = None
        if '#' in note_name:
            note_name = note_name[0]
            accidental = "sharp"

        duration_sec = note_event["end_time"] - note_event["start_time"]
        duration_beats = None
        note_value = None

        if tempo_data.get("tempo_detected") and tempo_data.get("bpm"):
            beat_duration = 60.0 / tempo_data["bpm"]
            duration_beats = duration_sec / beat_duration
            note_value = self._quantize_duration(duration_beats)

        return Note(
            midi_pitch=midi_pitch,
            frequency=note_event["frequency"],
            start_time=note_event["start_time"],
            end_time=note_event["end_time"],
            duration_beats=duration_beats,
            note_value=note_value,
            velocity=note_event["velocity"],
            confidence=note_event["confidence"],
            staff_name=note_name,
            accidental=accidental,
            octave=octave,
        )

    def _quantize_duration(self, beats: float) -> NoteValue:
        """Snap a beat duration to the nearest standard note value."""
        thresholds = [
            (3.0, NoteValue.WHOLE),
            (1.5, NoteValue.HALF),
            (0.75, NoteValue.QUARTER),
            (0.375, NoteValue.EIGHTH),
            (0.0, NoteValue.SIXTEENTH),
        ]
        for threshold, value in thresholds:
            if beats >= threshold:
                return value
        return NoteValue.SIXTEENTH

    def _insert_rests(self, notes, tempo_data):
        """Insert rest notes in gaps between notes."""
        # ... Implementation: detect gaps > minimum rest duration
        # and insert Note(is_rest=True) objects
        pass

    def _group_into_bars(self, notes, tempo_data):
        """Group notes into bars based on beat positions."""
        # ... Implementation: use beat_times to determine bar boundaries
        pass
```

**Deliverables:** Working staff notation converter.

---

### Task 3.3: MIDI-to-Hindustani Sargam Converter

**Goal:** Convert MIDI notes to Hindustani sargam notation (Sa, Re, Ga, Ma, Pa, Dha, Ni) relative to user-selected base pitch.

**What to build:**
```python
# backend/app/services/converters/hindustani_converter.py

class HindustaniConverter:
    """
    Hindustani Music Notation Rules:
    - 7 svaras: Sa, Re, Ga, Ma, Pa, Dha, Ni
    - Sa and Pa are fixed (achala svaras)
    - Re, Ga, Dha, Ni can be "komal" (flat) — marked with underline
    - Ma can be "tivra" (sharp) — marked with vertical line above
    - Octave: dot below = lower (mandra), no dot = middle (madhya),
              dot above = upper (taar)
    - Duration: single letter = 1 beat, dash = sustained
    - Written left to right in devanagari or romanized
    """

    # Semitone offset from Sa → Swara name
    # In Bilawal thaat (equivalent to major scale / Ionian mode):
    SEMITONE_TO_SWARA = {
        0: ("Sa", None),     # Shadja - fixed
        1: ("Re", "komal"),  # Komal Rishabh
        2: ("Re", None),     # Shuddha Rishabh
        3: ("Ga", "komal"),  # Komal Gandhar
        4: ("Ga", None),     # Shuddha Gandhar
        5: ("Ma", None),     # Shuddha Madhyam
        6: ("Ma", "tivra"),  # Tivra Madhyam
        7: ("Pa", None),     # Pancham - fixed
        8: ("Dha", "komal"), # Komal Dhaivat
        9: ("Dha", None),    # Shuddha Dhaivat
        10: ("Ni", "komal"), # Komal Nishad
        11: ("Ni", None),    # Shuddha Nishad
    }

    def convert(self, notes: list[dict], tempo_data: dict,
                base_pitch: str = "C4") -> TranscriptionResult:
        """
        base_pitch: The note the user selects as "Sa".
        e.g., if base_pitch = "C4", then C = Sa, D = Re, E = Ga, etc.
        """
        base_midi = self._pitch_str_to_midi(base_pitch)
        result = TranscriptionResult(
            notation_type=NotationType.HINDUSTANI,
            base_pitch=base_pitch
        )

        note_objects = []
        for n in notes:
            # Calculate semitone offset from base pitch
            semitones = (n["midi_pitch"] - base_midi) % 12
            octave_offset = (n["midi_pitch"] - base_midi) // 12

            swara, variant = self.SEMITONE_TO_SWARA[semitones]

            # Determine octave marker
            if octave_offset < 0:
                octave_marker = "lower"   # Mandra saptak (dot below)
            elif octave_offset == 0:
                octave_marker = "middle"  # Madhya saptak (no dot)
            else:
                octave_marker = "upper"   # Taar saptak (dot above)

            note_obj = Note(
                midi_pitch=n["midi_pitch"],
                frequency=n["frequency"],
                start_time=n["start_time"],
                end_time=n["end_time"],
                duration_beats=None,  # Computed if tempo exists
                note_value=None,
                velocity=n["velocity"],
                confidence=n["confidence"],
                swara=swara,
                swara_variant=variant,  # "komal" or "tivra" or None
                octave_marker=octave_marker,
            )
            note_objects.append(note_obj)

        result.ungrouped_notes = note_objects
        if tempo_data.get("tempo_detected"):
            result.bars = self._group_by_taal(note_objects, tempo_data)

        return result

    def _pitch_str_to_midi(self, pitch_str: str) -> int:
        """Convert 'C4' → 60, 'D#3' → 51, etc."""
        import librosa
        return librosa.note_to_midi(pitch_str)

    def _group_by_taal(self, notes, tempo_data):
        """Group notes according to taal (rhythmic cycle).
        Default: Teentaal (16 beats = 4+4+4+4)"""
        # Implementation: use beat_times to create taal divisions
        pass
```

**Deliverables:** Hindustani sargam converter with komal/tivra markings and octave indicators.

---

### Task 3.4: MIDI-to-Carnatic Sargam Converter

**Goal:** Convert MIDI notes to Carnatic notation. Key differences from Hindustani: different swara names (Ri vs Re, Da vs Dha), 16 svara positions, tala system.

**What to build:**
```python
# backend/app/services/converters/carnatic_converter.py

class CarnaticConverter:
    """
    Carnatic Notation Rules:
    - 7 svaras: Sa, Ri, Ga, Ma, Pa, Da, Ni
    - Sa and Pa are fixed (prakrti svaras)
    - Ri has 3 variants (R1, R2, R3)
    - Ga has 3 variants (G1, G2, G3)
    - Ma has 2 variants (M1, M2)
    - Da has 3 variants (D1, D2, D3)
    - Ni has 3 variants (N1, N2, N3)
    - Some positions overlap (R2 = G1, R3 = G2, D2 = N1, D3 = N2)
    - Dot above = upper octave, dot below = lower octave
    - Duration: single char = 1 maathra, doubled vowel (saa) = 2 maathra
    - Comma (,) = 1 maathra silence, semicolon (;) = 2 maathra silence
    """

    SEMITONE_TO_CARNATIC = {
        0:  ("Sa", None),
        1:  ("Ri", "R1"),    # Suddha Rishabham
        2:  ("Ri", "R2"),    # Chatusruti Rishabham (= Suddha Gandharam G1)
        3:  ("Ri", "R3"),    # Shatsruti Rishabham (= Sadharana Gandharam G2)
        4:  ("Ga", "G3"),    # Antara Gandharam
        5:  ("Ma", "M1"),    # Suddha Madhyamam
        6:  ("Ma", "M2"),    # Prati Madhyamam
        7:  ("Pa", None),
        8:  ("Da", "D1"),    # Suddha Dhaivatam
        9:  ("Da", "D2"),    # Chatusruti Dhaivatam (= Suddha Nishadam N1)
        10: ("Da", "D3"),    # Shatsruti Dhaivatam (= Kaisiki Nishadam N2)
        11: ("Ni", "N3"),    # Kakali Nishadam
    }

    def convert(self, notes: list[dict], tempo_data: dict,
                base_pitch: str = "C4") -> TranscriptionResult:
        """Convert MIDI notes to Carnatic notation."""
        base_midi = librosa.note_to_midi(base_pitch)
        result = TranscriptionResult(
            notation_type=NotationType.CARNATIC,
            base_pitch=base_pitch
        )

        note_objects = []
        for n in notes:
            semitones = (n["midi_pitch"] - base_midi) % 12
            octave_offset = (n["midi_pitch"] - base_midi) // 12

            swara, variant = self.SEMITONE_TO_CARNATIC[semitones]

            octave_marker = "middle"
            if octave_offset < 0:
                octave_marker = "lower"
            elif octave_offset > 0:
                octave_marker = "upper"

            # Duration in maathras (if tempo detected)
            maathras = None
            if tempo_data.get("bpm"):
                beat_dur = 60.0 / tempo_data["bpm"]
                duration = n["end_time"] - n["start_time"]
                maathras = duration / (beat_dur / 4)  # 4 maathras per beat

            note_obj = Note(
                midi_pitch=n["midi_pitch"],
                frequency=n["frequency"],
                start_time=n["start_time"],
                end_time=n["end_time"],
                duration_beats=maathras,
                note_value=None,
                velocity=n["velocity"],
                confidence=n["confidence"],
                swara=swara,
                swara_variant=variant,
                octave_marker=octave_marker,
            )
            note_objects.append(note_obj)

        # Insert silence markers
        note_objects = self._insert_silences(note_objects)

        result.ungrouped_notes = note_objects
        if tempo_data.get("tempo_detected"):
            result.bars = self._group_by_tala(note_objects, tempo_data)

        return result

    def _insert_silences(self, notes):
        """Insert Carnatic silence markers (comma/semicolon) between notes."""
        # ... detect gaps, insert rest notes with appropriate duration
        pass

    def _group_by_tala(self, notes, tempo_data):
        """Group by Carnatic tala. Default: Adi tala (8 beats: 4+2+2)."""
        pass
```

**Deliverables:** Carnatic converter with proper 16-svara system and variant support.

---

### Task 3.5: Unified Transcription Orchestrator

**Goal:** Wire together audio processing → pitch detection → tempo analysis → notation conversion into a single pipeline.

**What to build:**
```python
# backend/app/services/transcription_orchestrator.py

class TranscriptionOrchestrator:
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.pitch_detector = PitchDetector()
        self.tempo_analyzer = TempoAnalyzer()
        self.converters = {
            "staff": StaffConverter(),
            "hindustani": HindustaniConverter(),
            "carnatic": CarnaticConverter(),
        }

    async def transcribe(self, audio_path: str, base_pitch: str = "C4",
                         notation_type: str = "staff") -> TranscriptionResult:
        # Step 1: Load and preprocess audio
        y, sr = self.audio_processor.validate_and_load(audio_path)
        y = self.audio_processor.normalize_audio(y)

        # Step 2: Detect tempo and beats
        tempo_data = self.tempo_analyzer.analyze(y, sr)

        # Step 3: Detect pitches
        pitch_data = self.pitch_detector.detect(audio_path)

        # Step 4: Convert to requested notation
        converter = self.converters[notation_type]
        result = converter.convert(
            pitch_data["notes"], tempo_data, base_pitch
        )

        # Step 5: Add metadata
        result.confidence_mean = np.mean(
            [n["confidence"] for n in pitch_data["notes"]]
        ) if pitch_data["notes"] else 0.0

        return result
```

**Deliverables:** Full pipeline from audio file → notation result.

---

## Phase 4: Frontend Notation Rendering

### Task 4.1: Staff Notation Renderer (VexFlow)

**Goal:** Render western staff notation in the browser using VexFlow 5.

**What to build:**
```typescript
// frontend/src/components/StaffRenderer.tsx
import { useEffect, useRef } from 'react';
import * as VexFlow from 'vexflow';

interface StaffRendererProps {
  bars: Bar[];           // From TranscriptionResult
  tempoDetected: boolean;
  bpm: number | null;
  timeSignature: string;
}

export function StaffRenderer({ bars, tempoDetected, bpm, timeSignature }: StaffRendererProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current || bars.length === 0) return;

    const { Factory, EasyScore, System, Stave, StaveNote,
            Formatter, Voice, Renderer } = VexFlow;

    // Clear previous render
    containerRef.current.innerHTML = '';

    const renderer = new Renderer(
      containerRef.current, Renderer.Backends.SVG
    );
    renderer.resize(1200, bars.length * 120 + 100);
    const context = renderer.getContext();

    let yPosition = 40;

    for (const bar of bars) {
      const stave = new Stave(10, yPosition, 1100);

      if (bar.bar_number === 1) {
        stave.addClef('treble');
        if (timeSignature) stave.addTimeSignature(timeSignature);
        if (bpm) stave.setTempo({ duration: 'q', bpm }, 0);
      }

      stave.setContext(context).draw();

      // Convert notes to VexFlow format
      const vexNotes = bar.beats.flatMap(beat =>
        beat.notes.map(note => {
          if (note.is_rest) {
            return new StaveNote({
              keys: ['b/4'],
              duration: `${note.note_value}r`,
            });
          }

          const staveNote = new StaveNote({
            keys: [`${note.staff_name}/${note.octave}`],
            duration: note.note_value || 'q',
          });

          // Add accidentals
          if (note.accidental === 'sharp') {
            staveNote.addModifier(
              new VexFlow.Accidental('#'), 0
            );
          }

          // Color-code by confidence
          const color = getConfidenceColor(note.confidence);
          staveNote.setStyle({ fillStyle: color, strokeStyle: color });

          return staveNote;
        })
      );

      if (vexNotes.length > 0) {
        const voice = new Voice({ num_beats: 4, beat_value: 4 })
          .setStrict(false) // Allow non-standard durations
          .addTickables(vexNotes);
        new Formatter().joinVoices([voice]).format([voice], 1050);
        voice.draw(context, stave);
      }

      yPosition += 100;
    }
  }, [bars, bpm, timeSignature]);

  return <div ref={containerRef} className="overflow-x-auto" />;
}

function getConfidenceColor(confidence: number): string {
  if (confidence >= 0.8) return '#000000';       // Black - high confidence
  if (confidence >= 0.6) return '#4A5568';       // Dark gray
  if (confidence >= 0.4) return '#A0AEC0';       // Light gray
  return '#FC8181';                               // Red - low confidence
}
```

**Deliverables:** VexFlow-based staff renderer with confidence color-coding.

---

### Task 4.2: Hindustani Sargam Renderer (Custom SVG)

**Goal:** Build a custom SVG renderer for Hindustani notation. No existing library handles this.

**What to build:**
```typescript
// frontend/src/components/HindustaniRenderer.tsx

interface HindustaniRendererProps {
  notes: Note[];
  bars?: Bar[];
  tempoDetected: boolean;
}

export function HindustaniRenderer({ notes, bars, tempoDetected }: HindustaniRendererProps) {
  /*
   * Hindustani notation layout:
   *
   * Line 1 (upper dots):    .         .
   * Line 2 (swaras):    Sa Re Ga  Ma Pa Dha Ni  Ṡa
   * Line 3 (lower dots):                .
   * Line 4 (lyrics):    optional text alignment
   * Line 5 (bar lines): |___|___|  |___|___|
   *
   * Rendering rules:
   * - Upper dot → taar saptak (higher octave)
   * - Lower dot → mandra saptak (lower octave)
   * - No dot → madhya saptak (middle)
   * - Underline → komal swara (flat)
   * - Vertical line above Ma → tivra (sharp)
   * - Dash (—) → sustain for one beat
   * - Vertical bar | → beat division
   * - Double bar || → section end
   */

  const svgContent = useMemo(() => {
    const CELL_WIDTH = 60;
    const CELL_HEIGHT = 80;
    const FONT_SIZE = 18;

    let elements: JSX.Element[] = [];
    let x = 40; // Starting x position
    let y = 60; // Starting y position

    const renderNote = (note: Note, index: number) => {
      const group: JSX.Element[] = [];
      const centerX = x + CELL_WIDTH / 2;

      // Swara text
      const displayText = note.is_rest ? '—' : note.swara;
      const confidence = note.confidence;
      const opacity = Math.max(0.4, confidence);

      group.push(
        <text key={`swara-${index}`}
              x={centerX} y={y}
              textAnchor="middle"
              fontSize={FONT_SIZE}
              fontFamily="serif"
              opacity={opacity}
              fill={confidence < 0.5 ? '#E53E3E' : '#1A202C'}>
          {displayText}
        </text>
      );

      // Komal indicator (underline)
      if (note.swara_variant === 'komal') {
        group.push(
          <line key={`komal-${index}`}
                x1={centerX - 12} y1={y + 4}
                x2={centerX + 12} y2={y + 4}
                stroke="#1A202C" strokeWidth={1.5} />
        );
      }

      // Tivra indicator (vertical line above Ma)
      if (note.swara_variant === 'tivra') {
        group.push(
          <line key={`tivra-${index}`}
                x1={centerX} y1={y - FONT_SIZE - 2}
                x2={centerX} y2={y - FONT_SIZE + 6}
                stroke="#1A202C" strokeWidth={1.5} />
        );
      }

      // Octave dots
      if (note.octave_marker === 'upper') {
        group.push(
          <circle key={`dot-upper-${index}`}
                  cx={centerX} cy={y - FONT_SIZE - 6}
                  r={2.5} fill="#1A202C" />
        );
      } else if (note.octave_marker === 'lower') {
        group.push(
          <circle key={`dot-lower-${index}`}
                  cx={centerX} cy={y + 10}
                  r={2.5} fill="#1A202C" />
        );
      }

      // Confidence indicator (small bar below)
      group.push(
        <rect key={`conf-${index}`}
              x={centerX - 10}
              y={y + 18}
              width={20 * confidence}
              height={3}
              fill={confidence >= 0.7 ? '#48BB78' : '#ECC94B'}
              rx={1} />
      );

      x += CELL_WIDTH;
      return group;
    };

    // Render all notes with bar lines if tempo detected
    const allNotes = bars
      ? bars.flatMap(bar => bar.beats.flatMap(beat => beat.notes))
      : notes;

    allNotes.forEach((note, i) => renderNote(note, i));

    return elements;
  }, [notes, bars]);

  return (
    <svg width="100%" viewBox={`0 0 ${totalWidth} ${totalHeight}`}
         className="bg-white border rounded">
      {svgContent}
    </svg>
  );
}
```

**Deliverables:** Custom SVG renderer for Hindustani notation.

---

### Task 4.3: Carnatic Sargam Renderer (Custom SVG)

**Goal:** Build a custom SVG renderer for Carnatic notation. Similar to Hindustani but with different swara names and tala markings.

**Key differences from Hindustani:**
- Swara names: Ri (not Re), Da (not Dha)
- Variant notation: subscript numbers (R₁, R₂, R₃)
- Duration: doubled vowel ("saa" = 2 maathras) shown as elongated text
- Tala markings: | for each akshara, || for section end
- Silence: comma (,) = 1 maathra, semicolon (;) = 2 maathras

**Structure:** Same component pattern as HindustaniRenderer, but with Carnatic-specific rendering rules.

**Deliverables:** Custom SVG renderer for Carnatic notation.

---

### Task 4.4: Confidence Visualization & Silence Display

**Goal:** Show per-note confidence and properly display pauses, silences, and sustained notes.

**What to build:**
- Confidence heat map overlay on each note (green → yellow → red)
- Tooltip on hover showing exact confidence percentage
- Rest/silence symbols appropriate to each notation type:
  - Staff: standard rest symbols (whole rest, half rest, quarter rest, etc.)
  - Hindustani: dash (—) for sustain, empty space for silence
  - Carnatic: comma (,) and semicolon (;) for silences
- Sustained/elongated notes:
  - Staff: tied notes with arc
  - Hindustani: dash after swara name
  - Carnatic: doubled vowel or comma extension

**Deliverables:** Confidence visualization and silence/sustain display in all three renderers.

---

## Phase 5: Live Audio Mode

### Task 5.1: Browser Audio Capture with AudioWorklet

**Goal:** Capture real-time audio from the user's microphone using the Web Audio API.

**What to build:**
```typescript
// frontend/src/hooks/useAudioCapture.ts

export function useAudioCapture() {
  const [isListening, setIsListening] = useState(false);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<AudioWorkletNode | null>(null);

  const startListening = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        sampleRate: 22050,
      }
    });

    const audioContext = new AudioContext({ sampleRate: 22050 });
    audioContextRef.current = audioContext;

    // Register the AudioWorklet processor
    await audioContext.audioWorklet.addModule('/audio-processor.js');

    const source = audioContext.createMediaStreamSource(stream);
    const processor = new AudioWorkletNode(audioContext, 'audio-processor', {
      processorOptions: { bufferSize: 4096 }
    });

    processor.port.onmessage = (event) => {
      // Send audio chunks to backend via WebSocket
      sendAudioChunk(event.data.audioBuffer);
    };

    source.connect(processor);
    processorRef.current = processor;
    setIsListening(true);
  };

  return { isListening, startListening, stopListening };
}
```

```javascript
// public/audio-processor.js (AudioWorklet processor)
class AudioBufferProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.bufferSize = options.processorOptions?.bufferSize || 4096;
    this.buffer = new Float32Array(this.bufferSize);
    this.bufferIndex = 0;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0][0]; // First input, first channel
    if (!input) return true;

    for (let i = 0; i < input.length; i++) {
      this.buffer[this.bufferIndex++] = input[i];
      if (this.bufferIndex >= this.bufferSize) {
        this.port.postMessage({
          audioBuffer: this.buffer.slice()
        });
        this.bufferIndex = 0;
      }
    }
    return true;
  }
}
registerProcessor('audio-processor', AudioBufferProcessor);
```

**Deliverables:** Real-time audio capture hook.

---

### Task 5.2: WebSocket Streaming Backend

**Goal:** Accept audio chunks via WebSocket, run incremental pitch detection, and stream note events back.

**What to build:**
```python
# backend/app/api/live_transcription.py
from fastapi import WebSocket, WebSocketDisconnect
import numpy as np
import asyncio

class LiveTranscriptionHandler:
    def __init__(self):
        self.pitch_detector = PitchDetector()
        self.buffer = np.array([], dtype=np.float32)
        self.WINDOW_SIZE = 22050 * 2   # 2 seconds of audio
        self.HOP_SIZE = 22050 // 2     # Process every 0.5 seconds

    async def handle_connection(self, websocket: WebSocket,
                                 base_pitch: str, notation_type: str):
        await websocket.accept()

        converter = get_converter(notation_type)

        try:
            while True:
                # Receive audio chunk (binary float32 data)
                data = await websocket.receive_bytes()
                chunk = np.frombuffer(data, dtype=np.float32)
                self.buffer = np.concatenate([self.buffer, chunk])

                # When buffer is large enough, process
                if len(self.buffer) >= self.WINDOW_SIZE:
                    # Save temp wav for Basic Pitch
                    temp_path = self._save_temp_wav(
                        self.buffer[:self.WINDOW_SIZE]
                    )

                    # Detect pitches in this window
                    pitch_data = self.pitch_detector.detect(temp_path)

                    # Convert to notation
                    # (simplified tempo data for live mode)
                    tempo_data = {"tempo_detected": False}
                    result = converter.convert(
                        pitch_data["notes"], tempo_data, base_pitch
                    )

                    # Send back to client
                    await websocket.send_json({
                        "type": "notes",
                        "notes": [
                            self._note_to_dict(n)
                            for n in result.ungrouped_notes
                        ],
                        "timestamp": float(len(self.buffer)) / 22050
                    })

                    # Slide the window
                    self.buffer = self.buffer[self.HOP_SIZE:]

        except WebSocketDisconnect:
            pass
```

**Deliverables:** Live WebSocket transcription handler with sliding window.

---

### Task 5.3: Live Notation Display (Scrolling)

**Goal:** Display notes in real-time as they are detected, scrolling the notation view.

**What to build:**
- Auto-scrolling notation container that appends new notes as they arrive
- Visual indicator for "currently playing" note (highlight/animation)
- Accumulate notes into a growing transcription result
- "Stop & Finalize" button that runs full tempo analysis on the accumulated audio and re-renders with proper bar groupings

**Deliverables:** Real-time scrolling notation display for live mode.

---

## Phase 6: PDF Export

### Task 6.1: Server-Side PDF Generation

**Goal:** Generate PDF export of the transcription using ReportLab with embedded SVG notation.

**What to build:**
```python
# backend/app/services/pdf_exporter.py
from reportlab.lib.pagesizes import A4, letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
import io

class PDFExporter:
    def export(self, transcription: TranscriptionResult,
               page_size=letter) -> bytes:
        """Generate PDF bytes from transcription result."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=page_size)
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph("Music Transcription", styles['Title']))
        story.append(Spacer(1, 12))

        # Metadata
        meta_lines = [
            f"Notation Type: {transcription.notation_type.value.title()}",
            f"Base Pitch: {transcription.base_pitch}",
        ]
        if transcription.tempo_detected:
            meta_lines.append(f"Tempo: {transcription.bpm:.0f} BPM")
            meta_lines.append(
                f"Time Signature: {transcription.time_signature}"
            )
        else:
            meta_lines.append("Tempo: Not detected (free time)")

        meta_lines.append(
            f"Average Confidence: {transcription.confidence_mean:.1%}"
        )

        for line in meta_lines:
            story.append(Paragraph(line, styles['Normal']))
        story.append(Spacer(1, 24))

        # Render notation to SVG, then embed in PDF
        svg_content = self._render_notation_svg(transcription)
        drawing = svg2rlg(io.BytesIO(svg_content.encode()))
        if drawing:
            story.append(drawing)

        doc.build(story)
        buffer.seek(0)
        return buffer.read()

    def _render_notation_svg(self, transcription) -> str:
        """Generate SVG string for the notation.
        For Staff: use a headless VexFlow render (via puppeteer/node)
        For Indian: use Python SVG generation"""
        if transcription.notation_type == NotationType.STAFF:
            return self._render_staff_svg(transcription)
        elif transcription.notation_type == NotationType.HINDUSTANI:
            return self._render_hindustani_svg(transcription)
        else:
            return self._render_carnatic_svg(transcription)
```

**Alternative approach for staff notation PDF:** Use `music21` library (Python) which can generate MusicXML → render to PDF via LilyPond.

```python
# Alternative: music21 for high-quality staff notation PDFs
from music21 import stream, note, meter, tempo, key

def export_staff_pdf(transcription):
    s = stream.Score()
    p = stream.Part()

    if transcription.bpm:
        p.insert(0, tempo.MetronomeMark(number=transcription.bpm))
    if transcription.time_signature:
        p.insert(0, meter.TimeSignature(transcription.time_signature))

    for bar in transcription.bars:
        m = stream.Measure()
        for beat in bar.beats:
            for n in beat.notes:
                if n.is_rest:
                    m.append(note.Rest(quarterLength=n.duration_beats))
                else:
                    m.append(note.Note(
                        n.midi_pitch,
                        quarterLength=n.duration_beats or 1.0
                    ))
        p.append(m)

    s.append(p)
    # Requires LilyPond installed for PDF output
    s.write('lily.pdf', fp='output.pdf')
```

**Deliverables:** PDF export endpoint returning downloadable PDF.

---

## Phase 7: Notation-to-Notation Translation (OMR)

### Task 7.1: Sheet Music Upload & OMR

**Goal:** Allow users to upload an image of sheet music (staff notation), run Optical Music Recognition to extract notes, and translate to other notation systems.

**What to build:**
```python
# backend/app/services/omr_engine.py

class OMREngine:
    def __init__(self):
        # oemer: Open-source end-to-end OMR
        # Outputs MusicXML from sheet music images
        pass

    def recognize(self, image_path: str) -> list[dict]:
        """
        Run OMR on uploaded sheet music image.
        Returns list of note events extracted from the notation.
        """
        import subprocess
        import xml.etree.ElementTree as ET

        # Run oemer CLI
        result = subprocess.run(
            ['oemer', image_path],
            capture_output=True, text=True
        )

        # Parse the output MusicXML
        musicxml_path = image_path.replace('.png', '.musicxml')
        notes = self._parse_musicxml(musicxml_path)
        return notes

    def _parse_musicxml(self, path: str) -> list[dict]:
        """Parse MusicXML and extract note events."""
        import music21
        score = music21.converter.parse(path)
        notes = []

        for element in score.recurse().notes:
            if isinstance(element, music21.note.Note):
                notes.append({
                    "midi_pitch": element.pitch.midi,
                    "frequency": element.pitch.frequency,
                    "start_time": float(element.offset),
                    "end_time": float(element.offset + element.quarterLength),
                    "velocity": element.volume.velocity or 80,
                    "confidence": 1.0,  # OMR doesn't provide confidence
                    "note_value": element.duration.type,
                })
        return notes
```

**Translation flow:**
1. User uploads image of sheet music (PNG/JPG)
2. OMR extracts MusicXML
3. Parse MusicXML → Internal Music Representation
4. Convert to target notation using existing converters from Phase 3

**Deliverables:** OMR pipeline that reads sheet music images and feeds into the notation converter.

---

### Task 7.2: Notation Translation API Endpoint

**Goal:** Wire OMR + converter into the `/api/translate-notation` endpoint.

```python
@app.post("/api/translate-notation")
async def translate_notation(
    file: UploadFile,
    source_type: str = "staff",     # Currently only staff→others
    target_type: str = "hindustani",
    base_pitch: str = "C4"
):
    # Save uploaded image
    image_path = save_upload(file)

    # Run OMR
    omr = OMREngine()
    notes = omr.recognize(image_path)

    # Convert to target notation
    converter = get_converter(target_type)
    tempo_data = extract_tempo_from_musicxml(notes)  # From MusicXML data
    result = converter.convert(notes, tempo_data, base_pitch)

    return result.to_dict()
```

**Deliverables:** Complete notation translation endpoint.

---

## Phase 8: Interactive Notation Editor & Playback Engine

This phase adds a fully interactive canvas where users can compose music by placing notes directly onto notation grids, play back their compositions, and also view/replay detected notes from transcription or live recording.

**Core capabilities:**
1. **Blank canvas composition** — Place notes on a staff (5 lines) or Carnatic svara grid by clicking/tapping
2. **Playback** — Click "Play" to hear the placed notes using a browser-based synthesizer
3. **Detection view** — After transcription or live recording, detected notes appear on the same canvas as editable objects
4. **Detection playback** — Once recording stops, click "Play" to hear the detected transcription rendered as audio
5. **Edit detected notes** — Drag, delete, or add notes to correct transcription errors before exporting

**Technology choices:**

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Audio synthesis** | Tone.js | Industry-standard Web Audio framework. `Tone.Synth` for quick synth playback, `Tone.Sampler` with Salamander piano samples for realistic instrument sound. Supports precise scheduling via `Tone.Transport`. |
| **Staff canvas interaction** | VexFlow 5 SVG + custom hit-detection layer | VexFlow renders the notation; an overlay SVG/div layer captures clicks, resolves y-position → pitch and x-position → time slot. |
| **Carnatic canvas interaction** | Custom SVG grid + click handlers | Cell-based grid layout (svara × time). Clicking a cell places the corresponding svara at that beat position. |
| **Playback scheduling** | Tone.Transport + Tone.Part | `Tone.Part` accepts an array of `{time, note, duration}` events and schedules them precisely against the transport timeline. Handles tempo changes. |
| **State management** | Zustand `editorStore` | Tracks placed notes, cursor position, selected tool (note value, rest, eraser), playback state, and tempo. Shared with the existing transcription store so detected notes flow into the editor. |

---

### Task 8.1: Playback Audio Engine (Tone.js Integration)

**Goal:** Build a browser-side audio engine that can synthesize and play back any sequence of notes with accurate timing.

```typescript
// frontend/src/audio/PlaybackEngine.ts
import * as Tone from "tone";

interface PlayableNote {
  pitch: string;       // e.g., "C4", "D#5"
  startTime: number;   // seconds from beginning
  duration: number;    // seconds
  velocity?: number;   // 0-1
}

export class PlaybackEngine {
  private sampler: Tone.Sampler | null = null;
  private synth: Tone.PolySynth | null = null;
  private part: Tone.Part | null = null;
  private mode: "sampler" | "synth" = "synth";

  /**
   * Initialize with Salamander piano samples for realistic playback.
   * Falls back to PolySynth if samples fail to load.
   */
  async init(mode: "sampler" | "synth" = "synth") {
    this.mode = mode;

    if (mode === "sampler") {
      this.sampler = new Tone.Sampler({
        urls: {
          C4: "C4.mp3",
          "D#4": "Ds4.mp3",
          "F#4": "Fs4.mp3",
          A4: "A4.mp3",
        },
        release: 1,
        baseUrl: "https://tonejs.github.io/audio/salamander/",
      }).toDestination();

      await Tone.loaded();
    } else {
      this.synth = new Tone.PolySynth(Tone.Synth, {
        oscillator: { type: "triangle" },
        envelope: {
          attack: 0.02,
          decay: 0.1,
          sustain: 0.3,
          release: 0.8,
        },
      }).toDestination();
    }
  }

  /**
   * Schedule and play a sequence of notes.
   * Uses Tone.Part for sample-accurate timing.
   */
  async playSequence(
    notes: PlayableNote[],
    bpm: number = 120,
    onNotePlay?: (index: number) => void
  ) {
    await Tone.start(); // Required: resume AudioContext after user gesture

    Tone.getTransport().bpm.value = bpm;
    this.stop(); // Clear any previous playback

    const instrument = this.mode === "sampler" ? this.sampler! : this.synth!;

    // Build Tone.Part events
    const events = notes.map((note, i) => ({
      time: note.startTime,
      pitch: note.pitch,
      duration: note.duration,
      velocity: note.velocity ?? 0.8,
      index: i,
    }));

    this.part = new Tone.Part((time, event) => {
      instrument.triggerAttackRelease(
        event.pitch,
        event.duration,
        time,
        event.velocity
      );
      // Fire callback on main thread for visual highlighting
      Tone.getDraw().schedule(() => {
        onNotePlay?.(event.index);
      }, time);
    }, events);

    this.part.start(0);
    Tone.getTransport().start();
  }

  /**
   * Play a single note immediately (for click-to-preview while editing).
   */
  playNote(pitch: string, duration: number = 0.3) {
    const instrument = this.mode === "sampler" ? this.sampler! : this.synth!;
    instrument?.triggerAttackRelease(pitch, duration);
  }

  stop() {
    this.part?.stop();
    this.part?.dispose();
    this.part = null;
    Tone.getTransport().stop();
    Tone.getTransport().position = 0;
  }

  get isPlaying(): boolean {
    return Tone.getTransport().state === "started";
  }
}
```

**Key design decisions:**
- **Dual mode**: `synth` mode (lightweight `PolySynth`) loads instantly for quick preview; `sampler` mode (Salamander piano samples) gives realistic sound but requires loading ~2MB of audio files. User can toggle between them.
- **`Tone.Draw.schedule`**: Bridges the Web Audio clock (high-precision, runs on audio thread) to the UI thread for synchronized note highlighting — without this, visual feedback drifts from audio.
- **`Tone.start()` on user gesture**: Browsers block AudioContext creation until a user interaction; the engine calls `Tone.start()` at the beginning of every play action to guarantee it works.

**Deliverables:** `PlaybackEngine` class with `playSequence()`, `playNote()`, `stop()`, and `init()`.

---

### Task 8.2: Staff Notation Editor Canvas

**Goal:** Build an interactive staff editor where users click to place notes on a standard 5-line staff, with drag-to-reposition and click-to-delete.

**Architecture:** A transparent interaction layer sits on top of the VexFlow SVG rendering. User clicks are translated to (pitch, time-slot) coordinates using the staff geometry. The note is added to state, and VexFlow re-renders the entire staff.

```typescript
// frontend/src/components/editor/StaffEditor.tsx
import { useCallback, useRef, useState } from "react";
import { useEditorStore } from "../../stores/editorStore";
import { PlaybackEngine } from "../../audio/PlaybackEngine";

/**
 * Pitch mapping: Y-position on staff → note name.
 *
 * The treble clef staff spans 5 lines (E4, G4, B4, D5, F5)
 * and 4 spaces (F4, A4, C5, E5). Ledger lines extend above/below.
 *
 * Each half-step in vertical position = one staff position.
 */
const STAFF_POSITIONS: Record<number, string> = {
  // Ledger lines below
  0: "C4", 1: "D4",
  // On the staff
  2: "E4", 3: "F4", 4: "G4", 5: "A4", 6: "B4",
  7: "C5", 8: "D5", 9: "E5", 10: "F5",
  // Ledger lines above
  11: "G5", 12: "A5",
};

interface EditorConfig {
  timeSlots: number;        // How many note positions horizontally (e.g., 16)
  beatsPerMeasure: number;  // e.g., 4
  subdivisions: number;     // Slots per beat (1 = quarter notes, 2 = eighths)
  bpm: number;
}

export function StaffEditor({ config }: { config: EditorConfig }) {
  const canvasRef = useRef<HTMLDivElement>(null);
  const { notes, addNote, removeNote, selectedDuration } = useEditorStore();
  const playbackEngine = useRef(new PlaybackEngine());

  // Translate a click event's Y coordinate → staff pitch
  const yToPitch = useCallback((y: number, staffTop: number, lineSpacing: number) => {
    // staffTop = Y of the top line (F5), lineSpacing = px between lines
    const halfLineSpacing = lineSpacing / 2;
    const positionFromTop = Math.round((y - staffTop) / halfLineSpacing);
    // Top line (F5) = position 10, going down to C4 = position 0
    const staffIndex = 10 - positionFromTop;
    return STAFF_POSITIONS[Math.max(0, Math.min(12, staffIndex))] ?? "C4";
  }, []);

  // Translate a click event's X coordinate → time slot index
  const xToTimeSlot = useCallback((x: number, gridLeft: number, slotWidth: number) => {
    return Math.floor((x - gridLeft) / slotWidth);
  }, []);

  const handleCanvasClick = useCallback((e: React.MouseEvent) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const staffTop = 40;      // px — top line Y
    const lineSpacing = 12;   // px between staff lines
    const gridLeft = 80;      // px — after clef/time sig
    const slotWidth = (rect.width - gridLeft - 20) / config.timeSlots;

    const pitch = yToPitch(y, staffTop, lineSpacing);
    const timeSlot = xToTimeSlot(x, gridLeft, slotWidth);

    if (timeSlot >= 0 && timeSlot < config.timeSlots) {
      // Check if note already exists at this position → toggle off
      const existing = notes.find(
        n => n.pitch === pitch && n.timeSlot === timeSlot
      );
      if (existing) {
        removeNote(existing.id);
      } else {
        addNote({ pitch, timeSlot, duration: selectedDuration });
        // Audible feedback: play the note on click
        playbackEngine.current.playNote(pitch);
      }
    }
  }, [notes, config, addNote, removeNote, selectedDuration, yToPitch, xToTimeSlot]);

  // ... VexFlow render logic + overlay div with onClick
  return (
    <div className="staff-editor" ref={canvasRef} onClick={handleCanvasClick}>
      {/* VexFlow SVG renders here */}
      <div className="vexflow-layer" id="staff-svg-container" />
      {/* Transparent click-capture overlay with visual grid lines */}
      <div className="interaction-overlay">
        {/* Grid lines, hover indicators, note head previews */}
      </div>
    </div>
  );
}
```

**Interaction layer design:**

```
┌─────────────────────────────────────────────────────────┐
│  ♩ = 120 BPM   4/4 Time                                │
│                                                         │
│  ┌─ Clef ─┐  ┌─ Beat 1 ─┐  ┌─ Beat 2 ─┐  ┌─ Beat 3 ─┐│
│  │  𝄞     │  │  ●  .  .  │  │  .  ●  .  │  │  .  .  .  ││  ← F5 (top line)
│  │        │──│──.──.──.──│──│──.──.──.──│──│──.──.──.──││  ← E5
│  │        │  │  .  .  .  │  │  .  .  .  │  │  .  .  .  ││  ← D5
│  │        │──│──.──.──.──│──│──.──.──.──│──│──.──.──.──││  ← C5
│  │        │  │  .  .  .  │  │  .  .  .  │  │  .  .  .  ││  ← B4
│  │        │──│──.──.──.──│──│──.──.──.──│──│──.──.──.──││  ← A4
│  │        │  │  .  .  .  │  │  .  .  .  │  │  .  .  .  ││  ← G4
│  │        │──│──.──.──.──│──│──.──.──.──│──│──.──.──.──││  ← F4
│  │        │  │  .  .  .  │  │  .  .  .  │  │  .  .  .  ││  ← E4 (bottom line)
│  └────────┘  └───────────┘  └───────────┘  └───────────┘│
│                                                         │
│  [𝅝][𝅗𝅥][♩][♪][𝅘𝅥𝅯]  [Rest] [Eraser]  [▶ Play] [⏹ Stop] │
│  [Undo] [Redo] [Clear All]   Tempo: [120▼] BPM         │
└─────────────────────────────────────────────────────────┘

● = placed note    . = empty grid position (shown on hover)
```

**Key interactions:**
- **Click on grid intersection** → places a note head; audible preview plays
- **Click on existing note** → removes it (toggle behavior)
- **Drag note vertically** → changes pitch; horizontal → moves time slot
- **Toolbar** selects note duration (whole, half, quarter, eighth, sixteenth), rest, or eraser
- **Sharps/flats** → secondary click or modifier key (Shift+click = sharp, Ctrl+click = flat)
- **Hover** → ghost note-head appears at nearest valid pitch, snapping to staff lines/spaces

**Deliverables:** Interactive `StaffEditor` component with click-to-place, drag, delete, toolbar, and audible preview.

---

### Task 8.3: Carnatic Svara Editor Canvas

**Goal:** Build an interactive grid editor for Carnatic notation where users click cells to place svaras at specific time positions.

**Architecture:** A cell-based SVG grid where columns represent time (maathras) and rows represent the 16 svara positions. Clicking a cell fills it with the corresponding svara.

```typescript
// frontend/src/components/editor/CarnaticEditor.tsx

/**
 * Carnatic svara grid layout:
 *
 * Rows (top to bottom):    Columns (left to right):
 *   Upper octave svaras      Maathra 1, 2, 3, 4, ... N
 *   Middle octave svaras     (grouped into aksharas of 4)
 *   Lower octave svaras
 *
 * Each cell = 1 maathra duration
 * 4 maathras = 1 akshara (beat)
 * Default: Adi tala = 8 beats = 32 maathras
 */

const CARNATIC_ROWS = [
  // Upper octave (taar saptak - dot above)
  { svara: "Ṡa", midi_offset: 12, octave: "upper" },
  { svara: "Ṅi₃", midi_offset: 11, octave: "upper" },
  // Middle octave
  { svara: "Ni₃", midi_offset: 11, octave: "middle" },
  { svara: "Da₃", midi_offset: 10, octave: "middle" },
  { svara: "Da₂", midi_offset: 9, octave: "middle" },
  { svara: "Da₁", midi_offset: 8, octave: "middle" },
  { svara: "Pa",  midi_offset: 7, octave: "middle" },
  { svara: "Ma₂", midi_offset: 6, octave: "middle" },
  { svara: "Ma₁", midi_offset: 5, octave: "middle" },
  { svara: "Ga₃", midi_offset: 4, octave: "middle" },
  { svara: "Ri₃", midi_offset: 3, octave: "middle" },  // = Ga₂
  { svara: "Ri₂", midi_offset: 2, octave: "middle" },  // = Ga₁
  { svara: "Ri₁", midi_offset: 1, octave: "middle" },
  { svara: "Sa",  midi_offset: 0, octave: "middle" },
  // Lower octave (mandra saptak - dot below)
  { svara: "Ṇi₃", midi_offset: -1, octave: "lower" },
  { svara: "Ḍa₃", midi_offset: -2, octave: "lower" },
];

interface CarnaticEditorProps {
  basePitch: string;     // User-selected tonic, e.g. "C4"
  totalMaathras: number; // e.g. 32 (= 8 beats × 4 maathras)
  bpm: number;
}

export function CarnaticEditor({ basePitch, totalMaathras, bpm }: CarnaticEditorProps) {
  // ...state from Zustand editorStore

  const handleCellClick = (rowIndex: number, maathraIndex: number) => {
    const row = CARNATIC_ROWS[rowIndex];
    const baseMidi = noteToMidi(basePitch);
    const midiPitch = baseMidi + row.midi_offset;
    const pitchName = midiToNoteName(midiPitch);

    // Toggle note on/off at this cell
    const existing = notes.find(
      n => n.row === rowIndex && n.timeSlot === maathraIndex
    );
    if (existing) {
      removeNote(existing.id);
    } else {
      addNote({
        pitch: pitchName,
        svara: row.svara,
        timeSlot: maathraIndex,
        duration: 1,  // 1 maathra; user can drag right to extend
        octave: row.octave,
      });
      playbackEngine.playNote(pitchName);
    }
  };

  // Render SVG grid
  return (
    <div className="carnatic-editor">
      <svg width={...} height={...}>
        {/* Tala grouping lines: | every 4 maathras, || every section */}
        {/* Row labels (svara names) on left */}
        {/* Grid cells — filled cells show svara text + color */}
        {CARNATIC_ROWS.map((row, ri) =>
          Array.from({ length: totalMaathras }).map((_, mi) => (
            <rect
              key={`${ri}-${mi}`}
              x={labelWidth + mi * cellWidth}
              y={ri * cellHeight}
              width={cellWidth}
              height={cellHeight}
              className={hasNote(ri, mi) ? "cell-filled" : "cell-empty"}
              onClick={() => handleCellClick(ri, mi)}
            />
          ))
        )}
      </svg>
      {/* Toolbar: play, stop, clear, tempo, tala selector */}
    </div>
  );
}
```

**Grid visual layout:**

```
         Tala: Adi (4+2+2)    Tempo: 80 BPM    Base: C4 (Sa)
         ┃  Akshara 1  ┃  Akshara 2  ┃  Akshara 3  ┃
         ┃ 1  2  3  4  ┃ 1  2  3  4  ┃ 1  2  3  4  ┃
  ──────┬┬────────────┬┬────────────┬┬────────────┬┬──
  Ṡa   ││            ││            ││            ││  ← upper
  ──────┼┼────────────┼┼────────────┼┼────────────┼┼──
  Ni₃   ││            ││         ██ ││            ││
  Da₂   ││         ██ ││            ││            ││
  Pa    ││      ██    ││            ││            ││
  Ma₁   ││            ││            ││      ██    ││
  Ga₃   ││   ██       ││            ││   ██       ││
  Ri₂   ││            ││   ██       ││            ││
  Sa    ││██          ││            ││██          ││
  ──────┼┼────────────┼┼────────────┼┼────────────┼┼──
  Ṇi₃   ││            ││            ││            ││  ← lower
  ──────┴┴────────────┴┴────────────┴┴────────────┴┴──

  ██ = placed svara    (click to place / remove)

  [▶ Play] [⏹ Stop] [🗑 Clear]  Duration: [1][2][4] maathras
```

**Key interactions:**
- **Click cell** → places svara; plays audible preview of the pitch
- **Click filled cell** → removes svara (toggle)
- **Drag right on filled cell** → extends duration across maathras (sustained note, shown as "saa", "saaa" etc.)
- **Tala selector** → changes column grouping (Adi 8, Rupaka 6, Misra Chapu 7, etc.)
- **Raga filter (optional)** → grays out svaras not in the selected raga, preventing invalid note placement

**Deliverables:** Interactive `CarnaticEditor` component with cell-based svara placement, tala grouping, and audible preview.

---

### Task 8.4: Detected Notes Visualization & Edit

**Goal:** After audio transcription or live recording, render the detected notes onto the same interactive editor canvas so users can see, play back, and correct the transcription.

**Data flow:**

```
┌──────────────────┐      ┌─────────────────┐      ┌──────────────────┐
│  Transcription    │      │  Editor Store    │      │  Editor Canvas   │
│  Result           │─────▶│  (Zustand)       │─────▶│  (Staff or       │
│  (from Phase 3/5) │      │                  │      │   Carnatic)      │
│                   │      │  notes[]         │      │                  │
│  notes[]:         │      │  tempo           │      │  ● rendered      │
│   - midi_pitch    │      │  timeSignature   │      │  ● clickable     │
│   - start_time    │      │  isFromDetection │      │  ● draggable     │
│   - end_time      │      │                  │      │  ● playable      │
│   - confidence    │      └─────────────────┘      └──────────────────┘
│   - svara         │             ▲
└──────────────────┘             │
                                 │  User edits
                           ┌─────────────────┐
                           │  Corrections:    │
                           │  - drag to move  │
                           │  - delete wrong  │
                           │  - add missing   │
                           │  - adjust timing │
                           └─────────────────┘
```

```typescript
// frontend/src/stores/editorStore.ts
import { create } from "zustand";

interface EditorNote {
  id: string;
  pitch: string;          // "C4", "D#5", etc.
  svara?: string;         // "Sa", "Ri₂", etc. (for Carnatic/Hindustani)
  timeSlot: number;       // Grid position (quantized)
  duration: number;       // In grid units (slots or maathras)
  startTime?: number;     // Original time in seconds (from detection)
  velocity?: number;
  confidence?: number;    // From detection — null if user-placed
  octave?: string;        // "lower" | "middle" | "upper"
  isFromDetection: boolean;
}

interface EditorState {
  notes: EditorNote[];
  bpm: number;
  timeSignature: [number, number];  // e.g., [4, 4]
  selectedDuration: number;         // Current tool: note duration in slots
  selectedTool: "note" | "rest" | "eraser";
  playbackState: "stopped" | "playing" | "paused";
  highlightedNoteIndex: number | null;  // Currently playing note

  // Actions
  addNote: (note: Omit<EditorNote, "id">) => void;
  removeNote: (id: string) => void;
  moveNote: (id: string, newPitch: string, newTimeSlot: number) => void;
  clearAll: () => void;
  loadFromTranscription: (result: TranscriptionResult) => void;
}

export const useEditorStore = create<EditorState>((set) => ({
  notes: [],
  bpm: 120,
  timeSignature: [4, 4],
  selectedDuration: 1,
  selectedTool: "note",
  playbackState: "stopped",
  highlightedNoteIndex: null,

  /**
   * Load detected notes from a transcription result into the editor.
   * Quantizes continuous start_time values into discrete time slots.
   */
  loadFromTranscription: (result) => {
    const slotDuration = 60 / (result.bpm * result.subdivisions);
    const editorNotes: EditorNote[] = result.notes.map((n, i) => ({
      id: `det-${i}`,
      pitch: n.staff_name ?? midiToPitch(n.midi_pitch),
      svara: n.swara ?? undefined,
      timeSlot: Math.round(n.start_time / slotDuration),
      duration: Math.max(1, Math.round(
        (n.end_time - n.start_time) / slotDuration
      )),
      startTime: n.start_time,
      velocity: n.velocity,
      confidence: n.confidence,
      octave: n.octave_marker,
      isFromDetection: true,
    }));
    set({ notes: editorNotes, bpm: result.bpm });
  },

  // ... other actions (addNote, removeNote, moveNote, clearAll)
}));
```

**Visual distinction for detected vs. user-placed notes:**
- **Detected notes**: Rendered with a confidence-based color gradient (green ≥ 0.8, amber ≥ 0.5, red < 0.5) and a subtle dashed border
- **User-placed notes**: Solid fill, no border decoration
- **Hover tooltip on detected notes**: Shows "Detected: C4, confidence: 87%, original time: 1.234s"

**Playback highlighting:** During playback, the currently-playing note receives a pulsing glow animation. A vertical "playhead" line sweeps across the canvas in sync with `Tone.Transport`.

```css
/* Playback cursor animation */
.playhead-line {
  stroke: #3b82f6;
  stroke-width: 2;
  animation: none;
}
.playhead-line.playing {
  transition: transform linear;
  /* X position updated via JS in sync with Tone.Draw */
}
.note-highlight {
  filter: drop-shadow(0 0 6px rgba(59, 130, 246, 0.8));
  transform: scale(1.1);
  transition: all 0.05s ease;
}
```

**Deliverables:** `loadFromTranscription()` store action, detected-note styling, playhead cursor, and playback-highlight integration.

---

### Task 8.5: Unified Editor Page & Toolbar

**Goal:** Create the `/editor` page that brings together the Staff and Carnatic editor canvases, the playback engine, and a comprehensive toolbar.

**Page layout:**

```
┌─────────────────────────────────────────────────────────────┐
│  Notation Editor                                            │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  [Staff] [Carnatic] [Hindustani]    Base pitch: [C4 ▼]  ││
│  └─────────────────────────────────────────────────────────┘│
│                                                             │
│  ┌── Toolbar ──────────────────────────────────────────────┐│
│  │ Note: [𝅝][𝅗𝅥][♩][♪][𝅘𝅥𝅯]  [♯][♭][♮]  [Rest] [Eraser]   ││
│  │ Tempo: [120] BPM    Time Sig: [4/4 ▼]   Bars: [4 ▼]   ││
│  └─────────────────────────────────────────────────────────┘│
│                                                             │
│  ┌── Canvas ───────────────────────────────────────────────┐│
│  │                                                         ││
│  │   (Staff or Carnatic editor renders here)               ││
│  │   (Detected notes shown if loaded from transcription)   ││
│  │                                                         ││
│  └─────────────────────────────────────────────────────────┘│
│                                                             │
│  ┌── Transport ────────────────────────────────────────────┐│
│  │  [⏮][▶ Play][⏹ Stop]  0:00 / 0:08  ════════●══════    ││
│  │  Sound: [Synth ▼]  Volume: ═══●════                     ││
│  └─────────────────────────────────────────────────────────┘│
│                                                             │
│  ┌── Actions ──────────────────────────────────────────────┐│
│  │  [Load from transcription]  [Export PDF]  [Export MIDI]  ││
│  │  [Undo] [Redo] [Clear All]                              ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

**Integration with existing features:**
- **"Load from transcription"** button: Takes the most recent `TranscriptionResult` from the transcription store and feeds it into `loadFromTranscription()`. Notes appear on the canvas, ready to play or edit.
- **"Load from live recording"** button: After the user stops a live recording session (Phase 5), the accumulated notes are loaded into the editor. The user can then click Play to hear the full playback of what was just recorded/detected.
- **"Export PDF"** button: Sends the current editor notes to the Phase 6 PDF export endpoint.
- **"Export MIDI"** button: Converts editor notes to a MIDI file client-side using the `midi-writer-js` npm package.

**Conversion between editor types:** When the user switches between Staff ↔ Carnatic tabs, the notes are automatically translated using the Phase 3 converters:
```typescript
const switchNotationType = (targetType: "staff" | "carnatic") => {
  const currentNotes = editorStore.getState().notes;
  // Convert each note's pitch representation
  const converted = currentNotes.map(note => {
    if (targetType === "carnatic") {
      const { svara, octave } = pitchToCarnatic(note.pitch, basePitch);
      return { ...note, svara, octave };
    } else {
      return { ...note, svara: undefined, octave: undefined };
    }
  });
  editorStore.setState({ notes: converted });
};
```

**New dependencies to install:**
```bash
npm install tone midi-writer-js
```

**Deliverables:** Complete `/editor` page with notation type tabs, toolbar, transport controls, and integration with transcription/live-mode output.

---

## Phase 9: ML Model Training (Optional Enhancement)

### Task 9.1: Dataset Preparation

**Goal:** Prepare training data for a custom polyphonic transcription model that improves on Basic Pitch for non-western instruments.

**Datasets to use:**

| Dataset | Size | Content | Use |
|---------|------|---------|-----|
| **MAESTRO v3** | 172 hrs | Piano audio + MIDI | Primary training |
| **MusicNet** | 34 hrs | Multi-instrument + labels | Multi-instrument training |
| **GuitarSet** | 3 hrs | Guitar + annotations | Guitar generalization |
| **Slakh2100** | 145 hrs | Synthesized multi-track | Multi-instrument variety |
| **Saraga (CompMusic)** | ~100 hrs | Indian classical music with annotations | Indian music specialization |

**Synthetic data generation for Indian music:**
```python
# ml/data/synthetic_indian_generator.py
"""
Generate synthetic training data for Indian classical music:
1. Use MIDI files of ragas (available from various Indian music databases)
2. Synthesize audio using Indian instrument soundfonts (sitar, tabla, veena)
3. Apply audio augmentations (room reverb, noise, pitch shift)
4. Create paired (audio, MIDI) datasets
"""
import pretty_midi
import numpy as np
from pedalboard import Reverb, Chorus
from pedalboard.io import AudioFile

class IndianMusicSynthesizer:
    RAGAS = {
        "yaman": [0, 2, 4, 6, 7, 9, 11],     # S R G M̃ P D N
        "bhairav": [0, 1, 4, 5, 7, 8, 11],    # S r G M P d N
        "kafi": [0, 2, 3, 5, 7, 9, 10],       # S R g M P D n
        "bilawal": [0, 2, 4, 5, 7, 9, 11],    # S R G M P D N (major)
    }

    def generate_raga_midi(self, raga_name: str, base_pitch: int = 60,
                           n_notes: int = 50, tempo: float = 80):
        """Generate a MIDI file following raga rules."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        inst = pretty_midi.Instrument(program=0)

        scale = self.RAGAS[raga_name]
        current_time = 0.0

        for _ in range(n_notes):
            # Pick a scale degree (weighted towards Sa, Pa, Ga)
            degree = np.random.choice(scale, p=self._get_weights(scale))
            octave = np.random.choice([-12, 0, 12], p=[0.15, 0.7, 0.15])
            pitch = base_pitch + degree + octave

            duration = np.random.choice(
                [0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
                p=[0.1, 0.25, 0.2, 0.25, 0.1, 0.1]
            )

            note = pretty_midi.Note(
                velocity=np.random.randint(60, 110),
                pitch=pitch,
                start=current_time,
                end=current_time + duration
            )
            inst.notes.append(note)
            current_time += duration + np.random.uniform(0, 0.2)

        midi.instruments.append(inst)
        return midi
```

**Deliverables:** Data preparation pipeline with download scripts, preprocessing, and synthetic data generation.

---

### Task 8.2: Custom Model Architecture

**Goal:** Train a custom PyTorch model that extends Basic Pitch's approach for better performance, especially on non-piano instruments and Indian classical music.

**Recommended architecture:** CNN + Transformer hybrid (similar to the approach in the 2025 AMT Challenge winning submissions).

```python
# ml/models/transcription_model.py
import torch
import torch.nn as nn

class MusicTranscriptionModel(nn.Module):
    """
    Architecture:
    1. Mel spectrogram input (n_mels=229, like Basic Pitch)
    2. CNN feature extractor (ResNet-style blocks)
    3. Transformer encoder for temporal context
    4. Three output heads:
       - Onset detection (frame-level)
       - Note activation (frame-level)
       - Velocity estimation (note-level)
    """

    def __init__(self, n_mels=229, n_pitches=88,
                 cnn_channels=64, transformer_heads=8,
                 transformer_layers=4, hidden_dim=256):
        super().__init__()

        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_channels, (3, 3), padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.Conv2d(cnn_channels, cnn_channels, (3, 3), padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),

            nn.Conv2d(cnn_channels, cnn_channels * 2, (3, 3), padding=1),
            nn.BatchNorm2d(cnn_channels * 2),
            nn.ReLU(),
            nn.Conv2d(cnn_channels * 2, cnn_channels * 2, (3, 3), padding=1),
            nn.BatchNorm2d(cnn_channels * 2),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
        )

        # Project CNN output to transformer dimension
        cnn_output_freq = n_mels // 4
        self.projector = nn.Linear(
            cnn_channels * 2 * cnn_output_freq, hidden_dim
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )

        # Output Heads
        self.onset_head = nn.Linear(hidden_dim, n_pitches)
        self.note_head = nn.Linear(hidden_dim, n_pitches)
        self.velocity_head = nn.Linear(hidden_dim, n_pitches)

    def forward(self, mel_spectrogram):
        # mel_spectrogram: (batch, 1, n_frames, n_mels)
        x = self.cnn(mel_spectrogram)           # (B, C, T, F')
        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * F)  # (B, T, C*F')
        x = self.projector(x)                   # (B, T, hidden_dim)
        x = self.transformer(x)                 # (B, T, hidden_dim)

        onset = torch.sigmoid(self.onset_head(x))     # (B, T, 88)
        note = torch.sigmoid(self.note_head(x))       # (B, T, 88)
        velocity = torch.sigmoid(self.velocity_head(x)) # (B, T, 88)

        return onset, note, velocity
```

**Training script structure:**
```python
# ml/training/train.py
# - DataLoader: load MAESTRO/MusicNet audio-MIDI pairs
# - Loss: Binary cross-entropy for onset/note, MSE for velocity
# - Optimizer: AdamW with cosine annealing
# - Augmentation: pitch shift, time stretch, noise injection, room reverb
# - Evaluation: mir_eval note-level F1, onset-only F1
```

**Deliverables:** Model definition, training loop, and evaluation script.

---

### Task 9.3: Model Evaluation & ONNX Export

**Goal:** Evaluate the trained model against standard benchmarks and export to ONNX for deployment.

**Evaluation metrics (using `mir_eval`):**
- Note-level F1 (onset + offset within 50ms tolerance)
- Onset-only F1 (onset within 50ms)
- Frame-level F1 (pitch presence per frame)
- Velocity accuracy (within threshold)

**ONNX export for browser deployment:**
```python
# ml/export/export_onnx.py
import torch

model = MusicTranscriptionModel()
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

dummy_input = torch.randn(1, 1, 100, 229)  # (batch, channel, frames, mels)
torch.onnx.export(
    model, dummy_input, "transcription_model.onnx",
    input_names=["mel_spectrogram"],
    output_names=["onset", "note", "velocity"],
    dynamic_axes={
        "mel_spectrogram": {2: "n_frames"},
        "onset": {1: "n_frames"},
        "note": {1: "n_frames"},
        "velocity": {1: "n_frames"},
    },
    opset_version=17
)
```

**For browser inference:** Use `onnxruntime-web` to run the ONNX model directly in the browser (enables offline and lower-latency live mode).

**Deliverables:** Evaluated model exported to ONNX.

---

## Implementation Order & Dependencies

```
Phase 1 (Setup)
├── Task 1.1: Repo setup                    [No dependencies]
├── Task 1.2: Backend skeleton               [1.1]
└── Task 1.3: Frontend shell                 [1.1]

Phase 2 (Audio Processing)
├── Task 2.1: Audio upload & preprocessing   [1.2]
├── Task 2.2: Tempo & beat detection         [2.1]
└── Task 2.3: Pitch detection (Basic Pitch)  [2.1]

Phase 3 (Notation Engine)
├── Task 3.1: Internal music representation  [No dependencies]
├── Task 3.2: Staff converter                [3.1, 2.3]
├── Task 3.3: Hindustani converter           [3.1, 2.3]
├── Task 3.4: Carnatic converter             [3.1, 2.3]
└── Task 3.5: Orchestrator                   [3.2, 3.3, 3.4, 2.2]

Phase 4 (Frontend Rendering)
├── Task 4.1: Staff renderer (VexFlow)       [3.5, 1.3]
├── Task 4.2: Hindustani renderer (SVG)      [3.5, 1.3]
├── Task 4.3: Carnatic renderer (SVG)        [3.5, 1.3]
└── Task 4.4: Confidence & silence display   [4.1, 4.2, 4.3]

Phase 5 (Live Mode)
├── Task 5.1: Browser audio capture          [1.3]
├── Task 5.2: WebSocket streaming backend    [2.3, 1.2]
└── Task 5.3: Live notation display          [5.1, 5.2, 4.x]

Phase 6 (PDF Export)
└── Task 6.1: PDF generation                 [3.5, 4.x]

Phase 7 (OMR Translation)
├── Task 7.1: OMR engine                     [3.x]
└── Task 7.2: Translation API                [7.1, 3.x]

Phase 8 (Interactive Editor & Playback)
├── Task 8.1: Playback audio engine (Tone.js)        [1.3]
├── Task 8.2: Staff notation editor canvas            [4.1, 8.1]
├── Task 8.3: Carnatic svara editor canvas            [4.3, 8.1]
├── Task 8.4: Detected notes visualization & edit     [8.2, 8.3, 3.5, 5.3]
└── Task 8.5: Unified editor page & toolbar           [8.2, 8.3, 8.4, 6.1]

Phase 9 (Custom ML - Optional)
├── Task 9.1: Dataset preparation            [No dependencies]
├── Task 9.2: Model architecture             [9.1]
└── Task 9.3: Evaluation & ONNX export       [9.2]
```

---

## Key Technical Decisions & Rationale

### Why Basic Pitch over training from scratch?
- Only ~17K parameters (extremely lightweight)
- Polyphonic and instrument-agnostic out of the box
- Available in both Python and TypeScript (npm)
- Apache 2.0 license
- Competitive accuracy against much larger models
- Can be used immediately while custom model trains in parallel

### Why VexFlow for staff notation?
- Most mature JS music notation library (14+ years)
- Supports Canvas and SVG output
- EasyScore API for quick rendering
- Active community and ongoing development (v5)
- MIT licensed

### Why custom SVG for Indian notations?
- No existing library renders Hindustani or Carnatic notation
- The iSargam encoding system provides a reference for digital representation
- SVG gives full control over dots, underlines, and tala markings
- SVG can be directly embedded in PDFs

### Why FastAPI + Python backend?
- All major audio/ML libraries are Python (librosa, Basic Pitch, music21, oemer)
- FastAPI provides native async + WebSocket support
- Pydantic models ensure type safety
- Excellent for ML serving workloads

### Why Tone.js for playback?
- Most mature Web Audio framework (~10+ years, actively maintained)
- Built-in synthesizers (`PolySynth`) and sample-based instruments (`Sampler`)
- `Tone.Transport` provides sample-accurate event scheduling with tempo control
- `Tone.Draw` bridges audio thread timing to UI thread for synchronized visual highlighting
- Supports the Salamander piano sample set for realistic instrument sounds
- Works across desktop and mobile browsers
- MIT licensed

### Why not run everything in the browser?
- Basic Pitch has a TypeScript port (basic-pitch-ts) that CAN run in browser
- However, for production quality: server-side gives access to heavier models, librosa for tempo analysis, music21 for MusicXML handling, and oemer for OMR
- Recommended hybrid: use `basic-pitch-ts` for live mode (fast, low-latency), and server-side for file uploads (higher accuracy, more features)

---

## Estimated Effort

| Phase | Tasks | Estimated Days (1 developer) |
|-------|-------|-----|
| Phase 1: Setup | 3 | 3-4 days |
| Phase 2: Audio Processing | 3 | 5-7 days |
| Phase 3: Notation Engine | 5 | 8-12 days |
| Phase 4: Frontend Rendering | 4 | 10-14 days |
| Phase 5: Live Mode | 3 | 7-10 days |
| Phase 6: PDF Export | 1 | 3-5 days |
| Phase 7: OMR Translation | 2 | 5-7 days |
| Phase 8: Interactive Editor & Playback | 5 | 12-18 days |
| Phase 9: Custom ML (optional) | 3 | 15-25 days |
| **Total** | **29 (core) + 3 (ML)** | **53-77 days (core)** |

---

## Quick Start Commands

```bash
# Clone and setup
git clone <repo>
cd music-transcription-app

# Backend
cd backend
python -m venv venv && source venv/bin/activate
pip install fastapi uvicorn basic-pitch[tf] librosa music21 \
            reportlab svglib pretty-midi oemer numpy scipy
uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend
npm install
npm install vexflow zustand wavesurfer.js tone midi-writer-js
npm run dev

# Docker (recommended)
docker-compose up --build
```