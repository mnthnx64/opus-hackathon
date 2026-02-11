"""GET /api/config/* — configuration endpoints."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()

PITCH_OPTIONS = []
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
for octave in range(2, 7):
    for i, name in enumerate(NOTE_NAMES):
        midi = (octave + 1) * 12 + i
        PITCH_OPTIONS.append({
            "midi": midi,
            "name": f"{name}{octave}",
            "label": f"{name}{octave} (MIDI {midi})",
        })

NOTATION_TYPES = [
    {"id": "staff", "name": "Staff (Western)", "description": "Standard Western music notation"},
    {"id": "hindustani", "name": "Hindustani (Sargam)", "description": "North Indian sargam notation with Sa Re Ga Ma Pa Dha Ni"},
    {"id": "carnatic", "name": "Carnatic (Svara)", "description": "South Indian svara notation with Ri/Ga/Dha/Ni variants"},
]

TAAL_OPTIONS = [
    {"id": "teentaal", "name": "Teentaal", "beats": 16, "divisions": "4+4+4+4"},
    {"id": "jhaptaal", "name": "Jhaptaal", "beats": 10, "divisions": "2+3+2+3"},
    {"id": "ektaal", "name": "Ektaal", "beats": 12, "divisions": "2+2+2+2+2+2"},
]

TALA_OPTIONS = [
    {"id": "adi", "name": "Adi Tala", "beats": 8, "divisions": "4+2+2"},
    {"id": "rupaka", "name": "Rupaka Tala", "beats": 3, "divisions": "2+1"},
    {"id": "mishra_chapu", "name": "Mishra Chapu", "beats": 7, "divisions": "3+2+2"},
]


@router.get("/api/config/pitches")
async def get_pitches():
    return PITCH_OPTIONS


@router.get("/api/config/notation-types")
async def get_notation_types():
    return NOTATION_TYPES


@router.get("/api/config/taals")
async def get_taals():
    return TAAL_OPTIONS


@router.get("/api/config/talas")
async def get_talas():
    return TALA_OPTIONS
