# Sangeet - Musical Audio to Notation

Transcribe musical audio into Staff (Western), Hindustani (Sargam), or Carnatic (Svara) notation. Features live microphone transcription, file upload, playback, a click-to-place editor, and PDF export.

## Quick Start

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### Frontend

```bash
cd frontend
npm install
npm run dev
```

- App: http://localhost:3000

## Features

| Feature | Route | Description |
|---------|-------|-------------|
| Upload | `/` | Upload audio file, get notation |
| Live | `/live` | Real-time mic → sargam notation |
| Editor | `/editor` | Click-to-place notes on a grid |

### Notation Types
- **Staff (Western)** — VexFlow SVG rendering with clefs, time signatures
- **Hindustani (Sargam)** — Sa Re Ga Ma Pa Dha Ni with komal underlines, tivra overlines, octave dots
- **Carnatic (Svara)** — Sa Ri Ga Ma Pa Dha Ni with R1/R2/R3 variant subscripts

### Additional Features
- Tone.js playback with note highlighting
- PDF export (html2canvas + jsPDF)
- WebSocket live streaming with sliding window pitch detection

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 15, React 19, TypeScript, Tailwind CSS |
| State | Zustand |
| Notation | VexFlow 4 (staff), Custom SVG (Indian) |
| Playback | Tone.js (PolySynth) |
| Backend | Python 3.11+, FastAPI |
| Pitch Detection | basic-pitch (ONNX/CoreML) |
| Tempo/Beat | librosa |
| PDF Export | html2canvas + jsPDF (client-side) |
| Real-time | WebSockets (FastAPI) |

## Running Tests

```bash
cd backend
python -m pytest tests/ -v
```
