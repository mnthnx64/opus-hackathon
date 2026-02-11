"""WebSocket /ws/live — real-time audio streaming and transcription."""

from __future__ import annotations

import logging
import struct

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.orchestrator import transcribe_live_chunk
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/live")
async def live_transcription(websocket: WebSocket):
    """Handle live audio streaming via WebSocket.

    Client sends binary audio chunks (float32 PCM at 22050Hz).
    Server accumulates a sliding window and sends back detected notes.
    """
    await websocket.accept()
    logger.info("Live WebSocket connected")

    sr = settings.sample_rate
    window_samples = int(settings.live_window_seconds * sr)
    hop_samples = int(settings.live_hop_seconds * sr)

    audio_buffer = np.array([], dtype=np.float32)
    base_pitch = 60
    notation_type = "hindustani"

    try:
        while True:
            data = await websocket.receive()

            # Handle text messages (config)
            if "text" in data:
                import json
                msg = json.loads(data["text"])
                if msg.get("type") == "config":
                    base_pitch = msg.get("base_pitch", 60)
                    notation_type = msg.get("notation_type", "hindustani")
                    logger.info("Config updated: base_pitch=%d, notation_type=%s",
                                base_pitch, notation_type)
                    continue

            # Handle binary messages (audio data)
            if "bytes" in data:
                raw = data["bytes"]
                # Decode float32 PCM
                n_samples = len(raw) // 4
                samples = np.array(
                    struct.unpack(f"<{n_samples}f", raw),
                    dtype=np.float32,
                )
                audio_buffer = np.concatenate([audio_buffer, samples])

                # Process when we have enough data
                if len(audio_buffer) >= window_samples:
                    chunk = audio_buffer[:window_samples]
                    # Slide the window
                    audio_buffer = audio_buffer[hop_samples:]

                    try:
                        result = transcribe_live_chunk(
                            chunk, sr, base_pitch, notation_type
                        )
                        await websocket.send_json({
                            "type": "notes",
                            "data": result.to_dict(),
                        })
                    except Exception as e:
                        logger.warning("Live transcription error: %s", e)
                        await websocket.send_json({
                            "type": "error",
                            "message": str(e),
                        })

    except WebSocketDisconnect:
        logger.info("Live WebSocket disconnected")
    except Exception as e:
        logger.exception("Live WebSocket error")
        try:
            await websocket.close()
        except Exception:
            pass
