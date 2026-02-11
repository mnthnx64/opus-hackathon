"""POST /api/transcribe — upload audio file, get notation back."""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.services.audio_processor import save_upload_to_temp
from app.services.orchestrator import transcribe

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    base_pitch: int = Form(60),
    notation_type: str = Form("staff"),
):
    """Transcribe an uploaded audio file to musical notation."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    temp_path = await save_upload_to_temp(file)
    try:
        result = transcribe(temp_path, base_pitch, notation_type)
        return result.to_dict()
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
