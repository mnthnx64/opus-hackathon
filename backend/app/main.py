"""FastAPI application — main entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api import transcribe, live, config
from app.services.pitch_detector import pitch_detector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML model at startup to avoid cold-start on first request."""
    logger.info("Loading Basic Pitch model...")
    try:
        pitch_detector.load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.warning("Failed to load model at startup (will retry on first request): %s", e)
    yield
    logger.info("Shutting down")


app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(transcribe.router)
app.include_router(live.router)
app.include_router(config.router)


@app.get("/health")
async def health_check():
    return {"status": "ok"}
