from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "Musical Notation Transcriber"
    debug: bool = True
    cors_origins: list[str] = ["http://localhost:3000"]
    sample_rate: int = 22050
    default_base_pitch: int = 60  # Middle C (Sa)
    default_notation_type: str = "staff"
    live_window_seconds: float = 2.0
    live_hop_seconds: float = 1.0
    confidence_threshold: float = 0.3


settings = Settings()
