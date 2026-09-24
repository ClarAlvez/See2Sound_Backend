from typing import Any, Dict, Optional

from pydantic import BaseModel


class GenerationOptions(BaseModel):
    frame_interval_seconds: float = 1.0
    whisper_language: Optional[str] = None
    run_spectra: bool = True
    run_narrative: bool = True
    run_tts: bool = True


class GenerationCreatedResponse(BaseModel):
    job_id: str
    status: str
    message: str

    status_url: str
    result_url: str

    audio_description_url: str
    audio_description_metadata_url: str


class GenerationStatusResponse(BaseModel):
    job_id: str
    status: str
    original_filename: str
    created_at: str

    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None

    options: Dict[str, Any]


class GenerationResultResponse(BaseModel):
    job_id: str
    status: str

    result: Dict[str, Any]
    artifact_urls: Dict[str, str]