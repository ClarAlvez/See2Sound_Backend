from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse

from API.core.config import settings
from API.schemas.generation import (
    GenerationCreatedResponse,
    GenerationOptions,
    GenerationResultResponse,
    GenerationStatusResponse,
)
from API.services.generation_service import (
    load_generation_result,
    run_generation_job,
)
from API.services.job_store import (
    create_job,
    read_job,
)
from API.utils.files import (
    UploadTooLargeError,
    save_upload_file,
)


router = APIRouter(
    prefix="/api/v1/generations",
    tags=["Generations"],
)


# ============================================================
# Criar geração
# ============================================================


@router.post(
    "",
    response_model=GenerationCreatedResponse,
    status_code=202,
)
async def create_generation(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    frame_interval_seconds: float = Form(1.0),
    whisper_language: Optional[str] = Form(None),
    run_spectra: bool = Form(True),
    run_narrative: bool = Form(True),
    run_tts: bool = Form(True),
):
    original_filename = (
        video.filename
        or "video"
    )

    extension = Path(
        original_filename
    ).suffix.lower()

    if (
        extension
        not in settings.allowed_video_extensions
    ):
        raise HTTPException(
            status_code=415,
            detail=(
                "Formato de vídeo não suportado. "
                "Formatos aceitos: "
                + ", ".join(
                    sorted(
                        settings.allowed_video_extensions
                    )
                )
            ),
        )

    if frame_interval_seconds <= 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "frame_interval_seconds "
                "deve ser maior que zero."
            ),
        )

    job_id = uuid4().hex

    video_path = (
        settings.upload_dir
        / f"{job_id}{extension}"
    )

    try:
        upload_size = await save_upload_file(
            upload=video,
            destination=video_path,
            max_size_bytes=(
                settings.max_upload_size_bytes
            ),
        )

    except UploadTooLargeError as error:
        raise HTTPException(
            status_code=413,
            detail=str(error),
        )

    options = GenerationOptions(
        frame_interval_seconds=(
            frame_interval_seconds
        ),
        whisper_language=(
            whisper_language
        ),
        run_spectra=run_spectra,
        run_narrative=run_narrative,
        run_tts=run_tts,
    )

    options_dict = (
        options.model_dump()
    )

    create_job(
        job_id=job_id,
        original_filename=(
            original_filename
        ),
        input_path=str(
            video_path
        ),
        upload_size=upload_size,
        options=options_dict,
    )

    background_tasks.add_task(
        run_generation_job,
        job_id,
        str(video_path),
        options_dict,
    )

    return GenerationCreatedResponse(
        job_id=job_id,

        status="queued",

        message=(
            "Vídeo recebido e "
            "geração iniciada."
        ),

        status_url=(
            f"/api/v1/generations/"
            f"{job_id}"
        ),

        result_url=(
            f"/api/v1/generations/"
            f"{job_id}/result"
        ),

        audio_description_url=(
            f"/api/v1/generations/"
            f"{job_id}/audio-description"
        ),

        audio_description_metadata_url=(
            f"/api/v1/generations/"
            f"{job_id}"
            "/audio-description/metadata"
        ),
    )


# ============================================================
# Status
# ============================================================


@router.get(
    "/{job_id}",
    response_model=GenerationStatusResponse,
)
def get_generation_status(
    job_id: str,
):
    try:
        job = read_job(
            job_id
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=(
                "Geração não encontrada."
            ),
        )

    return GenerationStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        original_filename=(
            job["original_filename"]
        ),
        created_at=(
            job["created_at"]
        ),
        started_at=job.get(
            "started_at"
        ),
        finished_at=job.get(
            "finished_at"
        ),
        error=job.get(
            "error"
        ),
        options=job.get(
            "options",
            {},
        ),
    )


# ============================================================
# Resultado completo
# ============================================================


@router.get(
    "/{job_id}/result",
    response_model=GenerationResultResponse,
)
def get_generation_result(
    job_id: str,
):
    try:
        job = read_job(
            job_id
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=(
                "Geração não encontrada."
            ),
        )

    if job["status"] == "failed":
        raise HTTPException(
            status_code=500,
            detail=job.get(
                "error",
                "A geração falhou.",
            ),
        )

    if job["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail=(
                "A geração ainda "
                "não terminou."
            ),
        )

    result_path = job.get(
        "result_path"
    )

    if not result_path:
        raise HTTPException(
            status_code=500,
            detail=(
                "Resultado da geração "
                "não encontrado."
            ),
        )

    try:
        result = (
            load_generation_result(
                job_id=job_id,
                result_path=(
                    result_path
                ),
            )
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail=(
                "Arquivo de resultado "
                "não encontrado."
            ),
        )

    artifacts = result.get(
        "artifacts",
        {},
    )

    artifact_urls = {
        artifact_name: (
            f"/api/v1/generations/"
            f"{job_id}/artifacts/"
            f"{artifact_name}"
        )
        for artifact_name
        in artifacts
    }

    return GenerationResultResponse(
        job_id=job_id,
        status="completed",
        result=result,
        artifact_urls=(
            artifact_urls
        ),
    )


# ============================================================
# Artefatos
# ============================================================


@router.get(
    "/{job_id}/artifacts/{artifact_name}"
)
def download_artifact(
    job_id: str,
    artifact_name: str,
):
    try:
        job = read_job(
            job_id
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=(
                "Geração não encontrada."
            ),
        )

    if job["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail=(
                "A geração ainda "
                "não terminou."
            ),
        )

    result_path = job.get(
        "result_path"
    )

    if not result_path:
        raise HTTPException(
            status_code=404,
            detail=(
                "Resultado não encontrado."
            ),
        )

    result = (
        load_generation_result(
            job_id=job_id,
            result_path=result_path,
        )
    )

    artifacts = result.get(
        "artifacts",
        {},
    )

    artifact_path_value = (
        artifacts.get(
            artifact_name
        )
    )

    if not artifact_path_value:
        raise HTTPException(
            status_code=404,
            detail=(
                "Artefato não encontrado."
            ),
        )

    artifact_path = Path(
        artifact_path_value
    ).resolve()

    generation_root = (
        settings.generations_dir
        / job_id
    ).resolve()

    if (
        artifact_path
        != generation_root
        and generation_root
        not in artifact_path.parents
    ):
        raise HTTPException(
            status_code=403,
            detail=(
                "Acesso ao arquivo negado."
            ),
        )

    if not artifact_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Arquivo não encontrado."
            ),
        )

    return FileResponse(
        path=str(
            artifact_path
        ),
        filename=(
            artifact_path.name
        ),
    )


# ============================================================
# Áudio final com audiodescrição
# ============================================================


@router.get(
    "/{job_id}/audio-description"
)
def get_audio_description(
    job_id: str,
):
    try:
        job = read_job(
            job_id
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=(
                "Geração não encontrada."
            ),
        )

    if job["status"] == "failed":
        raise HTTPException(
            status_code=500,
            detail=job.get(
                "error",
                "A geração falhou.",
            ),
        )

    if job["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail=(
                "A geração ainda "
                "não terminou."
            ),
        )

    result_path = job.get(
        "result_path"
    )

    if not result_path:
        raise HTTPException(
            status_code=404,
            detail=(
                "Resultado não encontrado."
            ),
        )

    result = (
        load_generation_result(
            job_id=job_id,
            result_path=(
                result_path
            ),
        )
    )

    voice_result = result.get(
        "voice_engine"
    )

    if not voice_result:
        raise HTTPException(
            status_code=404,
            detail=(
                "Audiodescrição "
                "não foi gerada."
            ),
        )

    audio_path_value = (
        voice_result.get(
            "modified_audio_path"
        )
    )

    if not audio_path_value:
        raise HTTPException(
            status_code=404,
            detail=(
                "Arquivo de áudio "
                "não encontrado."
            ),
        )

    audio_path = Path(
        audio_path_value
    ).resolve()

    generation_root = (
        settings.generations_dir
        / job_id
    ).resolve()

    if (
        audio_path
        != generation_root
        and generation_root
        not in audio_path.parents
    ):
        raise HTTPException(
            status_code=403,
            detail=(
                "Acesso ao arquivo "
                "negado."
            ),
        )

    if not audio_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Arquivo de áudio "
                "não existe."
            ),
        )

    original_name = Path(
        job["original_filename"]
    ).stem

    filename = (
        f"{original_name}"
        "_audiodescription.wav"
    )

    return FileResponse(
        path=str(
            audio_path
        ),
        media_type="audio/wav",
        filename=filename,
    )


# ============================================================
# Informações da audiodescrição
# ============================================================


@router.get(
    "/{job_id}/audio-description/metadata"
)
def get_audio_description_metadata(
    job_id: str,
):
    try:
        job = read_job(
            job_id
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=(
                "Geração não encontrada."
            ),
        )

    if job["status"] == "failed":
        raise HTTPException(
            status_code=500,
            detail=job.get(
                "error",
                "A geração falhou.",
            ),
        )

    if job["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail=(
                "A geração ainda "
                "não terminou."
            ),
        )

    result_path = job.get(
        "result_path"
    )

    if not result_path:
        raise HTTPException(
            status_code=404,
            detail=(
                "Resultado não encontrado."
            ),
        )

    result = (
        load_generation_result(
            job_id=job_id,
            result_path=(
                result_path
            ),
        )
    )

    voice_result = result.get(
        "voice_engine"
    )

    if not voice_result:
        raise HTTPException(
            status_code=404,
            detail=(
                "Audiodescrição "
                "não foi gerada."
            ),
        )

    return {
        "job_id": job_id,

        "status": "completed",

        "total_descriptions": (
            voice_result.get(
                "total_descriptions",
                0,
            )
        ),

        "inserted_descriptions": (
            voice_result.get(
                "inserted_descriptions",
                0,
            )
        ),

        "skipped_descriptions": (
            voice_result.get(
                "skipped_descriptions",
                0,
            )
        ),

        "cues": (
            voice_result.get(
                "cues",
                [],
            )
        ),

        "audio_url": (
            f"/api/v1/generations/"
            f"{job_id}"
            "/audio-description"
        ),
    }