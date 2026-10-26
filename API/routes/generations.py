import math
import logging
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

from starlette.concurrency import run_in_threadpool
from API.services.runtime import (
    reserve_generation_slot, release_generation_slot, run_admitted_generation,
)
from API.utils.video_validation import (
    InvalidVideoError, VideoValidationUnavailable, validate_uploaded_video,
)

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


logger = logging.getLogger(__name__)

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
    original_filename = video.filename or "video"
    extension = Path(original_filename).suffix.lower()
    job_id = uuid4().hex
    video_path = settings.upload_dir / f"{job_id}{extension}"
    accepted = False
    reserved = False
    saved = False
    try:
        if extension not in settings.allowed_video_extensions:
            raise HTTPException(415, "Formato de vídeo não suportado. Formatos aceitos: " +
                                ", ".join(sorted(settings.allowed_video_extensions)))
        if not math.isfinite(frame_interval_seconds) or frame_interval_seconds <= 0:
            raise HTTPException(400, "frame_interval_seconds deve ser finito e maior que zero.")
        options_dict = GenerationOptions(
            frame_interval_seconds=frame_interval_seconds,
            whisper_language=whisper_language, run_spectra=run_spectra,
            run_narrative=run_narrative, run_tts=run_tts,
        ).model_dump()
        if not reserve_generation_slot():
            raise HTTPException(503, "Fila de processamento cheia. Tente novamente em instantes.",
                                headers={"Retry-After": "10"})
        reserved = True
        upload_size = await save_upload_file(video, video_path, settings.max_upload_size_bytes)
        saved = True
        metadata = await run_in_threadpool(validate_uploaded_video, video_path)
        logger.info("job=%s stage=upload_validated bytes=%s duration_seconds=%.3f",
                    job_id, upload_size, metadata["duration_seconds"])
        await run_in_threadpool(create_job, job_id=job_id, original_filename=original_filename,
                                input_path=str(video_path), upload_size=upload_size, options=options_dict)
        background_tasks.add_task(run_admitted_generation, job_id, str(video_path), options_dict, run_generation_job)
        accepted = True
    except UploadTooLargeError as error:
        logger.warning("job=%s stage=upload rejected reason=size", job_id)
        raise HTTPException(413, str(error)) from error
    except InvalidVideoError as error:
        logger.warning("job=%s stage=video_validation rejected", job_id)
        raise HTTPException(400, str(error)) from error
    except VideoValidationUnavailable as error:
        raise HTTPException(503, str(error)) from error
    except OSError as error:
        logger.exception("job=%s stage=upload storage_failed", job_id)
        raise HTTPException(503, "Armazenamento indisponível. Verifique o espaço e as permissões do servidor.") from error
    finally:
        try:
            await video.close()
        except OSError:
            logger.exception("job=%s stage=upload close_failed", job_id)
        if reserved and not accepted:
            release_generation_slot()
        if saved and not accepted:
            try:
                await run_in_threadpool(video_path.unlink, missing_ok=True)
            except OSError:
                logger.exception("job=%s stage=upload cleanup_failed", job_id)

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
        error=job.get("error"),
        stage=job.get("stage"),
        stage_durations_seconds=job.get("stage_durations_seconds", {}),
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

    if job["status"] == "failed":
        raise HTTPException(
            status_code=409,
            detail="A geração falhou. Consulte o status da geração para detalhes.",
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

    if not artifact_path.is_file() or artifact_path.stat().st_size == 0:
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

    if not audio_path.is_file() or audio_path.stat().st_size == 0:
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
