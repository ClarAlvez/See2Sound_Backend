import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from API.core.config import settings


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def get_job_path(
    job_id: str,
) -> Path:
    return (
        settings.jobs_dir
        / f"{job_id}.json"
    )


def create_job(
    job_id: str,
    original_filename: str,
    input_path: str,
    upload_size: int,
    options: Dict[str, Any],
) -> Dict[str, Any]:
    job = {
        "job_id": job_id,
        "status": "queued",
        "original_filename": (
            original_filename
        ),
        "input_path": input_path,
        "upload_size": upload_size,
        "options": options,
        "created_at": utc_now(),
        "started_at": None,
        "finished_at": None,
        "result_path": None,
        "error": None,
    }

    _write_job(job)

    return job


def read_job(
    job_id: str,
) -> Dict[str, Any]:
    path = get_job_path(job_id)

    if not path.exists():
        raise FileNotFoundError(
            f"Job não encontrado: {job_id}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def update_job(
    job_id: str,
    **changes,
) -> Dict[str, Any]:
    job = read_job(job_id)

    job.update(changes)

    _write_job(job)

    return job


def _write_job(
    job: Dict[str, Any],
):
    path = get_job_path(
        job["job_id"]
    )

    temporary_path = path.with_suffix(
        ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            job,
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )

    temporary_path.replace(path)