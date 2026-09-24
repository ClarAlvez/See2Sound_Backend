import json
from pathlib import Path
from typing import Any, Dict

from API.core.config import settings
from API.services.job_store import (
    update_job,
    utc_now,
)


def run_generation_job(
    job_id: str,
    video_path: str,
    options: Dict[str, Any],
):
    update_job(
        job_id,
        status="processing",
        started_at=utc_now(),
        error=None,
    )

    output_dir = (
        settings.generations_dir
        / job_id
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        from pipeline.orchestration.process_video import (
            process_video,
        )

        result = process_video(
            video_path=video_path,
            output_base_dir=str(
                output_dir
            ),
            frame_interval_seconds=options.get(
                "frame_interval_seconds",
                1.0,
            ),
            whisper_language=options.get(
                "whisper_language"
            ),
            run_spectra=options.get(
                "run_spectra",
                True,
            ),
            run_narrative=options.get(
                "run_narrative",
                True,
            ),
            run_tts=options.get(
                "run_tts",
                True,
            ),
        )

        result_path = (
            output_dir
            / "result.json"
        )

        with result_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                result,
                file,
                indent=4,
                ensure_ascii=False,
                default=str,
            )

        update_job(
            job_id,
            status="completed",
            finished_at=utc_now(),
            result_path=str(
                result_path
            ),
        )

    except Exception as error:
        update_job(
            job_id,
            status="failed",
            finished_at=utc_now(),
            error=str(error),
        )


def load_generation_result(
    job_id: str,
    result_path: str,
) -> Dict[str, Any]:
    path = Path(result_path)

    if not path.exists():
        raise FileNotFoundError(
            "Arquivo de resultado não encontrado."
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)