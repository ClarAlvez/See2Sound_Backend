
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
        from pipeline.orchestration.verified_pipeline import (
            process_video_verified,
        )

        run_spectra = options.get(
            "run_spectra",
            True,
        )

        run_narrative = options.get(
            "run_narrative",
            True,
        )

        run_tts = options.get(
            "run_tts",
            True,
        )

        settings.validate_model_files(
            run_spectra=run_spectra,
            run_narrative=run_narrative,
        )

        print(
            "[API] Iniciando processamento."
        )

        if run_spectra:
            print(
                "[API] Spectra Actions:",
                settings.action_model_path,
            )

            print(
                "[API] X-CLIP 16 Frames: ativado"
            )

        result = process_video_verified(
            video_path=video_path,
            output_base_dir=str(
                output_dir
            ),

            # Frames
            frame_interval_seconds=(
                options.get(
                    "frame_interval_seconds",
                    1.0,
                )
            ),

            # Whisper
            whisper_language=(
                options.get(
                    "whisper_language"
                )
            ),

            # Modelos Spectra
            scene_model_path=str(
                settings.scene_model_path
            ),
            person_model_path=str(
                settings.person_model_path
            ),
            object_model_path=str(
                settings.object_model_path
            ),
            atmosphere_model_path=str(
                settings.atmosphere_model_path
            ),
            action_model_path=str(
                settings.action_model_path
            ),

            # Thresholds
            spectra_scene_threshold=0.45,
            spectra_person_threshold=0.50,
            spectra_object_threshold=0.35,
            spectra_atmosphere_threshold=0.50,
            spectra_action_threshold=0.60,
            spectra_top_k=10,

            # Person
            use_person_model_on_full_frame=False,
            use_person_cropper=True,
            person_cropper_model_name="yolov8n.pt",
            person_cropper_confidence_threshold=0.35,
            person_max_people=3,

            # Object
            use_object_model_on_full_frame=False,
            use_object_cropper=True,
            object_cropper_model_name="yolov8n.pt",
            object_cropper_confidence_threshold=0.25,
            object_max_objects=20,

            # Actions
            use_action_model=True,
            use_action_person_cropper=True,
            action_max_people=5,

            # X-CLIP
            run_xclip=run_spectra,

            # Narrative
            narrative_model_path=str(
                settings.narrative_model_path
            ),

            # Configurações da mixagem
            voice_background_volume=0.18,
            voice_description_volume=1.0,

            # Validação
            strict_model_loading=True,

            # Controle
            run_spectra=run_spectra,
            run_narrative=run_narrative,
            run_tts=run_tts,
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
        print(
            "[API] Geração falhou:",
            repr(error),
        )

        update_job(
            job_id,
            status="failed",
            finished_at=utc_now(),
            error=(
                f"{type(error).__name__}: "
                f"{error}"
            ),
        )


def load_generation_result(
    job_id: str,
    result_path: str,
) -> Dict[str, Any]:
    path = Path(
        result_path
    )

    if not path.exists():
        raise FileNotFoundError(
            "Arquivo de resultado "
            "não encontrado."
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )
