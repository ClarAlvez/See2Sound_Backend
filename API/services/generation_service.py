
import json
import logging
from pathlib import Path
from typing import Any, Dict

from ai.voice_engine.data_models import VoiceGenerationError

from API.core.config import settings
from API.core.errors import StoredDataError

from API.services.job_store import (
    update_job,
    utc_now,
)

logger = logging.getLogger(__name__)


def run_generation_job(
    job_id: str,
    video_path: str,
    options: Dict[str, Any],
):
    try:
        update_job(
            job_id,
            status="processing",
            started_at=utc_now(),
            error=None,
        )

        output_dir = settings.generations_dir / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

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

        if run_tts and run_narrative and result.get("voice_engine") is None:
            timeline = result.get("narrative", {}).get("timeline", [])
            if not any((item.get("description") or "").strip() for item in timeline):
                logger.info("job=%s voice_outcome=no_descriptions", job_id)

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
        logger.exception("job=%s generation_failed", job_id)
        try:
            update_job(
                job_id,
                status="failed",
                finished_at=utc_now(),
                error=(str(error) if isinstance(error, VoiceGenerationError)
                       else f"{type(error).__name__}: {error}"),
            )
        except Exception:
            logger.exception("job=%s failed_state_persistence_failed", job_id)
            # Preserve the processing/preparation error as the primary failure.
            raise error from error.__cause__


def load_generation_result(
    job_id: str,
    result_path: str,
) -> Dict[str, Any]:
    try:
        path = Path(result_path)
        with path.open("r", encoding="utf-8") as file:
            result = json.load(file)
        if not isinstance(result, dict):
            raise ValueError("Result must be an object")
        if not isinstance(result.get("artifacts", {}), dict):
            raise ValueError("Invalid artifacts container")
        if result.get("voice_engine") is not None and not isinstance(result["voice_engine"], dict):
            raise ValueError("Invalid voice result container")
        return result
    except FileNotFoundError as error:
        raise StoredDataError("Arquivo de resultado não encontrado.") from error
    except (OSError, ValueError, TypeError) as error:
        raise StoredDataError("Resultado da geração está ilegível. Consulte os logs do servidor.") from error
