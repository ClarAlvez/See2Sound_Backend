
from ai.spectra.Object.object_intermediate import (
    ObjectCLIPVerifier,
    apply_object_intermediate,
    scene_for_narrative,
)
from ai.narrative.person_fusion_narrative import (
    PersonFusionNarrativeGenerator,
)
from ai.spectra.Person.person_intermediate import (
    PersonCLIPVerifier,
    apply_person_intermediate,
)

import os
import logging
from pathlib import Path

import cv2

from ai.spectra.Actions.xclip_verifier import (
    get_xclip_verifier,
)

from ai.spectra.Actions.action_comparator import (
    compare_scene,
    build_summary,
)

from ai.spectra.Actions.action_fusion import (
    apply_action_correction,
)

from ai.narrative.fusion_narrative import (
    FusionNarrativeGenerator,
)

from pipeline.orchestration.process_video import (
    process_video,
    generate_voice_audio,
    save_json,
)


def corrections_enabled():
    return (
        os.getenv(
            "SEE2SOUND_APPLY_ACTION_CORRECTIONS",
            "false",
        ).strip().lower()
        in ("true", "1", "yes", "on")
    )


def is_xclip_decode_error(exc):
    """
    Identifica somente falhas conhecidas de leitura
    de frames produzidas pelo XCLIPVerifier.

    Não mascara erros de inferência, configuração
    ou execução dos modelos.
    """
    message = str(exc)

    return (
        message.startswith("Falha ao ler frame ")
        or message == "Falha ao avançar no vídeo."
    )


def run_xclip_comparison(
    video_path,
    spectra_outputs,
):
    if not spectra_outputs:
        summary = build_summary([])
        summary["skipped_windows"] = 0

        return {
            "scenes": [],
            "summary": summary,
        }

    verifier = get_xclip_verifier()

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise FileNotFoundError(
            "Vídeo não encontrado para X-CLIP: "
            + str(video_path)
        )

    enabled = corrections_enabled()

    print(
        "[FUSION V2] Correções habilitadas:",
        enabled,
    )

    total_skipped_windows = 0

    try:
        for index, scene in enumerate(
            spectra_outputs
        ):
            print(
                "[XCLIP] Cena",
                index + 1,
                "/",
                len(spectra_outputs),
            )

            scene_start = float(
                scene.get(
                    "start_time", 0.0
                )
            )

            scene_end = float(
                scene.get(
                    "end_time", scene_start
                )
            )

            clips = []
            skipped_windows = []

            for frame in scene.get(
                "raw_frame_results", []
            ):
                timestamp = float(
                    frame["timestamp"]
                )

                print(
                    "[XCLIP] Janela:",
                    timestamp,
                    "s",
                )

                # Uma falha de decodificação localizada
                # não deve interromper toda a geração.
                try:
                    result = verifier.predict_clip(
                        capture=capture,
                        timestamp=timestamp,
                        scene_start=scene_start,
                        scene_end=scene_end,
                    )

                except RuntimeError as exc:
                    if not is_xclip_decode_error(exc):
                        raise

                    total_skipped_windows += 1

                    error_info = {
                        "timestamp": timestamp,
                        "reason": str(exc),
                    }

                    skipped_windows.append(
                        error_info
                    )

                    print(
                        "[XCLIP] AVISO:",
                        "Falha de leitura na cena",
                        index + 1,
                        "tempo",
                        timestamp,
                        "s.",
                        str(exc),
                    )

                    print(
                        "[XCLIP] Ignorando somente "
                        "esta janela e continuando."
                    )

                    continue

                clips.append(result)

            comparison = compare_scene(
                scene,
                clips,
            )

            print(
                "[FUSION V2] Status:",
                comparison["status"],
            )

            print(
                "[FUSION V2] Proposta:",
                comparison["proposal"],
            )

            corrected = apply_action_correction(
                scene=scene,
                comparison=comparison,
                enabled=enabled,
            )

            if corrected is not scene:
                scene.clear()
                scene.update(corrected)

            applied = bool(
                scene.get(
                    "action_correction", {}
                ).get("applied", False)
            )

            comparison[
                "correction_applied"
            ] = applied

            scene["action_verification"] = {
                "model": (
                    "microsoft/"
                    "xclip-base-patch32-16-frames"
                ),
                "mode": (
                    "experimental_fusion_v2"
                    if enabled
                    else "shadow_comparison"
                ),
                "xclip_clips": clips,
                "comparison": comparison,
                "correction_applied": applied,

                # Informações adicionais de diagnóstico.
                "skipped_windows": skipped_windows,
                "skipped_windows_count": len(
                    skipped_windows
                ),
            }

            if skipped_windows:
                print(
                    "[XCLIP] Cena",
                    index + 1,
                    "- janelas ignoradas:",
                    len(skipped_windows),
                )

            print(
                "[FUSION V2] Aplicada:",
                applied,
            )

            print(
                "[FUSION V2] Labels finais:",
                scene.get(
                    "labels", []
                ),
            )

            print(
                "[FUSION V2] Ação preferida:",
                scene.get(
                    "context", {}
                ).get(
                    "preferred_action"
                ),
            )

    finally:
        capture.release()

    summary = build_summary(
        spectra_outputs
    )

    summary["device"] = (
        verifier.device
    )

    summary["model_load_ms"] = (
        verifier.load_ms
    )

    summary["corrections_enabled"] = (
        enabled
    )

    summary["skipped_windows"] = (
        total_skipped_windows
    )

    print(
        "[XCLIP] Comparação concluída:",
        summary,
    )

    if total_skipped_windows > 0:
        print(
            "[XCLIP] AVISO: Foram ignoradas",
            total_skipped_windows,
            "janelas por falhas de leitura.",
        )

    return {
        "scenes": spectra_outputs,
        "summary": summary,
    }


def process_video_verified(
    video_path,
    output_base_dir="data/output",
    run_spectra=True,
    run_narrative=True,
    run_tts=True,
    run_xclip=True,
    progress_callback=None,
    **pipeline_options,
):
    def progress(stage):
        if progress_callback is not None:
            try:
                progress_callback(stage)
            except Exception:
                logging.getLogger(__name__).exception("Progress callback failed: %s", stage)

    def initial_progress(stage):
        # Narrative/Voice have not run yet in the verified path.
        if stage not in {"narrative", "voice_engine"}:
            progress("initial_artifacts" if stage == "saving_artifacts" else stage)

    # Sem Spectra ou sem X-CLIP,
    # preserva o pipeline tradicional.
    if not run_spectra or not run_xclip:
        return process_video(
            video_path=video_path,
            output_base_dir=output_base_dir,
            run_spectra=run_spectra,
            run_narrative=run_narrative,
            run_tts=run_tts,
            progress_callback=progress,
            **pipeline_options,
        )

    # Primeira fase:
    # processar a Spectra e o áudio,
    # mas ainda não gerar Narrative/TTS.
    result = process_video(
        video_path=video_path,
        output_base_dir=output_base_dir,
        run_spectra=True,
        run_narrative=False,
        run_tts=False,
        progress_callback=initial_progress,
        **pipeline_options,
    )

    spectra = result.setdefault(
        "spectra", {}
    )

    spectra_outputs = spectra.get(
        "outputs", []
    )

    progress("xclip")
    verified = run_xclip_comparison(
        video_path,
        spectra_outputs,
    )

    spectra_outputs = verified["scenes"]
    summary = verified["summary"]

    progress("verification_processing")
    # Person Intermediate V1:
    # antes de salvar a Spectra e narrar.
    person_enabled = os.getenv(
        "SEE2SOUND_PERSON_INTERMEDIATE",
        "false",
    ).lower() in (
        "1", "true", "yes", "on"
    )

    if person_enabled:
        clip_verifier = PersonCLIPVerifier()

        detection_cap = int(
            pipeline_options.get(
                "person_max_people", 3
            )
        )

        spectra_outputs = [
            apply_person_intermediate(
                scene,
                verifier=clip_verifier,
                detection_cap=detection_cap,
            )
            for scene in spectra_outputs
        ]

        for index, updated_scene in enumerate(
            spectra_outputs, 1
        ):
            info = updated_scene.get(
                "person_intermediate", {}
            )

            print(
                "[PERSON INTERMEDIATE]",
                index,
                info.get("count_type"),
                info.get("frame_counts"),
            )

    # Object Intermediate V2:
    # verificação independente antes da Narrative.
    object_enabled = os.getenv(
        "SEE2SOUND_OBJECT_INTERMEDIATE",
        "false",
    ).lower() in (
        "true", "1", "yes", "on"
    )

    if object_enabled:
        object_apply = os.getenv(
            "SEE2SOUND_OBJECT_APPLY_REMOVALS",
            "false",
        ).lower() in (
            "true", "1", "yes", "on"
        )

        object_add = os.getenv(
            "SEE2SOUND_OBJECT_APPLY_ADDITIONS",
            "false",
        ).lower() in (
            "true", "1", "yes", "on"
        )

        object_hide = os.getenv(
            "SEE2SOUND_OBJECT_SUPPRESS_NARRATIVE",
            "true",
        ).lower() in (
            "true", "1", "yes", "on"
        )

        verifier_obj = ObjectCLIPVerifier()

        spectra_outputs = [
            apply_object_intermediate(
                scene,
                verifier=verifier_obj,
                apply_removals=object_apply,
                apply_additions=object_add,
                suppress_narrative=object_hide,
            )
            for scene in spectra_outputs
        ]

        for i, scene in enumerate(
            spectra_outputs, 1
        ):
            info = scene.get(
                "object_intermediate", {}
            )

            print(
                "[OBJECT V2]",
                i,
                "add=",
                info.get("proposed_additions"),
                "remove=",
                info.get("proposed_removals"),
                "hide=",
                info.get("narrative_suppression"),
            )

    spectra["outputs"] = (
        spectra_outputs
    )

    spectra[
        "action_verification_summary"
    ] = summary

    root = Path(
        output_base_dir
    )

    spectra_dir = (
        root / "spectra"
    )

    narrative_dir = (
        root / "narrative"
    )

    voice_dir = (
        root / "voice_engine"
    )

    progress("verified_artifacts")
    spectra_path = save_json(
        data=spectra_outputs,
        output_path=(
            spectra_dir
            / "spectra_outputs.json"
        ),
    )

    verification_path = save_json(
        data={
            "summary": summary,
            "scenes": [
                {
                    "start_time": scene.get(
                        "start_time"
                    ),
                    "end_time": scene.get(
                        "end_time"
                    ),
                    "action_verification": (
                        scene.get(
                            "action_verification",
                            {},
                        )
                    ),
                    "action_correction": (
                        scene.get(
                            "action_correction",
                            {},
                        )
                    ),
                }
                for scene in spectra_outputs
            ],
        },
        output_path=(
            spectra_dir
            / "action_verification.json"
        ),
    )

    artifacts = result.setdefault(
        "artifacts", {}
    )

    artifacts[
        "spectra_outputs_json"
    ] = str(spectra_path)

    artifacts[
        "action_verification_json"
    ] = str(verification_path)

    # Narrative recebe as cenas
    # já corrigidas pela fusão.
    narrative_timeline = []

    if run_narrative and spectra_outputs:
        progress("narrative")
        model_path = str(
            pipeline_options.get(
                "narrative_model_path",
                (
                    "data/models/qwen/"
                    "qwen2.5-3b-instruct-"
                    "q4_k_m.gguf"
                ),
            )
        )

        generator = (
            PersonFusionNarrativeGenerator(
                model_path=model_path
            )
        )

        # Objetos duvidosos são ocultados
        # somente da cópia narrativa.
        narrative_scenes = [
            scene_for_narrative(s)
            for s in spectra_outputs
        ]

        narrative_timeline = (
            generator
            .generate_timeline_from_dicts(
                spectra_outputs=(
                    narrative_scenes
                )
            )
        )

    narrative = result.setdefault(
        "narrative", {}
    )

    narrative["timeline"] = (
        narrative_timeline
    )

    narrative_path = save_json(
        data=narrative_timeline,
        output_path=(
            narrative_dir
            / "narrative_timeline.json"
        ),
    )

    artifacts[
        "narrative_timeline_json"
    ] = str(narrative_path)

    # Voice Engine, sem alterar seus
    # parâmetros internos ou modelos.
    voice_result = None

    if run_tts and narrative_timeline:
        progress("voice_engine")
        audio = result.get(
            "audio", {}
        )

        voice_result = generate_voice_audio(
            audio_result=(
                audio["extraction"]
            ),
            whisper_result=(
                audio["speech_analysis"]
            ),
            narrative_timeline=(
                narrative_timeline
            ),
            metadata_result=(
                result["metadata"]
            ),
            output_dir=voice_dir,
            tts_rate=pipeline_options.get(
                "tts_rate", 170
            ),
            tts_volume=pipeline_options.get(
                "tts_volume", 1.0
            ),
            min_pause_duration=(
                pipeline_options.get(
                    "min_pause_duration",
                    0.5,
                )
            ),
            background_volume=(
                pipeline_options.get(
                    "voice_background_volume",
                    0.18,
                )
            ),
            description_volume=(
                pipeline_options.get(
                    "voice_description_volume",
                    1.0,
                )
            ),
        )

    result["voice_engine"] = (
        voice_result
    )

    if voice_result:
        audio_path = voice_result.get(
            "modified_audio_path"
        )

        manifest_path = voice_result.get(
            "manifest_path"
        )

        if audio_path:
            artifacts[
                "audio_description"
            ] = audio_path

        if manifest_path:
            artifacts[
                "voice_manifest"
            ] = manifest_path

    progress("saving_artifacts")
    return result
