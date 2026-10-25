
import os
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


def run_xclip_comparison(
    video_path,
    spectra_outputs,
):
    if not spectra_outputs:
        return {
            "scenes": [],
            "summary": build_summary([]),
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

                result = verifier.predict_clip(
                    capture=capture,
                    timestamp=timestamp,
                    scene_start=scene_start,
                    scene_end=scene_end,
                )

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
            }

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

    print(
        "[XCLIP] Comparação concluída:",
        summary,
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
    **pipeline_options,
):
    # Sem Spectra ou sem X-CLIP,
    # preserva o pipeline tradicional.
    if not run_spectra or not run_xclip:
        return process_video(
            video_path=video_path,
            output_base_dir=output_base_dir,
            run_spectra=run_spectra,
            run_narrative=run_narrative,
            run_tts=run_tts,
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
        **pipeline_options,
    )

    spectra = result.setdefault(
        "spectra", {}
    )

    spectra_outputs = spectra.get(
        "outputs", []
    )

    verified = run_xclip_comparison(
        video_path,
        spectra_outputs,
    )

    spectra_outputs = verified["scenes"]
    summary = verified["summary"]

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
            FusionNarrativeGenerator(
                model_path=model_path
            )
        )

        narrative_timeline = (
            generator
            .generate_timeline_from_dicts(
                spectra_outputs=(
                    spectra_outputs
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

    return result
