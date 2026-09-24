import json
from pathlib import Path
from typing import Any, Dict, List

from ai.audio_description.tts_client import (
    TTSClient,
)

from ai.voice_engine.audio_mixer import (
    VoiceAudioMixer,
)

from ai.voice_engine.data_models import (
    VoiceCue,
    VoiceEngineResult,
)

from ai.voice_engine.pause_scheduler import (
    PauseScheduler,
)


class VoiceEngine:
    def __init__(
        self,
        output_dir: str,
        tts_rate: int = 170,
        tts_volume: float = 1.0,
        min_pause_duration: float = 0.5,
        background_volume: float = 0.45,
        description_volume: float = 1.0,
    ):
        self.output_dir = Path(
            output_dir
        )

        self.cues_dir = (
            self.output_dir
            / "cues"
        )

        self.manifest_dir = (
            self.output_dir
            / "manifests"
        )

        self.cues_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.manifest_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.tts = TTSClient(
            rate=tts_rate,
            volume=tts_volume,
        )

        self.scheduler = PauseScheduler(
            min_pause_duration=(
                min_pause_duration
            )
        )

        self.mixer = VoiceAudioMixer()

        self.background_volume = (
            background_volume
        )

        self.description_volume = (
            description_volume
        )

    def generate(
        self,
        original_audio_path: str,
        narrative_timeline: List[
            Dict[str, Any]
        ],
        speech_segments: List[
            Dict[str, Any]
        ],
        total_duration: float,
    ) -> VoiceEngineResult:

        # ====================================================
        # Monta todas as regiões sem fala
        # ====================================================

        pauses = (
            self.scheduler
            .build_available_pauses(
                speech_segments=(
                    speech_segments
                ),
                total_duration=(
                    total_duration
                ),
            )
        )

        print(
            "[Voice Engine DEBUG] "
            f"Pausas disponíveis: {pauses}"
        )

        cues: List[VoiceCue] = []

        # Intervalos já ocupados por audiodescrições.
        # Isso permite várias descrições dentro
        # da mesma pausa, sem sobreposição.
        occupied_intervals = []

        # ====================================================
        # Processa cada descrição narrativa
        # ====================================================

        for index, item in enumerate(
            narrative_timeline
        ):
            text = (
                item.get(
                    "description",
                    "",
                )
                or ""
            ).strip()

            if not text:
                continue

            scene_start = item.get(
                "start_time"
            )

            scene_end = item.get(
                "end_time"
            )

            cue = VoiceCue(
                index=index,
                text=text,
                scene_start_time=(
                    scene_start
                ),
                scene_end_time=(
                    scene_end
                ),
            )

            tts_path = (
                self.cues_dir
                / f"cue_{index:04d}.wav"
            )

            try:
                # ============================================
                # Gera TTS
                # ============================================

                print(
                    "[Voice Engine] "
                    f"Gerando cue #{index}: "
                    f"{text}"
                )

                generated_path = (
                    self.tts.save_to_file(
                        text=text,
                        output_path=str(
                            tts_path
                        ),
                    )
                )

                cue.audio_path = str(
                    generated_path
                )

                # ============================================
                # Obtém duração real do TTS
                # ============================================

                narration_duration = (
                    self.mixer.get_duration(
                        str(generated_path)
                    )
                )

                cue.tts_duration = round(
                    narration_duration,
                    3,
                )

                print(
                    "[Voice Engine DEBUG] "
                    f"Cue #{index}: "
                    f"{cue.tts_duration}s"
                )

                # Um arquivo sem duração não é utilizável.
                if narration_duration <= 0:
                    cue.inserted = False
                    cue.skip_reason = (
                        "O TTS gerou um áudio "
                        "com duração inválida."
                    )

                    cues.append(
                        cue
                    )

                    continue

                # ============================================
                # Procura espaço livre
                # ============================================

                selected_slot = (
                    self.scheduler
                    .find_slot(
                        pauses=pauses,
                        occupied_intervals=(
                            occupied_intervals
                        ),
                        narration_duration=(
                            narration_duration
                        ),
                        scene_start=(
                            scene_start
                        ),
                        scene_end=(
                            scene_end
                        ),
                    )
                )

                if not selected_slot:
                    cue.inserted = False

                    cue.skip_reason = (
                        "Nenhum espaço de áudio "
                        "suficientemente longo "
                        "foi encontrado."
                    )

                    cues.append(
                        cue
                    )

                    print(
                        "[Voice Engine] "
                        f"Cue #{index} pulado: "
                        f"{cue.skip_reason}"
                    )

                    continue

                # ============================================
                # Registra posição escolhida
                # ============================================

                cue.pause_start = (
                    selected_slot[
                        "pause_start"
                    ]
                )

                cue.pause_end = (
                    selected_slot[
                        "pause_end"
                    ]
                )

                cue.playback_start = (
                    selected_slot[
                        "playback_start"
                    ]
                )

                cue.playback_end = (
                    selected_slot[
                        "playback_end"
                    ]
                )

                cue.inserted = True
                cue.skip_reason = None

                # ============================================
                # Reserva somente o trecho realmente ocupado
                # ============================================

                occupied_intervals.append(
                    (
                        selected_slot[
                            "occupied_start"
                        ],
                        selected_slot[
                            "occupied_end"
                        ],
                    )
                )

                occupied_intervals.sort(
                    key=lambda interval: (
                        interval[0]
                    )
                )

                print(
                    "[Voice Engine] "
                    f"Cue #{index} inserido em "
                    f"{cue.playback_start:.3f}s "
                    f"→ {cue.playback_end:.3f}s"
                )

            except Exception as error:
                cue.inserted = False

                cue.skip_reason = (
                    f"{type(error).__name__}: "
                    f"{error}"
                )

                print(
                    "[Voice Engine ERROR] "
                    f"Cue #{index}: "
                    f"{cue.skip_reason}"
                )

            cues.append(
                cue
            )

        # ====================================================
        # Mixagem final
        # ====================================================

        modified_audio_path = (
            self.output_dir
            / "audio_with_description.wav"
        )

        print(
            "[Voice Engine] "
            "Iniciando mixagem final..."
        )

        modified_audio_path = (
            self.mixer.mix(
                original_audio_path=(
                    original_audio_path
                ),
                cues=cues,
                output_path=str(
                    modified_audio_path
                ),
                background_volume=(
                    self.background_volume
                ),
                description_volume=(
                    self.description_volume
                ),
            )
        )

        # ====================================================
        # Estatísticas
        # ====================================================

        inserted = sum(
            1
            for cue in cues
            if cue.inserted
        )

        skipped = (
            len(cues)
            - inserted
        )

        print(
            "[Voice Engine] "
            f"Descrições: {len(cues)} | "
            f"Inseridas: {inserted} | "
            f"Puladas: {skipped}"
        )

        # ====================================================
        # Manifest
        # ====================================================

        manifest_path = (
            self.manifest_dir
            / "voice_engine_manifest.json"
        )

        result = VoiceEngineResult(
            source_audio_path=(
                original_audio_path
            ),
            modified_audio_path=str(
                modified_audio_path
            ),
            manifest_path=str(
                manifest_path
            ),
            total_descriptions=len(
                cues
            ),
            inserted_descriptions=(
                inserted
            ),
            skipped_descriptions=(
                skipped
            ),
            cues=cues,
        )

        with manifest_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                result.to_dict(),
                file,
                ensure_ascii=False,
                indent=4,
            )

        print(
            "[Voice Engine] "
            f"Áudio final: "
            f"{modified_audio_path}"
        )

        return result