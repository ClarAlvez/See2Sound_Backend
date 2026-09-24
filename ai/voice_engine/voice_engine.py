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

        self.scheduler = (
            PauseScheduler(
                min_pause_duration=(
                    min_pause_duration
                )
            )
        )

        self.mixer = (
            VoiceAudioMixer()
        )

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

        cues = []

        used_pause_indexes = set()

        for index, item in enumerate(
            narrative_timeline
        ):
            text = (
                item.get(
                    "description",
                    ""
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
                / (
                    f"cue_"
                    f"{index:04d}.wav"
                )
            )

            try:
                generated_path = (
                    self.tts.save_to_file(
                        text=text,
                        output_path=str(
                            tts_path
                        ),
                    )
                )

                cue.audio_path = (
                    generated_path
                )

                narration_duration = (
                    self.mixer.get_duration(
                        generated_path
                    )
                )

                cue.tts_duration = (
                    round(
                        narration_duration,
                        3,
                    )
                )

                selected_pause = (
                    self.scheduler
                    .find_pause(
                        pauses=pauses,
                        used_pause_indexes=(
                            used_pause_indexes
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

                if not selected_pause:
                    cue.skip_reason = (
                        "Nenhuma pausa "
                        "suficientemente longa "
                        "foi encontrada."
                    )

                    cues.append(
                        cue
                    )

                    continue

                pause_index = (
                    selected_pause[
                        "pause_index"
                    ]
                )

                used_pause_indexes.add(
                    pause_index
                )

                cue.pause_start = (
                    selected_pause[
                        "start"
                    ]
                )

                cue.pause_end = (
                    selected_pause[
                        "end"
                    ]
                )

                cue.playback_start = (
                    selected_pause[
                        "playback_start"
                    ]
                )

                cue.playback_end = (
                    selected_pause[
                        "playback_end"
                    ]
                )

                cue.inserted = True

            except Exception as error:
                cue.inserted = False

                cue.skip_reason = str(
                    error
                )

            cues.append(
                cue
            )

        modified_audio_path = (
            self.output_dir
            / "audio_with_description.wav"
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

        inserted = sum(
            1
            for cue in cues
            if cue.inserted
        )

        skipped = (
            len(cues)
            - inserted
        )

        manifest_path = (
            self.manifest_dir
            / "voice_engine_manifest.json"
        )

        result = VoiceEngineResult(
            source_audio_path=(
                original_audio_path
            ),
            modified_audio_path=(
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

        return result