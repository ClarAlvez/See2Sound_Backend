from math import gcd
from pathlib import Path
from typing import List

import numpy as np

from scipy.io import wavfile
from scipy.signal import resample_poly

from ai.voice_engine.data_models import (
    VoiceCue,
)


class VoiceAudioMixer:
    def mix(
        self,
        original_audio_path: str,
        cues: List[VoiceCue],
        output_path: str,
        background_volume: float = 0.45,
        description_volume: float = 1.0,
    ) -> str:
        original_rate, original_audio = (
            wavfile.read(
                original_audio_path
            )
        )

        original_float = (
            self._to_float(
                original_audio
            )
        )

        if original_float.ndim == 1:
            channel_count = 1

        else:
            channel_count = (
                original_float.shape[1]
            )

        output_audio = (
            original_float.copy()
        )

        for cue in cues:
            if not cue.inserted:
                continue

            if not cue.audio_path:
                continue

            if cue.playback_start is None:
                continue

            narration_rate, narration = (
                wavfile.read(
                    cue.audio_path
                )
            )

            narration = self._to_float(
                narration
            )

            narration = (
                self._resample_if_needed(
                    narration=narration,
                    source_rate=(
                        narration_rate
                    ),
                    target_rate=(
                        original_rate
                    ),
                )
            )

            narration = (
                self._match_channels(
                    narration=narration,
                    target_channels=(
                        channel_count
                    ),
                )
            )

            start_sample = int(
                cue.playback_start
                * original_rate
            )

            end_sample = (
                start_sample
                + len(narration)
            )

            if (
                start_sample
                >= len(output_audio)
            ):
                continue

            if end_sample > len(
                output_audio
            ):
                narration = narration[
                    :(
                        len(output_audio)
                        - start_sample
                    )
                ]

                end_sample = len(
                    output_audio
                )

            original_region = (
                output_audio[
                    start_sample:
                    end_sample
                ]
            )

            original_region = (
                original_region
                * background_volume
            )

            narration = (
                narration
                * description_volume
            )

            mixed = (
                original_region
                + narration
            )

            output_audio[
                start_sample:
                end_sample
            ] = np.clip(
                mixed,
                -1.0,
                1.0,
            )

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        int_audio = (
            np.clip(
                output_audio,
                -1.0,
                1.0,
            )
            * 32767
        ).astype(
            np.int16
        )

        wavfile.write(
            str(output_path),
            original_rate,
            int_audio,
        )

        return str(
            output_path.resolve()
        )

    def get_duration(
        self,
        audio_path: str,
    ) -> float:
        sample_rate, audio = (
            wavfile.read(
                audio_path
            )
        )

        if sample_rate <= 0:
            return 0.0

        return (
            len(audio)
            / sample_rate
        )

    def _to_float(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:
        if np.issubdtype(
            audio.dtype,
            np.floating,
        ):
            return audio.astype(
                np.float32
            )

        info = np.iinfo(
            audio.dtype
        )

        scale = max(
            abs(info.min),
            info.max,
        )

        return (
            audio.astype(
                np.float32
            )
            / float(scale)
        )

    def _resample_if_needed(
        self,
        narration: np.ndarray,
        source_rate: int,
        target_rate: int,
    ):
        if source_rate == target_rate:
            return narration

        divisor = gcd(
            source_rate,
            target_rate,
        )

        up = (
            target_rate
            // divisor
        )

        down = (
            source_rate
            // divisor
        )

        return resample_poly(
            narration,
            up,
            down,
            axis=0,
        ).astype(
            np.float32
        )

    def _match_channels(
        self,
        narration: np.ndarray,
        target_channels: int,
    ):
        if target_channels == 1:
            if narration.ndim == 1:
                return narration

            return narration.mean(
                axis=1
            )

        if narration.ndim == 1:
            return np.repeat(
                narration[:, None],
                target_channels,
                axis=1,
            )

        current_channels = (
            narration.shape[1]
        )

        if (
            current_channels
            == target_channels
        ):
            return narration

        mono = narration.mean(
            axis=1
        )

        return np.repeat(
            mono[:, None],
            target_channels,
            axis=1,
        )