import os
import subprocess
from typing import List

import imageio_ffmpeg
from moviepy import VideoFileClip

from ai.audio_description.data_models import (
    AudioDescriptionCue,
)


class AudioMixer:
    """
    Responsável por:

    1. criar uma trilha única de audiodescrição;
    2. sincronizar as falas;
    3. misturar audiodescrição com o áudio original;
    4. gerar o vídeo final.

    O FFmpeg utilizado é obtido através de imageio-ffmpeg,
    evitando depender de uma instalação global do sistema.
    """

    def __init__(
        self,
        output_dir: str = "outputs/audio_descriptions",
        sample_rate: int = 44100,
    ):
        self.output_dir = output_dir
        self.sample_rate = sample_rate

        os.makedirs(
            self.output_dir,
            exist_ok=True,
        )

        self.ffmpeg_executable = (
            imageio_ffmpeg.get_ffmpeg_exe()
        )

    def create_description_track(
        self,
        cues: List[AudioDescriptionCue],
        output_audio_path: str,
    ) -> str:
        valid_cues = [
            cue
            for cue in cues
            if (
                cue.audio_path
                and os.path.exists(
                    cue.audio_path
                )
            )
        ]

        if not valid_cues:
            raise ValueError(
                "Nenhum cue com áudio válido "
                "foi informado."
            )

        output_audio_path = (
            self._ensure_wav_path(
                output_audio_path
            )
        )

        os.makedirs(
            os.path.dirname(
                output_audio_path
            ),
            exist_ok=True,
        )

        command = [
            self.ffmpeg_executable,
            "-y",
        ]

        for cue in valid_cues:
            command.extend(
                [
                    "-i",
                    cue.audio_path,
                ]
            )

        filter_parts = []
        delayed_labels = []

        for index, cue in enumerate(
            valid_cues
        ):
            delay_ms = int(
                float(
                    cue.start_time
                )
                * 1000
            )

            label = f"ad{index}"

            filter_parts.append(
                (
                    "[{}:a]"
                    "adelay={}|{}"
                    "[{}]"
                ).format(
                    index,
                    delay_ms,
                    delay_ms,
                    label,
                )
            )

            delayed_labels.append(
                f"[{label}]"
            )

        filter_parts.append(
            (
                "{}"
                "amix=inputs={}:"
                "duration=longest:"
                "normalize=0"
                "[adtrack]"
            ).format(
                "".join(
                    delayed_labels
                ),
                len(valid_cues),
            )
        )

        filter_complex = ";".join(
            filter_parts
        )

        command.extend(
            [
                "-filter_complex",
                filter_complex,
                "-map",
                "[adtrack]",
                "-ar",
                str(
                    self.sample_rate
                ),
                "-ac",
                "2",
                output_audio_path,
            ]
        )

        self._run_command(
            command
        )

        return output_audio_path

    def mix_description_track_into_video(
        self,
        video_path: str,
        description_track_path: str,
        output_video_path: str,
        keep_original_audio: bool = True,
        original_volume: float = 0.55,
        description_volume: float = 1.0,
    ) -> str:
        if not os.path.exists(
            video_path
        ):
            raise FileNotFoundError(
                "Vídeo original não encontrado: "
                f"{video_path}"
            )

        if not os.path.exists(
            description_track_path
        ):
            raise FileNotFoundError(
                "Trilha de audiodescrição "
                "não encontrada: "
                f"{description_track_path}"
            )

        os.makedirs(
            os.path.dirname(
                output_video_path
            ),
            exist_ok=True,
        )

        has_original_audio = (
            self.video_has_audio(
                video_path
            )
        )

        command = [
            self.ffmpeg_executable,
            "-y",
            "-i",
            video_path,
            "-i",
            description_track_path,
        ]

        if (
            keep_original_audio
            and has_original_audio
        ):
            filter_complex = (
                "[0:a]volume={}[orig];"
                "[1:a]volume={}[ad];"
                "[orig][ad]"
                "amix=inputs=2:"
                "duration=first:"
                "normalize=0"
                "[aout]"
            ).format(
                original_volume,
                description_volume,
            )

            command.extend(
                [
                    "-filter_complex",
                    filter_complex,
                    "-map",
                    "0:v",
                    "-map",
                    "[aout]",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-shortest",
                    output_video_path,
                ]
            )

        else:
            command.extend(
                [
                    "-map",
                    "0:v",
                    "-map",
                    "1:a",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-shortest",
                    output_video_path,
                ]
            )

        self._run_command(
            command
        )

        return output_video_path

    def video_has_audio(
        self,
        video_path: str,
    ) -> bool:
        """
        Verifica a presença de áudio usando MoviePy.

        Isso elimina a dependência direta de ffprobe.
        """

        clip = None

        try:
            clip = VideoFileClip(
                video_path
            )

            return (
                clip.audio
                is not None
            )

        finally:
            if clip is not None:
                clip.close()

    def _ensure_wav_path(
        self,
        output_path: str,
    ) -> str:
        root, extension = os.path.splitext(
            output_path
        )

        if extension.lower() != ".wav":
            return root + ".wav"

        return output_path

    def _run_command(
        self,
        command: List[str],
    ) -> None:
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        except subprocess.CalledProcessError as error:
            error_message = (
                error.stderr.decode(
                    "utf-8",
                    errors="ignore",
                )
            )

            raise RuntimeError(
                "Erro ao executar FFmpeg:\n"
                f"{error_message}"
            ) from error