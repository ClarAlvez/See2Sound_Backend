import os
import subprocess
from typing import List

from ai.audio_description.data_models import AudioDescriptionCue


class AudioMixer:
    """
    Usa FFmpeg para:
    1. criar uma trilha única de audiodescrição sincronizada;
    2. misturar essa trilha no vídeo original;
    3. exportar um vídeo final com audiodescrição.
    """

    def __init__(
        self,
        output_dir: str = "outputs/audio_descriptions",
        sample_rate: int = 44100,
    ):
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        os.makedirs(self.output_dir, exist_ok=True)

    def create_description_track(
        self,
        cues: List[AudioDescriptionCue],
        output_audio_path: str,
    ) -> str:
        """
        Cria uma trilha única de audiodescrição.

        Cada cue.audio_path é atrasado de acordo com cue.start_time.
        """

        valid_cues = [
            cue
            for cue in cues
            if cue.audio_path and os.path.exists(cue.audio_path)
        ]

        if not valid_cues:
            raise ValueError("Nenhum cue com áudio válido foi informado.")

        output_audio_path = self._ensure_wav_path(output_audio_path)
        os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)

        command = ["ffmpeg", "-y"]

        for cue in valid_cues:
            command.extend(["-i", cue.audio_path])

        filter_parts = []
        delayed_labels = []

        for index, cue in enumerate(valid_cues):
            delay_ms = int(float(cue.start_time) * 1000)
            label = "ad{}".format(index)

            # Para mono/estéreo, usa dois delays para garantir compatibilidade.
            filter_parts.append(
                "[{}:a]adelay={}|{}[{}]".format(
                    index,
                    delay_ms,
                    delay_ms,
                    label,
                )
            )
            delayed_labels.append("[{}]".format(label))

        filter_parts.append(
            "{}amix=inputs={}:duration=longest:normalize=0[adtrack]".format(
                "".join(delayed_labels),
                len(valid_cues),
            )
        )

        filter_complex = ";".join(filter_parts)

        command.extend(
            [
                "-filter_complex",
                filter_complex,
                "-map",
                "[adtrack]",
                "-ar",
                str(self.sample_rate),
                "-ac",
                "2",
                output_audio_path,
            ]
        )

        self._run_command(command)

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
        """
        Mistura a trilha de audiodescrição com o vídeo original.

        keep_original_audio=True:
            mantém o áudio original em volume reduzido e coloca a audiodescrição por cima.

        keep_original_audio=False:
            remove o áudio original e deixa apenas audiodescrição.
        """

        if not os.path.exists(video_path):
            raise FileNotFoundError("Vídeo original não encontrado: {}".format(video_path))

        if not os.path.exists(description_track_path):
            raise FileNotFoundError(
                "Trilha de audiodescrição não encontrada: {}".format(description_track_path)
            )

        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

        has_original_audio = self.video_has_audio(video_path)

        command = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-i",
            description_track_path,
        ]

        if keep_original_audio and has_original_audio:
            filter_complex = (
                "[0:a]volume={}[orig];"
                "[1:a]volume={}[ad];"
                "[orig][ad]amix=inputs=2:duration=first:normalize=0[aout]"
            ).format(original_volume, description_volume)

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
            # Caso o vídeo não tenha áudio, ou o usuário queira substituir o áudio original.
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

        self._run_command(command)

        return output_video_path

    def video_has_audio(self, video_path: str) -> bool:
        """
        Verifica se o vídeo possui trilha de áudio.
        """

        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            video_path,
        ]

        try:
            result = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return bool(result.stdout.decode("utf-8", errors="ignore").strip())

        except FileNotFoundError:
            raise RuntimeError(
                "FFprobe não foi encontrado. Instale FFmpeg para usar o módulo de áudio."
            )
        except subprocess.CalledProcessError:
            return False

    def _ensure_wav_path(self, output_path: str) -> str:
        root, ext = os.path.splitext(output_path)

        if ext.lower() != ".wav":
            return root + ".wav"

        return output_path

    def _run_command(self, command: List[str]) -> None:
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        except FileNotFoundError:
            raise RuntimeError(
                "FFmpeg não foi encontrado. Instale o FFmpeg e tente novamente."
            )

        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                "Erro ao executar FFmpeg:\n{}\n\nComando:\n{}".format(
                    error.stderr.decode("utf-8", errors="ignore"),
                    " ".join(command),
                )
            )
