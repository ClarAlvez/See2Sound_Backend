import os
import platform
import shutil
import subprocess
import tempfile
import wave
from typing import List, Optional

import imageio_ffmpeg
import pyttsx3


class TTSClient:
    """
    Cliente local de Text-To-Speech.

    macOS:
        usa o comando nativo `say`, que é mais estável
        para múltiplas gerações consecutivas.

    Outros sistemas:
        usa pyttsx3.

    Em ambos os casos:
        normaliza o resultado para WAV PCM 16-bit,
        44100 Hz, mono.
    """

    def __init__(
        self,
        rate: int = 170,
        volume: float = 1.0,
        voice_name_contains: Optional[str] = None,
    ):
        self.rate = rate

        self.volume = max(
            0.0,
            min(
                float(volume),
                1.0,
            ),
        )

        self.voice_name_contains = (
            voice_name_contains
        )

        self.system = platform.system()

    def save_to_file(
        self,
        text: str,
        output_path: str,
    ) -> str:
        clean_text = self._clean_text(
            text
        )

        if not clean_text:
            raise ValueError(
                "Texto vazio não pode ser "
                "convertido em áudio."
            )

        output_path = (
            self._ensure_wav_path(
                output_path
            )
        )

        output_directory = (
            os.path.dirname(
                output_path
            )
        )

        if output_directory:
            os.makedirs(
                output_directory,
                exist_ok=True,
            )

        with tempfile.TemporaryDirectory(
            prefix="see2sound_tts_"
        ) as temp_dir:

            if self.system == "Darwin":
                temp_output = os.path.join(
                    temp_dir,
                    "tts_output.aiff",
                )

                self._generate_macos(
                    text=clean_text,
                    output_path=temp_output,
                )

            else:
                temp_output = os.path.join(
                    temp_dir,
                    "tts_output.wav",
                )

                self._generate_pyttsx3(
                    text=clean_text,
                    output_path=temp_output,
                )

            self._validate_source_audio(
                temp_output
            )

            self._convert_to_wav(
                input_path=temp_output,
                output_path=output_path,
            )

        self._validate_final_wav(
            output_path
        )

        return output_path

    def speak(
        self,
        text: str,
    ) -> None:
        clean_text = self._clean_text(
            text
        )

        if not clean_text:
            return

        if self.system == "Darwin":
            command = [
                "/usr/bin/say",
                "-r",
                str(self.rate),
            ]

            if self.voice_name_contains:
                command.extend(
                    [
                        "-v",
                        self.voice_name_contains,
                    ]
                )

            command.append(
                clean_text
            )

            subprocess.run(
                command,
                check=True,
            )

            return

        engine = self._create_engine()

        try:
            engine.say(
                clean_text
            )

            engine.runAndWait()

        finally:
            engine.stop()

    def list_voices(
        self,
    ) -> List[dict]:

        if self.system == "Darwin":
            return (
                self._list_macos_voices()
            )

        engine = self._create_engine()

        try:
            voices = engine.getProperty(
                "voices"
            )

            result = []

            for voice in voices:
                result.append(
                    {
                        "id": getattr(
                            voice,
                            "id",
                            "",
                        ),
                        "name": getattr(
                            voice,
                            "name",
                            "",
                        ),
                        "languages": getattr(
                            voice,
                            "languages",
                            [],
                        ),
                    }
                )

            return result

        finally:
            engine.stop()

    def _generate_macos(
        self,
        text: str,
        output_path: str,
    ) -> None:

        say_path = "/usr/bin/say"

        if not os.path.exists(
            say_path
        ):
            raise RuntimeError(
                "O comando nativo 'say' "
                "não foi encontrado no macOS."
            )

        command = [
            say_path,
            "-r",
            str(self.rate),
        ]

        if self.voice_name_contains:
            command.extend(
                [
                    "-v",
                    self.voice_name_contains,
                ]
            )

        command.extend(
            [
                "-o",
                output_path,
                text,
            ]
        )

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "Erro ao gerar TTS com "
                "o comando 'say': "
                + result.stderr.decode(
                    "utf-8",
                    errors="ignore",
                )
            )

    def _generate_pyttsx3(
        self,
        text: str,
        output_path: str,
    ) -> None:

        engine = self._create_engine()

        try:
            engine.save_to_file(
                text,
                output_path,
            )

            engine.runAndWait()

        finally:
            engine.stop()

    def _create_engine(
        self,
    ):
        engine = pyttsx3.init()

        engine.setProperty(
            "rate",
            self.rate,
        )

        engine.setProperty(
            "volume",
            self.volume,
        )

        if self.voice_name_contains:
            voices = engine.getProperty(
                "voices"
            )

            name_part = (
                self.voice_name_contains
                .lower()
            )

            for voice in voices:
                voice_name = getattr(
                    voice,
                    "name",
                    "",
                ).lower()

                voice_id = getattr(
                    voice,
                    "id",
                    "",
                ).lower()

                if (
                    name_part in voice_name
                    or name_part in voice_id
                ):
                    engine.setProperty(
                        "voice",
                        voice.id,
                    )

                    break

        return engine

    def _convert_to_wav(
        self,
        input_path: str,
        output_path: str,
    ) -> None:

        ffmpeg_executable = (
            imageio_ffmpeg
            .get_ffmpeg_exe()
        )

        command = [
            ffmpeg_executable,
            "-y",

            "-i",
            input_path,

            "-vn",

            "-c:a",
            "pcm_s16le",

            "-ar",
            "44100",

            "-ac",
            "1",

            "-filter:a",
            f"volume={self.volume}",

            output_path,
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "Erro ao converter áudio "
                "TTS para WAV: "
                + result.stderr.decode(
                    "utf-8",
                    errors="ignore",
                )
            )

    def _validate_source_audio(
        self,
        path: str,
    ) -> None:

        if not os.path.exists(
            path
        ):
            raise RuntimeError(
                "O TTS não gerou o "
                "arquivo esperado: "
                f"{path}"
            )

        size = os.path.getsize(
            path
        )

        if size <= 0:
            raise RuntimeError(
                "O TTS gerou um arquivo "
                "de áudio vazio."
            )

    def _validate_final_wav(
        self,
        path: str,
    ) -> None:

        if not os.path.exists(
            path
        ):
            raise RuntimeError(
                "O WAV final não foi "
                "criado."
            )

        if os.path.getsize(
            path
        ) <= 44:
            raise RuntimeError(
                "O WAV final está vazio "
                "ou contém apenas cabeçalho."
            )

        try:
            with wave.open(
                path,
                "rb",
            ) as wav_file:

                frame_count = (
                    wav_file.getnframes()
                )

                sample_rate = (
                    wav_file.getframerate()
                )

                if (
                    frame_count <= 0
                    or sample_rate <= 0
                ):
                    raise RuntimeError(
                        "O WAV final não "
                        "contém áudio válido."
                    )

        except wave.Error as error:
            raise RuntimeError(
                "O arquivo produzido não "
                "é um WAV PCM válido: "
                f"{error}"
            )

    def _list_macos_voices(
        self,
    ) -> List[dict]:

        result = subprocess.run(
            [
                "/usr/bin/say",
                "-v",
                "?",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            return []

        voices = []

        for line in (
            result.stdout.splitlines()
        ):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if not parts:
                continue

            voices.append(
                {
                    "id": parts[0],
                    "name": parts[0],
                    "languages": (
                        parts[1:2]
                        if len(parts) > 1
                        else []
                    ),
                }
            )

        return voices

    def _ensure_wav_path(
        self,
        output_path: str,
    ) -> str:

        root, extension = (
            os.path.splitext(
                output_path
            )
        )

        if (
            extension.lower()
            != ".wav"
        ):
            return (
                root
                + ".wav"
            )

        return output_path

    def _clean_text(
        self,
        text: str,
    ) -> str:

        if not text:
            return ""

        return text.strip()