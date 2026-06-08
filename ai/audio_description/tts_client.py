import os
import subprocess
import tempfile
from typing import List, Optional

import pyttsx3


class TTSClient:
    """
    Cliente de Text-To-Speech local usando pyttsx3.

    Para o protótipo/produto local:
    - recebe texto;
    - gera arquivo de áudio;
    - normaliza para WAV usando FFmpeg quando necessário.

    Observação:
    No macOS, o mecanismo nativo pode gerar formatos diferentes dependendo da voz.
    Por isso, este cliente tenta converter a saída para WAV.
    """

    def __init__(
        self,
        rate: int = 170,
        volume: float = 1.0,
        voice_name_contains: Optional[str] = None,
    ):
        self.rate = rate
        self.volume = volume
        self.voice_name_contains = voice_name_contains

        self.engine = pyttsx3.init()
        self._configure_engine()

    def save_to_file(
        self,
        text: str,
        output_path: str,
    ) -> str:
        """
        Salva o texto como áudio.

        Retorna o caminho final do arquivo WAV.
        """

        clean_text = self._clean_text(text)

        if not clean_text:
            raise ValueError("Texto vazio não pode ser convertido em áudio.")

        output_path = self._ensure_wav_path(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Gera em arquivo temporário primeiro.
        # Isso facilita converter para WAV caso o sistema gere outro formato.
        temp_dir = tempfile.mkdtemp(prefix="spectra_tts_")
        temp_output = os.path.join(temp_dir, "tts_output.wav")

        self.engine.save_to_file(clean_text, temp_output)
        self.engine.runAndWait()

        if not os.path.exists(temp_output):
            raise RuntimeError(
                "O TTS não gerou o arquivo esperado: {}".format(temp_output)
            )

        self._convert_to_wav(
            input_path=temp_output,
            output_path=output_path,
        )

        return output_path

    def speak(self, text: str) -> None:
        """
        Reproduz a fala imediatamente.
        Útil apenas para teste rápido.
        """

        clean_text = self._clean_text(text)

        if not clean_text:
            return

        self.engine.say(clean_text)
        self.engine.runAndWait()

    def list_voices(self) -> List[dict]:
        voices = self.engine.getProperty("voices")
        result = []

        for voice in voices:
            result.append(
                {
                    "id": getattr(voice, "id", ""),
                    "name": getattr(voice, "name", ""),
                    "languages": getattr(voice, "languages", []),
                }
            )

        return result

    def _configure_engine(self) -> None:
        self.engine.setProperty("rate", self.rate)
        self.engine.setProperty("volume", self.volume)

        if self.voice_name_contains:
            self._select_voice_by_name(self.voice_name_contains)

    def _select_voice_by_name(self, name_part: str) -> None:
        voices = self.engine.getProperty("voices")
        name_part = name_part.lower()

        for voice in voices:
            voice_name = getattr(voice, "name", "").lower()
            voice_id = getattr(voice, "id", "").lower()

            if name_part in voice_name or name_part in voice_id:
                self.engine.setProperty("voice", voice.id)
                return

    def _convert_to_wav(
        self,
        input_path: str,
        output_path: str,
    ) -> None:
        """
        Converte o áudio para WAV mono 44.1kHz.

        Isso padroniza o arquivo para o FFmpeg mixar depois.
        """

        command = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-ar",
            "44100",
            "-ac",
            "1",
            output_path,
        ]

        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "FFmpeg não foi encontrado. Instale o FFmpeg para converter e mixar áudios."
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                "Erro ao converter áudio TTS para WAV: {}".format(
                    error.stderr.decode("utf-8", errors="ignore")
                )
            )

    def _ensure_wav_path(self, output_path: str) -> str:
        root, ext = os.path.splitext(output_path)

        if ext.lower() != ".wav":
            return root + ".wav"

        return output_path

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""

        return text.strip()
