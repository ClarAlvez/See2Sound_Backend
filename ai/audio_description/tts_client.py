"""TTS do See2Sound: Edge Neural (online) ou pyttsx3 (offline).

Interface compatível com VoiceEngine:
    TTSClient(rate=170, volume=1.0).save_to_file(text, output_path)
"""
import asyncio
import os
import platform
import subprocess
import tempfile
import threading
import wave
from pathlib import Path
from typing import List, Optional

import imageio_ffmpeg


class TTSClient:
    def __init__(
        self,
        rate: int = 170,
        volume: float = 1.0,
        voice_name_contains: Optional[str] = None,
    ):
        self.rate = int(rate)
        self.volume = max(0.0, min(float(volume), 1.0))
        self.voice_name_contains = voice_name_contains
        self.system = platform.system()
        self.provider = os.getenv(
            "SEE2SOUND_TTS_PROVIDER", "edge"
        ).strip().lower()

        if self.provider not in {"edge", "pyttsx3"}:
            raise ValueError(
                "SEE2SOUND_TTS_PROVIDER deve ser 'edge' ou 'pyttsx3'."
            )

        self.voice = os.getenv(
            "SEE2SOUND_TTS_VOICE", "pt-BR-FranciscaNeural"
        ).strip()

        # O TTS neural usa uma porcentagem relativa à velocidade
        # natural da voz, diferente do rate=170 do pyttsx3.
        self.edge_rate = os.getenv(
            "SEE2SOUND_TTS_RATE", "-5%"
        ).strip()
        self.edge_pitch = os.getenv(
            "SEE2SOUND_TTS_PITCH", "+0Hz"
        ).strip()

    @staticmethod
    def _run_async(coroutine_factory):
        """Executa coroutine também se houver event loop na thread atual."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine_factory())

        result = {}
        def worker():
            try:
                result["value"] = asyncio.run(coroutine_factory())
            except BaseException as exc:
                result["error"] = exc

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join()
        if "error" in result:
            raise result["error"]
        return result.get("value")

    async def _edge_save(self, text: str, path: str) -> None:
        try:
            import edge_tts
        except ImportError as exc:
            raise RuntimeError(
                "edge-tts ausente. Execute: python -m pip install edge-tts"
            ) from exc

        communicator = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.edge_rate,
            pitch=self.edge_pitch,
        )
        await communicator.save(path)

    def _create_engine(self):
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate)
        # Volume aplicado uma única vez na conversão final.
        engine.setProperty("volume", 1.0)

        wanted = self.voice_name_contains
        if wanted:
            for voice in engine.getProperty("voices"):
                name = str(getattr(voice, "name", "")).lower()
                identifier = str(getattr(voice, "id", "")).lower()
                if wanted.lower() in name or wanted.lower() in identifier:
                    engine.setProperty("voice", voice.id)
                    break
        return engine

    def _pyttsx3_save(self, text: str, path: str) -> None:
        engine = self._create_engine()
        try:
            engine.save_to_file(text, path)
            engine.runAndWait()
        finally:
            engine.stop()

    def _convert_to_wav(self, source: str, destination: str) -> None:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", source, "-vn",
            "-af", f"volume={self.volume}",
            "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "1",
            destination,
        ]
        process = subprocess.run(
            command, capture_output=True, text=True, check=False
        )
        if process.returncode != 0:
            raise RuntimeError(
                "Falha ao converter TTS para WAV: " + process.stderr.strip()
            )

    @staticmethod
    def _check_wav(path: str) -> None:
        if not Path(path).is_file() or Path(path).stat().st_size <= 44:
            raise RuntimeError("O TTS não produziu um WAV válido.")
        with wave.open(path, "rb") as wav_file:
            if wav_file.getnframes() <= 0:
                raise RuntimeError("Áudio TTS vazio.")
            if wav_file.getframerate() != 44100:
                raise RuntimeError("Sample rate do WAV inesperado.")
            if wav_file.getnchannels() != 1:
                raise RuntimeError("WAV não é mono.")
            if wav_file.getsampwidth() != 2:
                raise RuntimeError("WAV não é PCM de 16 bits.")

    def save_to_file(self, text: str, output_path: str) -> str:
        text = (text or "").strip()
        if not text:
            raise ValueError("Texto vazio não pode ser convertido em voz.")

        destination = str(Path(output_path).with_suffix(".wav"))
        Path(destination).parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="see2sound_tts_") as temporary:
            if self.provider == "edge":
                source = str(Path(temporary) / "speech.mp3")
                # Falhas de rede são explícitas: não produzimos uma voz
                # diferente silenciosamente no meio de uma geração.
                self._run_async(lambda: self._edge_save(text, source))
            else:
                source = str(Path(temporary) / "speech.wav")
                self._pyttsx3_save(text, source)

            if not Path(source).is_file() or Path(source).stat().st_size == 0:
                raise RuntimeError(
                    f"O provedor {self.provider} não gerou áudio."
                )

            self._convert_to_wav(source, destination)

        self._check_wav(destination)
        print(
            f"[TTS] provider={self.provider} "
            f"voice={self.voice if self.provider == 'edge' else 'sistema'} "
            f"path={destination}"
        )
        return destination

    def speak(self, text: str) -> None:
        """Reproduz texto usando um arquivo temporário."""
        text = (text or "").strip()
        if not text:
            return
        if self.provider == "pyttsx3":
            engine = self._create_engine()
            try:
                engine.say(text)
                engine.runAndWait()
            finally:
                engine.stop()
            return

        with tempfile.TemporaryDirectory(prefix="see2sound_speak_") as temp:
            path = self.save_to_file(text, str(Path(temp) / "speech.wav"))
            if self.system != "Windows":
                raise NotImplementedError(
                    "speak() neural direto está disponível apenas no Windows; "
                    "use save_to_file() em outras plataformas."
                )
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME)

    def list_voices(self) -> List[dict]:
        if self.provider == "edge":
            async def fetch():
                import edge_tts
                voices = await edge_tts.list_voices()
                return [
                    {
                        "id": v.get("ShortName", ""),
                        "name": v.get("FriendlyName", v.get("ShortName", "")),
                        "languages": [v.get("Locale", "")],
                    }
                    for v in voices
                ]
            return self._run_async(fetch)

        engine = self._create_engine()
        try:
            return [
                {
                    "id": getattr(voice, "id", ""),
                    "name": getattr(voice, "name", ""),
                    "languages": getattr(voice, "languages", []),
                }
                for voice in engine.getProperty("voices")
            ]
        finally:
            engine.stop()
