from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from faster_whisper import WhisperModel


@dataclass
class SpeechSegment:
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)


@dataclass
class SpeechPause:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)


class WhisperPauseDetector:
    """
    Responsável por:
    - transcrever o áudio com faster-whisper
    - detectar automaticamente o idioma, se não for informado
    - estruturar os segmentos de fala
    - calcular pausas entre falas
    - retornar um dicionário pronto para serialização em JSON
    """

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

    def _transcribe(
        self,
        audio_path: str | Path,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
    ):
        """
        Executa a transcrição no modelo e retorna:
        - lista de segmentos
        - info do áudio
        """
        segments, info = self.model.transcribe(
            str(audio_path),
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            task="transcribe",
        )

        return list(segments), info

    def transcribe_audio(
        self,
        audio_path: str | Path,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        refine_with_detected_language: bool = True,
    ) -> Dict[str, Any]:
        """
        Transcreve o áudio.

        Se `language` for None:
        1. detecta automaticamente o idioma;
        2. opcionalmente faz uma segunda passada com o idioma detectado fixado.

        Se `language` for informado:
        - transcreve diretamente com esse idioma.
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Áudio não encontrado: {audio_path}")

        # Primeira passada
        segments_list, info = self._transcribe(
            audio_path=audio_path,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )

        detected_language = info.language
        language_probability = round(info.language_probability, 4)

        if language is None and refine_with_detected_language and detected_language:
            segments_list, info = self._transcribe(
                audio_path=audio_path,
                language=detected_language,
                beam_size=beam_size,
                vad_filter=vad_filter,
            )
            detected_language = info.language
            language_probability = round(info.language_probability, 4)

        speech_segments: List[SpeechSegment] = []
        full_text_parts: List[str] = []

        for segment in segments_list:
            text = segment.text.strip()

            speech_segment = SpeechSegment(
                start=round(segment.start, 2),
                end=round(segment.end, 2),
                text=text,
            )

            speech_segments.append(speech_segment)

            if text:
                full_text_parts.append(text)

        return {
            "audio_path": str(audio_path),
            "language": detected_language,
            "language_probability": language_probability,
            "is_language_reliable": language_probability >= 0.8,
            "text": " ".join(full_text_parts).strip(),
            "segments": [
                {
                    **asdict(segment),
                    "duration": segment.duration,
                }
                for segment in speech_segments
            ],
        }

    def detect_pauses(
        self,
        segments: List[Dict[str, Any]],
        min_pause_duration: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Detecta pausas entre segmentos de fala.
        """
        if len(segments) < 2:
            return []

        pauses: List[SpeechPause] = []

        for i in range(len(segments) - 1):
            current_end = float(segments[i]["end"])
            next_start = float(segments[i + 1]["start"])

            gap = round(next_start - current_end, 2)

            if gap >= min_pause_duration:
                pauses.append(
                    SpeechPause(
                        start=current_end,
                        end=next_start,
                    )
                )

        return [
            {
                **asdict(pause),
                "duration": pause.duration,
            }
            for pause in pauses
        ]

    def analyze_audio(
        self,
        audio_path: str | Path,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        min_pause_duration: float = 0.5,
        refine_with_detected_language: bool = True,
    ) -> Dict[str, Any]:
        """
        Pipeline principal:
        - transcreve o áudio
        - detecta pausas
        - devolve tudo estruturado
        """
        transcription = self.transcribe_audio(
            audio_path=audio_path,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            refine_with_detected_language=refine_with_detected_language,
        )

        pauses = self.detect_pauses(
            segments=transcription["segments"],
            min_pause_duration=min_pause_duration,
        )

        return {
            "audio_path": transcription["audio_path"],
            "language": transcription["language"],
            "language_probability": transcription["language_probability"],
            "is_language_reliable": transcription["is_language_reliable"],
            "full_text": transcription["text"],
            "speech_segments": transcription["segments"],
            "speech_pauses": pauses,
            "stats": {
                "total_segments": len(transcription["segments"]),
                "total_pauses": len(pauses),
                "min_pause_duration": min_pause_duration,
                "model_size": self.model_size,
                "device": self.device,
                "compute_type": self.compute_type,
            },
        }