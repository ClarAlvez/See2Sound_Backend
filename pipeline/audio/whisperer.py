from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

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
    - transcrever o áudio
    - identificar segmentos de fala
    - calcular pausas entre segmentos
    - retornar estrutura pronta para JSON
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )

    def transcribe_audio(
        self,
        audio_path: str | Path,
        language: Optional[str] = "pt",
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> Dict[str, Any]:
        """
        Transcreve o áudio e devolve os segmentos com timestamp.
        """
        audio_path = str(audio_path)

        segments_generator, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )

        # No faster-whisper, os segmentos são um generator.
        # Converter para lista já força a execução completa da transcrição.
        segments_list = list(segments_generator)

        speech_segments: List[SpeechSegment] = []
        full_text_parts: List[str] = []

        for segment in segments_list:
            text = segment.text.strip()
            speech_segments.append(
                SpeechSegment(
                    start=round(segment.start, 2),
                    end=round(segment.end, 2),
                    text=text
                )
            )
            if text:
                full_text_parts.append(text)

        return {
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "text": " ".join(full_text_parts).strip(),
            "segments": [asdict(seg) | {"duration": seg.duration} for seg in speech_segments],
        }

    def detect_pauses(
        self,
        segments: List[Dict[str, Any]],
        min_pause_duration: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Calcula pausas ENTRE segmentos de fala.
        Ex.: se uma fala termina em 3.2 e a próxima começa em 5.0,
        há uma pausa de 1.8s.
        """
        pauses: List[SpeechPause] = []

        if len(segments) < 2:
            return []

        for i in range(len(segments) - 1):
            current_end = float(segments[i]["end"])
            next_start = float(segments[i + 1]["start"])

            gap = round(next_start - current_end, 2)

            if gap >= min_pause_duration:
                pauses.append(
                    SpeechPause(
                        start=current_end,
                        end=next_start
                    )
                )

        return [asdict(pause) | {"duration": pause.duration} for pause in pauses]

    def analyze_audio(
        self,
        audio_path: str | Path,
        language: Optional[str] = "pt",
        beam_size: int = 5,
        vad_filter: bool = True,
        min_pause_duration: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Pipeline completo de análise do áudio.
        """
        transcription = self.transcribe_audio(
            audio_path=audio_path,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )

        pauses = self.detect_pauses(
            segments=transcription["segments"],
            min_pause_duration=min_pause_duration,
        )

        return {
            "audio_path": str(audio_path),
            "language": transcription["language"],
            "language_probability": transcription["language_probability"],
            "full_text": transcription["text"],
            "speech_segments": transcription["segments"],
            "speech_pauses": pauses,
            "stats": {
                "total_segments": len(transcription["segments"]),
                "total_pauses": len(pauses),
                "min_pause_duration": min_pause_duration,
            }
        }