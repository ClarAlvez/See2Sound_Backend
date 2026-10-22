from __future__ import annotations

from dataclasses import (
    dataclass,
    asdict,
)
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import numpy as np
from scipy.io import wavfile

from faster_whisper import (
    WhisperModel,
)


@dataclass
class SpeechSegment:
    start: float
    end: float
    text: str

    @property
    def duration(
        self,
    ) -> float:
        return round(
            self.end - self.start,
            2,
        )


@dataclass
class SpeechPause:
    start: float
    end: float

    @property
    def duration(
        self,
    ) -> float:
        return round(
            self.end - self.start,
            2,
        )


class WhisperPauseDetector:
    """
    Responsável por:

    - transcrever áudio com faster-whisper;
    - detectar idioma quando necessário;
    - estruturar segmentos de fala;
    - detectar pausas;
    - reconhecer WAV completamente silencioso;
    - evitar transcrição desnecessária de silêncio.

    O modelo Whisper é carregado de forma
    preguiçosa: ele só é inicializado se
    realmente houver áudio para analisar.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:

        self.model_size = (
            model_size
        )

        self.device = (
            device
        )

        self.compute_type = (
            compute_type
        )

        self._model: Optional[
            WhisperModel
        ] = None


    # ========================================================
    # Modelo
    # ========================================================

    def _get_model(
        self,
    ) -> WhisperModel:
        """
        Carrega o Whisper somente quando
        realmente for necessário.
        """
        if self._model is None:
            print(
                "[Whisper] Carregando "
                f"modelo '{self.model_size}'..."
            )

            self._model = (
                WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=(
                        self.compute_type
                    ),
                )
            )

        return self._model


    # ========================================================
    # Validação
    # ========================================================

    def _validate_audio_path(
        self,
        audio_path: str | Path,
    ) -> Path:

        path = Path(
            audio_path
        )

        if not path.exists():
            raise FileNotFoundError(
                "Áudio não encontrado: "
                f"{path}"
            )

        if not path.is_file():
            raise ValueError(
                "O caminho do áudio "
                "não é um arquivo: "
                f"{path}"
            )

        return path


    # ========================================================
    # Detecção de silêncio
    # ========================================================

    def _is_completely_silent_wav(
        self,
        audio_path: Path,
    ) -> bool:
        """
        Detecta WAV sem nenhuma amostra
        diferente de zero.

        Isso identifica diretamente o WAV
        artificial criado pelo extractor
        para vídeos sem faixa de áudio.

        Caso o arquivo não possa ser lido
        pelo scipy, simplesmente retorna
        False e deixa o Whisper processar
        normalmente.
        """
        try:
            sample_rate, audio = (
                wavfile.read(
                    str(audio_path)
                )
            )

        except Exception:
            return False

        if sample_rate <= 0:
            return False

        if audio.size == 0:
            return True

        return not bool(
            np.any(audio)
        )


    # ========================================================
    # Whisper
    # ========================================================

    def _transcribe(
        self,
        audio_path: str | Path,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
    ):
        """
        Executa uma transcrição.
        """
        model = (
            self._get_model()
        )

        segments, info = (
            model.transcribe(
                str(audio_path),
                language=language,
                beam_size=beam_size,
                vad_filter=vad_filter,
                task="transcribe",
            )
        )

        return (
            list(segments),
            info,
        )


    def transcribe_audio(
        self,
        audio_path: str | Path,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        refine_with_detected_language: bool = True,
    ) -> Dict[str, Any]:
        """
        Transcreve áudio que contém som.

        Se language for None:
        - Whisper detecta o idioma;
        - opcionalmente executa novamente
          usando o idioma detectado.

        Se language estiver definido:
        - usa diretamente esse idioma.
        """
        audio_path = (
            self._validate_audio_path(
                audio_path
            )
        )

        segments_list, info = (
            self._transcribe(
                audio_path=(
                    audio_path
                ),
                language=language,
                beam_size=beam_size,
                vad_filter=(
                    vad_filter
                ),
            )
        )

        detected_language = (
            info.language
        )

        language_probability = round(
            info.language_probability,
            4,
        )

        if (
            language is None
            and refine_with_detected_language
            and detected_language
        ):
            segments_list, info = (
                self._transcribe(
                    audio_path=(
                        audio_path
                    ),
                    language=(
                        detected_language
                    ),
                    beam_size=(
                        beam_size
                    ),
                    vad_filter=(
                        vad_filter
                    ),
                )
            )

            detected_language = (
                info.language
            )

            language_probability = (
                round(
                    info.language_probability,
                    4,
                )
            )

        speech_segments: List[
            SpeechSegment
        ] = []

        full_text_parts: List[str] = []

        for segment in segments_list:
            text = (
                segment.text.strip()
            )

            speech_segment = (
                SpeechSegment(
                    start=round(
                        segment.start,
                        2,
                    ),
                    end=round(
                        segment.end,
                        2,
                    ),
                    text=text,
                )
            )

            speech_segments.append(
                speech_segment
            )

            if text:
                full_text_parts.append(
                    text
                )

        return {
            "audio_path": str(
                audio_path
            ),

            "language": (
                detected_language
            ),

            "language_probability": (
                language_probability
            ),

            "is_language_reliable": (
                language_probability
                >= 0.8
            ),

            "text": " ".join(
                full_text_parts
            ).strip(),

            "segments": [
                {
                    **asdict(
                        segment
                    ),
                    "duration": (
                        segment.duration
                    ),
                }
                for segment
                in speech_segments
            ],
        }


    # ========================================================
    # Pausas
    # ========================================================

    def detect_pauses(
        self,
        segments: List[
            Dict[str, Any]
        ],
        min_pause_duration: float = 0.5,
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Detecta pausas entre segmentos
        de fala.

        A Voice Engine calcula também os
        espaços antes da primeira fala e
        depois da última fala.
        """
        if len(segments) < 2:
            return []

        pauses: List[
            SpeechPause
        ] = []

        for index in range(
            len(segments) - 1
        ):
            current_end = float(
                segments[
                    index
                ]["end"]
            )

            next_start = float(
                segments[
                    index + 1
                ]["start"]
            )

            gap = round(
                next_start
                - current_end,
                2,
            )

            if (
                gap
                >= min_pause_duration
            ):
                pauses.append(
                    SpeechPause(
                        start=(
                            current_end
                        ),
                        end=(
                            next_start
                        ),
                    )
                )

        return [
            {
                **asdict(
                    pause
                ),
                "duration": (
                    pause.duration
                ),
            }
            for pause in pauses
        ]


    # ========================================================
    # Resultado para silêncio
    # ========================================================

    def _build_silent_result(
        self,
        audio_path: Path,
        language: Optional[str],
        min_pause_duration: float,
    ) -> Dict[str, Any]:
        """
        Resultado determinístico para um
        WAV completamente silencioso.
        """
        language_probability = (
            1.0
            if language
            else 0.0
        )

        return {
            "audio_path": str(
                audio_path
            ),

            "language": (
                language
            ),

            "language_probability": (
                language_probability
            ),

            "is_language_reliable": (
                bool(language)
            ),

            "full_text": "",

            "speech_segments": [],

            "speech_pauses": [],

            "stats": {
                "total_segments": 0,

                "total_pauses": 0,

                "min_pause_duration": (
                    min_pause_duration
                ),

                "model_size": (
                    self.model_size
                ),

                "device": (
                    self.device
                ),

                "compute_type": (
                    self.compute_type
                ),

                "silence_detected": True,

                "whisper_executed": False,
            },
        }


    # ========================================================
    # Pipeline principal
    # ========================================================

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
        Pipeline principal da análise.

        Se o WAV for completamente silencioso:
            retorna imediatamente sem
            carregar/executar Whisper.

        Caso contrário:
            executa Whisper normalmente.
        """
        validated_audio_path = (
            self._validate_audio_path(
                audio_path
            )
        )

        if (
            self
            ._is_completely_silent_wav(
                validated_audio_path
            )
        ):
            print(
                "[Whisper] Áudio "
                "completamente silencioso."
            )

            print(
                "[Whisper] Nenhuma fala "
                "para transcrever."
            )

            return (
                self._build_silent_result(
                    audio_path=(
                        validated_audio_path
                    ),
                    language=language,
                    min_pause_duration=(
                        min_pause_duration
                    ),
                )
            )

        transcription = (
            self.transcribe_audio(
                audio_path=(
                    validated_audio_path
                ),
                language=language,
                beam_size=beam_size,
                vad_filter=(
                    vad_filter
                ),
                refine_with_detected_language=(
                    refine_with_detected_language
                ),
            )
        )

        pauses = (
            self.detect_pauses(
                segments=(
                    transcription[
                        "segments"
                    ]
                ),
                min_pause_duration=(
                    min_pause_duration
                ),
            )
        )

        return {
            "audio_path": (
                transcription[
                    "audio_path"
                ]
            ),

            "language": (
                transcription[
                    "language"
                ]
            ),

            "language_probability": (
                transcription[
                    "language_probability"
                ]
            ),

            "is_language_reliable": (
                transcription[
                    "is_language_reliable"
                ]
            ),

            "full_text": (
                transcription[
                    "text"
                ]
            ),

            "speech_segments": (
                transcription[
                    "segments"
                ]
            ),

            "speech_pauses": (
                pauses
            ),

            "stats": {
                "total_segments": len(
                    transcription[
                        "segments"
                    ]
                ),

                "total_pauses": len(
                    pauses
                ),

                "min_pause_duration": (
                    min_pause_duration
                ),

                "model_size": (
                    self.model_size
                ),

                "device": (
                    self.device
                ),

                "compute_type": (
                    self.compute_type
                ),

                "silence_detected": False,

                "whisper_executed": True,
            },
        }