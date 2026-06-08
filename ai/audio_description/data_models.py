from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class AudioDescriptionCue:
    """
    Representa uma fala de audiodescrição em um intervalo do vídeo.

    Essa estrutura é a ponte entre:
    módulo narrativo -> módulo de áudio.
    """

    start_time: float
    end_time: Optional[float]
    text: str
    audio_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text": self.text,
            "audio_path": self.audio_path,
        }


@dataclass
class AudioDescriptionResult:
    """
    Resultado final da geração de audiodescrição em arquivo.
    """

    video_path: str
    output_video_path: str
    cues: List[AudioDescriptionCue]
    description_track_path: Optional[str] = None
    manifest_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_path": self.video_path,
            "output_video_path": self.output_video_path,
            "description_track_path": self.description_track_path,
            "manifest_path": self.manifest_path,
            "cues": [
                cue.to_dict()
                for cue in self.cues
            ],
        }
