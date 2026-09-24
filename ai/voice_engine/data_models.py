from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class VoiceCue:
    index: int
    text: str

    scene_start_time: Optional[float]
    scene_end_time: Optional[float]

    pause_start: Optional[float] = None
    pause_end: Optional[float] = None

    playback_start: Optional[float] = None
    playback_end: Optional[float] = None

    tts_duration: Optional[float] = None
    audio_path: Optional[str] = None

    inserted: bool = False
    skip_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "scene_start_time": (
                self.scene_start_time
            ),
            "scene_end_time": (
                self.scene_end_time
            ),
            "pause_start": (
                self.pause_start
            ),
            "pause_end": (
                self.pause_end
            ),
            "playback_start": (
                self.playback_start
            ),
            "playback_end": (
                self.playback_end
            ),
            "tts_duration": (
                self.tts_duration
            ),
            "audio_path": (
                self.audio_path
            ),
            "inserted": (
                self.inserted
            ),
            "skip_reason": (
                self.skip_reason
            ),
        }


@dataclass
class VoiceEngineResult:
    source_audio_path: str
    modified_audio_path: str
    manifest_path: str

    total_descriptions: int
    inserted_descriptions: int
    skipped_descriptions: int

    cues: List[VoiceCue]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_audio_path": (
                self.source_audio_path
            ),
            "modified_audio_path": (
                self.modified_audio_path
            ),
            "manifest_path": (
                self.manifest_path
            ),
            "total_descriptions": (
                self.total_descriptions
            ),
            "inserted_descriptions": (
                self.inserted_descriptions
            ),
            "skipped_descriptions": (
                self.skipped_descriptions
            ),
            "cues": [
                cue.to_dict()
                for cue in self.cues
            ],
        }