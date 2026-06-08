import json
import os
from typing import Any, Dict, List, Optional

from ai.audio_description.audio_mixer import AudioMixer
from ai.audio_description.data_models import AudioDescriptionCue, AudioDescriptionResult
from ai.audio_description.timeline_builder import TimelineBuilder
from ai.audio_description.tts_client import TTSClient


class AudioDescriptionService:
    """
    Serviço principal do módulo de áudio.

    Fluxo:
    timeline textual
    ↓
    gera WAV para cada descrição
    ↓
    cria trilha única sincronizada
    ↓
    mistura a trilha no vídeo original
    ↓
    retorna vídeo final com audiodescrição
    """

    def __init__(
        self,
        output_dir: str = "outputs/audio_descriptions",
        tts_rate: int = 170,
        tts_volume: float = 1.0,
        voice_name_contains: Optional[str] = None,
    ):
        self.output_dir = output_dir
        self.cue_audio_dir = os.path.join(self.output_dir, "cues")
        self.tracks_dir = os.path.join(self.output_dir, "tracks")
        self.videos_dir = os.path.join(self.output_dir, "videos")
        self.manifest_dir = os.path.join(self.output_dir, "manifests")

        os.makedirs(self.cue_audio_dir, exist_ok=True)
        os.makedirs(self.tracks_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)
        os.makedirs(self.manifest_dir, exist_ok=True)

        self.tts_client = TTSClient(
            rate=tts_rate,
            volume=tts_volume,
            voice_name_contains=voice_name_contains,
        )

        self.timeline_builder = TimelineBuilder()
        self.audio_mixer = AudioMixer(output_dir=output_dir)

    def create_audio_described_video_from_timeline(
        self,
        video_path: str,
        timeline: List[Dict[str, Any]],
        output_video_path: Optional[str] = None,
        keep_original_audio: bool = True,
        original_volume: float = 0.55,
        description_volume: float = 1.0,
    ) -> AudioDescriptionResult:
        """
        Recebe timeline textual em dict e exporta um vídeo final com audiodescrição.
        """

        cues = self.timeline_builder.build_from_narrative_dicts(timeline)

        return self.create_audio_described_video_from_cues(
            video_path=video_path,
            cues=cues,
            output_video_path=output_video_path,
            keep_original_audio=keep_original_audio,
            original_volume=original_volume,
            description_volume=description_volume,
        )

    def create_audio_described_video_from_cues(
        self,
        video_path: str,
        cues: List[AudioDescriptionCue],
        output_video_path: Optional[str] = None,
        keep_original_audio: bool = True,
        original_volume: float = 0.55,
        description_volume: float = 1.0,
    ) -> AudioDescriptionResult:
        if not os.path.exists(video_path):
            raise FileNotFoundError("Vídeo original não encontrado: {}".format(video_path))

        if not cues:
            raise ValueError("Nenhuma descrição foi informada para gerar áudio.")

        if output_video_path is None:
            output_video_path = os.path.join(
                self.videos_dir,
                self._build_output_video_name(video_path),
            )

        cues_with_audio = self.generate_audio_for_cues(cues)

        description_track_path = os.path.join(
            self.tracks_dir,
            "audio_description_track.wav",
        )

        description_track_path = self.audio_mixer.create_description_track(
            cues=cues_with_audio,
            output_audio_path=description_track_path,
        )

        final_video_path = self.audio_mixer.mix_description_track_into_video(
            video_path=video_path,
            description_track_path=description_track_path,
            output_video_path=output_video_path,
            keep_original_audio=keep_original_audio,
            original_volume=original_volume,
            description_volume=description_volume,
        )

        manifest_path = self._save_manifest(
            video_path=video_path,
            output_video_path=final_video_path,
            description_track_path=description_track_path,
            cues=cues_with_audio,
        )

        return AudioDescriptionResult(
            video_path=video_path,
            output_video_path=final_video_path,
            cues=cues_with_audio,
            description_track_path=description_track_path,
            manifest_path=manifest_path,
        )

    def generate_audio_for_cues(
        self,
        cues: List[AudioDescriptionCue],
    ) -> List[AudioDescriptionCue]:
        processed_cues = []

        for index, cue in enumerate(cues):
            if not cue.text:
                continue

            output_path = os.path.join(
                self.cue_audio_dir,
                "ad_{:03d}_{}.wav".format(
                    index,
                    self._safe_time(cue.start_time),
                )
            )

            audio_path = self.tts_client.save_to_file(
                text=cue.text,
                output_path=output_path,
            )

            processed_cues.append(
                AudioDescriptionCue(
                    start_time=cue.start_time,
                    end_time=cue.end_time,
                    text=cue.text,
                    audio_path=audio_path,
                )
            )

        return processed_cues

    def _save_manifest(
        self,
        video_path: str,
        output_video_path: str,
        description_track_path: str,
        cues: List[AudioDescriptionCue],
    ) -> str:
        manifest = {
            "video_path": video_path,
            "output_video_path": output_video_path,
            "description_track_path": description_track_path,
            "cues": [
                cue.to_dict()
                for cue in cues
            ],
        }

        manifest_path = os.path.join(
            self.manifest_dir,
            "audio_description_manifest.json",
        )

        with open(manifest_path, "w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)

        return manifest_path

    def _build_output_video_name(self, video_path: str) -> str:
        base_name = os.path.basename(video_path)
        name, _ = os.path.splitext(base_name)

        return "{}_with_audio_description.mp4".format(name)

    def _safe_time(self, value: float) -> str:
        text = "{:.2f}".format(float(value))
        return text.replace(".", "_")
