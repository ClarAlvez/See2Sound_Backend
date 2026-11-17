"""Bridge See2Sound pipeline. Optional strong-ASD name confirmation (v0.8)."""
from __future__ import annotations
from ai.spectra.Correlation.config import CorrelationConfig
from ai.spectra.Correlation.engine import CorrelationEngine


def correlate_spectra_with_speech(spectra_outputs, whisper_result, *,
                                  engine=None, source_video_path=None, audio_path=None,
                                  run_diarization=False, run_active_speaker=False,
                                  light_asd_root='third_party/Light-ASD',
                                  speaker_entity_map=None, entity_hints=None,
                                  scene_intervals=None, auto_confirm_names=False):
    config = CorrelationConfig(light_asd_root=light_asd_root,
                               speaker_auto_confirm_names=bool(auto_confirm_names))
    engine = engine or CorrelationEngine(config=config)
    payload={'spectra': {'outputs': spectra_outputs},
             'audio': {'speech_analysis': whisper_result},
             'source_video_path': source_video_path,
             'scene_intervals': scene_intervals or []}
    return engine.process_video(payload, speaker_entity_map=speaker_entity_map,
                                entity_hints=entity_hints,
                                run_diarization=run_diarization,
                                run_active_speaker=run_active_speaker,
                                audio_path=audio_path, video_path=source_video_path)
