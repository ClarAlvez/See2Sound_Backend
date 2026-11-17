"""Central optional audio/video speaker resolver for the Correlation engine.

Must be invoked after person reconciliation, so its visual links use FINAL
entity_id; results are fed back to the pre-existing name resolver.
"""
from __future__ import annotations
from pathlib import Path
from .diarization import SpeakerDiarizer, align_transcription, normalize_diarization
from .audio_visual import LightASDRunner
from .speaker_linking import SpeakerLinkMemory


class SpeakerPipeline:
    def __init__(self, config=None, *, diarizer=None, active_speaker_runner=None):
        self.config=config
        self.diarizer=diarizer
        self.active_speaker_runner=active_speaker_runner

    def run(self, *, speech_analysis, processed_frames, entities,
            audio_path=None, video_path=None, diarization=None, active_speaker=None,
            run_diarization=False, run_active_speaker=False, remap=None, scene_intervals=None):
        analysis=dict(speech_analysis or {})
        warnings=[]
        if run_active_speaker:
            # The main pipeline defaults to 6 frames per scene. Sparse visual
            # samples cannot safely ground a dense face track to person IDs.
            by_scene={}
            for row in processed_frames:
                by_scene.setdefault(str(row.get('scene_id')),[]).append(float(row['timestamp']))
            gaps=[b-a for times in by_scene.values()
                  for a,b in zip(sorted(times),sorted(times)[1:])]
            if gaps and max(gaps)>1.5:
                warnings.append('Frames de Person estao esparsos dentro da cena (>1.5s): '
                                'Light-ASD pode nao vincular nenhum rosto a person_id. '
                                'Considere aumentar max_temporal_frames_per_scene para o teste.')
        raw_available=bool(analysis.get('speech_segments') or analysis.get('segments'))
        if run_diarization:
            if not audio_path:
                raise ValueError('Diarizacao exige --audio-file ou audio.speech_analysis.audio_path.')
            provider=self.diarizer or SpeakerDiarizer(
                model_id=getattr(self.config,'diarization_model_id','pyannote/speaker-diarization-community-1'),
                device=getattr(self.config,'diarization_device',None))
            # SEE2SOUND_PROGRESS_V1
            from tools.pipeline_progress_v1 import emit_progress as _emit_progress
            _emit_progress("correlation", None, None, "pyannote: diarizacao")
            diarization=provider.diarize(audio_path)
        diag=normalize_diarization(diarization) if diarization is not None else {
            'available':False,'turns':[],'exclusive_turns':[],'speakers':[]}
        if diag['available']:
            analysis=align_transcription(analysis,diag,
                min_coverage=float(getattr(self.config,'speaker_alignment_min_coverage',0.35)),
                min_margin=float(getattr(self.config,'speaker_alignment_min_margin',0.10)))
        if run_active_speaker:
            if not diag['available']:
                raise ValueError('Light-ASD requer diarizacao. Use --diarize ou --diarization-json.')
            if not video_path or not audio_path:
                raise ValueError('Light-ASD requer video e audio WAV extraido.')
            runner=self.active_speaker_runner or LightASDRunner(
                light_asd_root=getattr(self.config,'light_asd_root','third_party/Light-ASD'),
                device=getattr(self.config,'light_asd_device','auto'),
                timeout_seconds=getattr(self.config,'light_asd_timeout_seconds',900),
                max_turn_seconds=getattr(self.config,'light_asd_max_turn_seconds',8.),
                max_total_seconds=getattr(self.config,'light_asd_max_total_seconds',180.))
            from tools.pipeline_progress_v1 import emit_progress as _emit_progress
            _emit_progress("correlation", None, None, "Light-ASD: associacao audiovisual")
            active_speaker=runner.run(video_path,audio_path,diag['exclusive_turns'],processed_frames,
                                      scene_intervals=scene_intervals,
                                      speech_segments=analysis.get('speech_segments') or analysis.get('segments') or [])
        if isinstance(active_speaker,dict) and isinstance(active_speaker.get('active_speaker'),dict):
            active_speaker=active_speaker['active_speaker']
        active_speaker=active_speaker or {'available':False,'turn_scores':[]}
        allow_auto = bool(getattr(self.config,'speaker_auto_confirm_names',False))
        linking=SpeakerLinkMemory(
            min_turns=max(3,int(getattr(self.config,'speaker_min_confirming_turns',2))) if allow_auto else int(getattr(self.config,'speaker_min_confirming_turns',2)),
            min_raw_score=float(getattr(self.config,'speaker_asd_min_raw_score',0.0)),
            min_turn_margin=float(getattr(self.config,'speaker_asd_min_turn_margin',0.3)),
            min_support_fraction=max(0.85,float(getattr(self.config,'speaker_min_support_fraction',0.7))) if allow_auto else float(getattr(self.config,'speaker_min_support_fraction',0.7)),
            min_face_coverage=max(0.65,float(getattr(self.config,'speaker_min_face_coverage',0.5))) if allow_auto else float(getattr(self.config,'speaker_min_face_coverage',0.5)),
            min_face_frames=int(getattr(self.config,'speaker_min_face_frames',12)),
            allow_automatic_name_confirmation=allow_auto,
        ).resolve(active_speaker.get('turn_scores') or [], valid_entity_ids=entities, remap=remap)
        return {'speech_analysis':analysis, 'diarization':diag,'active_speaker':active_speaker,
                'speaker_links':linking,
                'warnings':warnings+([] if diag['available'] else ['Diarizacao nao executada.'])+
                           ([] if active_speaker.get('available') else ['ASD nao executado.']),
                'transcription_available':raw_available}
