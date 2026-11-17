"""Fast, offline tests; no model weights, GPU, network or Hugging Face login."""
from __future__ import annotations
import copy
from pathlib import Path
import pytest
from ai.spectra.Correlation.audio.diarization import align_transcription, normalize_diarization
from ai.spectra.Correlation.audio.speaker_linking import SpeakerLinkMemory
from ai.spectra.Correlation.audio.speaker_pipeline import SpeakerPipeline
from ai.spectra.Correlation.audio.audio_visual import LightASDRunner


def diag():
    return {'turns':[{'start':0,'end':2,'speaker_id':'SPEAKER_00'},
                     {'start':2,'end':4,'speaker_id':'SPEAKER_01'}],
            'exclusive_turns':[{'start':0,'end':2,'speaker_id':'SPEAKER_00'},
                               {'start':2,'end':4,'speaker_id':'SPEAKER_01'}]}


def test_align_whisper_preserves_text_and_timestamps():
    transcript={'language':'pt','full_text':'Olá. Meu nome é Marina.',
        'speech_segments':[{'start':0.1,'end':1.8,'text':'Olá.'},
                           {'start':2.2,'end':3.8,'text':'Meu nome é Marina.'}]}
    original=copy.deepcopy(transcript)
    new=align_transcription(transcript,diag())
    assert new['speech_segments'][0]['speaker_id']=='SPEAKER_00'
    assert new['speech_segments'][1]['speaker_id']=='SPEAKER_01'
    assert new['full_text']==original['full_text']
    assert [s['text'] for s in new['speech_segments']]==[s['text'] for s in original['speech_segments']]
    assert not any('speaker_id' in s for s in transcript['speech_segments'])


def test_near_equal_two_speakers_leaves_ambiguous():
    new=align_transcription({'speech_segments':[{'start':1,'end':3,'text':'sim e nao'}]},diag())
    assert new['speech_segments'][0]['speaker_id'] is None
    assert new['speech_segments'][0]['speaker_ambiguous']


def test_speaker_no_coverage_stays_unassigned():
    new=align_transcription({'speech_segments':[{'start':8,'end':10,'text':'fora da faixa'}]},diag())
    assert new['speech_segments'][0]['speaker_id'] is None
    assert not new['speech_segments'][0]['speaker_ambiguous']


def evidence(turn,person='person_0003',speaker='SPEAKER_00',score=1.9,runner=-999.,coverage=.88,frames=30):
    return {'turn_id':turn,'speaker_id':speaker,'entity_id':person,'raw_score':score,
            'runner_up_raw_score':runner,'face_coverage':coverage,'face_frames':frames}


def test_two_independent_turns_link_but_do_not_confirm_name():
    x=SpeakerLinkMemory().resolve([evidence('one'),evidence('two')],valid_entity_ids={'person_0003'})
    assert x['links']['SPEAKER_00']['status']=='high_confidence'
    assert x['speaker_entity_map']['SPEAKER_00']['entity_id']=='person_0003'
    assert not x['speaker_entity_map']['SPEAKER_00']['verified']


def test_one_long_turn_split_in_chunks_is_one_evidence():
    x=SpeakerLinkMemory().resolve([evidence('one',score=1.5),evidence('one',score=2.3)])
    assert x['links']['SPEAKER_00']['status']=='candidate'
    assert x['links']['SPEAKER_00']['evidence_count']==1
    assert x['speaker_entity_map']=={}


def test_ambiguous_faces_rejected():
    x=SpeakerLinkMemory().resolve([evidence('1',runner=1.85),evidence('2',runner=1.81)])
    assert x['links']=={}
    assert len(x['rejected'])==2


def test_low_coverage_rejected():
    x=SpeakerLinkMemory().resolve([evidence('1',coverage=.10),evidence('2',frames=6)])
    assert x['speaker_entity_map']=={}


def test_two_speakers_cannot_claim_one_person():
    x=SpeakerLinkMemory().resolve([evidence('a',speaker='A'),evidence('b',speaker='A'),
                                   evidence('c',speaker='B'),evidence('d',speaker='B')])
    assert x['speaker_entity_map']=={}
    assert all(v['status']=='conflict' for v in x['links'].values())


def test_reconciled_person_ids_are_canonical():
    x=SpeakerLinkMemory().resolve([evidence('1',person='person_0004'),
                                   evidence('2',person='person_0003')],
                                  valid_entity_ids={'person_0003'},
                                  remap={'person_0004':'person_0003'})
    assert x['speaker_entity_map']['SPEAKER_00']['entity_id']=='person_0003'


def test_audio_orchestrator_consumes_cached_models_without_loading(monkeypatch):
    class Bomb:
        def __getattr__(self,n):
            raise AssertionError('Nenhum modelo deve ser carregado quando ha cache')
    pipeline=SpeakerPipeline(diarizer=Bomb(), active_speaker_runner=Bomb())
    outcome=pipeline.run(
        speech_analysis={'speech_segments':[{'start':.25,'end':1.2,'text':'Oi!'}]},
        processed_frames=[],entities={'person_0003':{}},
        diarization=diag(), active_speaker={'available':True,'turn_scores':[
            evidence('a'),evidence('b')]})
    assert outcome['speech_analysis']['speech_segments'][0]['speaker_id']=='SPEAKER_00'
    assert outcome['speaker_links']['speaker_entity_map']['SPEAKER_00']['entity_id']=='person_0003'


def test_active_speaker_checks_missing_weights(tmp_path):
    runner=LightASDRunner(light_asd_root=tmp_path)
    with pytest.raises(FileNotFoundError,match='Light-ASD'):
        runner.run('no-video','no-audio',[],[])


def test_nested_light_asd_cache_is_accepted():
    result=SpeakerPipeline().run(
        speech_analysis={'speech_segments':[{'start':0.1,'end':0.8,'text':'Oi'}]},
        processed_frames=[],entities={'person_0003':{}},diarization=diag(),
        active_speaker={'active_speaker':{'available':True,'turn_scores':[evidence('1'),evidence('2')]}})
    assert result['speaker_links']['links']['SPEAKER_00']['entity_id']=='person_0003'


def test_pyannote_annotation_object_normalization():
    class Segment:
        def __init__(self,a,b):
            self.start,self.end=a,b
    class Annotation:
        def itertracks(self, yield_label=False):
            return iter([(Segment(0.,1.),'track_1','SPEAKER_A'),
                         (Segment(1.,2.),'track_2','SPEAKER_B')])
    class Output:
        speaker_diarization=Annotation()
        exclusive_speaker_diarization=Annotation()
    x=normalize_diarization(Output())
    assert x['speakers']==['SPEAKER_A','SPEAKER_B']
    assert x['exclusive_turns'][1]['start']==1.


def test_runner_deduplicates_chunks_of_one_turn(monkeypatch,tmp_path):
    root=tmp_path/'Light-ASD';(root/'model').mkdir(parents=True)
    (root/'model'/'Model.py').write_text('')
    (root/'weight').mkdir()
    (root/'weight'/'pretrain_AVA_CVPR.model').write_bytes(b'not-real-weights')
    vid=tmp_path/'test.mp4';vid.write_bytes(b'fake-video')
    wav=tmp_path/'test.wav';wav.write_bytes(b'fake-audio')
    class Process:
        returncode=0
        stdout=''
        stderr=''
    def fake_run(cmd,**kwargs):
        import json
        args=cmd[cmd.index('--input')+1]
        output=cmd[cmd.index('--output')+1]
        payload=json.loads(Path(args).read_text())
        assert payload['turns'][0]['turn_id']==payload['turns'][1]['turn_id']
        assert payload['turns'][0]['chunk_id']!=payload['turns'][1]['chunk_id']
        Path(output).write_text(json.dumps({'available':True,'turn_scores':[]}))
        return Process()
    import ai.spectra.Correlation.audio.audio_visual as av_mod
    monkeypatch.setattr(av_mod.subprocess,'run',fake_run)
    runner=LightASDRunner(root,max_turn_seconds=6,max_total_seconds=18)
    result=runner.run(vid,wav,[{'start':0.,'end':13.,'speaker_id':'VOICE_A'}],[])
    assert result['available']
