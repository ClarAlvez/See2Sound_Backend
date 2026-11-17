"""Testes offline v0.8. Sem FFmpeg, Whisper, modelos ou Hugging Face."""
from copy import deepcopy
import pytest

from ai.spectra.Correlation.speech_identity import (detect_mentions, anchored_names,
                                                    SpeechIdentityResolver)
from ai.spectra.Correlation.audio.audio_visual import plan_asd_tasks
from ai.spectra.Correlation.audio.speaker_linking import SpeakerLinkMemory


def test_self_intro_pt_and_en():
    assert ('Eleanor','self_introduction') in [(m['name'],m['kind'])
            for m in detect_mentions('Oi, Tiri. Sou Eleanor. Pode entrar.')]
    assert ('Marina','self_introduction') in [(m['name'],m['kind'])
            for m in detect_mentions('Eu sou Marina!')]
    assert detect_mentions("I'm Eleanor.")[0]['name']=='Eleanor'
    assert detect_mentions('Meu nome é Maria Clara.')[0]['name']=='Maria Clara'


@pytest.mark.parametrize('text',['Então, vamos embora.', 'Resumindo, isso foi tudo.',
                                 'Claramente, precisamos conversar.', 'Bom, certo.',
                                 'Vocês, venham aqui.', 'Sou professor.', 'Sou amigo.'])
def test_falso_nome_rejeitado(text):
    assert not detect_mentions(text)


def test_known_vocative_but_not_any_generic_word():
    known=anchored_names([{'text':'Sou Eleanor.'},{'text':'Oi, Chidi!'}])
    assert known=={'Eleanor'}
    match=detect_mentions('Eleanor, olha o que aconteceu.',known_names=known)
    assert len(match)==1 and match[0]['name']=='Eleanor' and match[0]['kind']=='address'
    assert not detect_mentions('Aprender, por favor.', known_names=known)
    assert detect_mentions('Marina!')[0]['kind']=='address'
    assert detect_mentions('Eleanor decidiu me pedir ajuda', known_names=known)[0]['kind']=='reference'


def test_name_grounding_requires_verified_link():
    analysis={'speech_segments':[{'start':0.,'end':1.5,'text':'Sou Eleanor.','speaker_id':'SPEAKER_02'},
                                 {'start':2.,'end':3.0,'text':'Eleanor, olha.','speaker_id':'SPEAKER_03'}]}
    frames=[{'timestamp':.5,'scene_id':'scene_0000','frame_id':0,
             'people':[{'entity_id':'person_0009'},{'entity_id':'person_0001'}]},
            {'timestamp':2.5,'scene_id':'scene_0000','frame_id':1,
             'people':[{'entity_id':'person_0009'},{'entity_id':'person_0001'}]}]
    entities={'person_0009':{},'person_0001':{}}
    base=SpeechIdentityResolver().resolve(analysis,frames,entities,
                    speaker_entity_map={'SPEAKER_02':{'entity_id':'person_0009','verified':False}})
    assert base['confirmed_names']=={}
    strong=SpeechIdentityResolver().resolve(analysis,frames,entities,
                    speaker_entity_map={'SPEAKER_02':{'entity_id':'person_0009','verified':True}})
    assert strong['confirmed_names']=={'person_0009':'Eleanor'}
    # O nome no vocativo e da pessoa CHAMADA, nao de quem fala.
    voc=strong['segments'][1]['name_mentions'][0]
    assert voc['kind']=='address' and voc['entity_id'] is None


def test_budget_reserves_name_anchor_and_time_coverage():
    turns=[]
    for i in range(25):
        turns.append({'start':float(i*16), 'end':float(i*16+7),
                      'speaker_id':'SPEAKER_00' if i%2 else 'SPEAKER_01'})
    speech=[{'start':241.,'end':247.,'speaker_id':'SPEAKER_00','text':'Sou Eleanor.'}]
    tasks, summary = plan_asd_tasks(turns,speech_segments=speech,
                                   max_turn_seconds=5,max_total_seconds=55)
    assert sum(t['end']-t['start'] for t in tasks) <=55.00001
    assert summary['policy']=='name_anchored_speaker_balanced'
    assert any(240<=t['start']<248 for t in tasks)
    assert any(t['start']<100 for t in tasks)
    assert len({t['speaker_id'] for t in tasks})==2
    assert len({t['turn_id'] for t in tasks})>=3


def test_long_turn_chunks_are_one_evidence():
    tasks, summary=plan_asd_tasks([{'start':0,'end':13,'speaker_id':'A'}],
                                 max_turn_seconds=6,max_total_seconds=18)
    assert len(tasks)==2
    assert len({t['turn_id'] for t in tasks})==1
    assert len({t['chunk_id'] for t in tasks})==2


def ev(turn,speaker,score=1.5):
    return {'turn_id':str(turn),'speaker_id':speaker,'entity_id':'person_0001',
            'raw_score':score,'runner_up_raw_score':-999.,'face_coverage':.91,'face_frames':24}


def test_two_speakers_equal_evidence_remain_conflict():
    data=[ev(x,'A') for x in range(3)]+[ev(x,'B') for x in range(3)]
    x=SpeakerLinkMemory().resolve(data,valid_entity_ids={'person_0001'})
    assert x['speaker_entity_map']=={}
    assert x['status_counts']=={'conflict':2}


def test_extreme_dominance_can_resolve_conflict():
    data=[ev(x,'A',2.3) for x in range(7)]+[ev(x,'B',1.1) for x in range(2)]
    x=SpeakerLinkMemory().resolve(data,valid_entity_ids={'person_0001'})
    assert x['speaker_entity_map']['A']['entity_id']=='person_0001'
    assert x['links']['B']['status']=='conflict'
    assert x['links']['A']['verified'] is False


def test_cached_replay_and_manual_review():
    from tools.replay_identity_v08 import replay
    result={'correlation':{
      'frames':[{'frame_id':0,'timestamp':0.3,'scene_id':'scene_0000',
                 'people':[{'entity_id':'person_0001'}]}],
      'entities':{'person_0001':{}}, 'entity_profiles':{'person_0001':{}},
      'audio_identity':{'segments':[{'segment_id':'speech_0000','start':0.,'end':1.5,
                                     'text':'Sou Eleanor.','speaker_id':'SPEAKER_02'}]},
      'active_speaker':{'turn_scores':[]}},
      'audio':{'speech_analysis':{'language':'pt'}},'spectra':{'outputs':[]}}
    old=deepcopy(result)
    r=replay(result,verified_map={'SPEAKER_02':'person_0001'})
    assert r['correlation']['audio_identity']['confirmed_names']=={'person_0001':'Eleanor'}
    assert old['correlation']['audio_identity']['segments'][0]['text']=='Sou Eleanor.'
    assert r['identity_v08_diagnostics']['narrative_and_video_unchanged']


def test_patch_idempotent():
    from patch_v08 import patch_pipeline,patch_runner
    source='''# SPECTRA_CORRELATION_V06_BRIDGE
# SPECTRA_NARRATIVE_V07
    correlation_light_asd_root: str = 'third_party/Light-ASD',
                light_asd_root=correlation_light_asd_root,
    voice_result: Optional[
'''
    result,applied=patch_pipeline(source)
    assert applied and 'correlation_auto_confirm_names' in result
    assert patch_pipeline(result)==(result,False)
    runner="""    parser.add_argument('--skip-diarization', action='store_true')
    parser.add_argument('--skip-tts', action='store_true')
        correlation_run_active_speaker=not args.skip_asd,
"""
    result,applied=patch_runner(runner)
    assert applied and '--auto-confirm-names' in result
    assert patch_runner(result)==(result,False)


def test_focus_prioritizes_persons_and_controls_rate():
    from ai.narrative.focus_filter_v08 import select_focused_descriptions
    frames=[{'scene_id':f'scene_{i:04d}','people':[{'entity_id':'person_0001'}]}
            for i in range(10)]
    timeline=[]
    for i in range(14):
        name=(i==5 or i==8)
        persons = [{'entity_id':'person_0001','confirmed_name':'Eleanor' if name else None,
                    'action':'walking' if i in (5,8) else 'sitting',
                    'relation':None}] if i in (0,5,8,10) else []
        timeline.append({'start_time':i*5.,'end_time':i*5.+3,
                         'description':f'Descricao {i}',
                         'scene_context':{'correlated_people':persons}})
    baseline=list(timeline)
    selected=select_focused_descriptions(timeline,correlation_result={'frames':frames},
                                         max_per_minute=3,min_gap_seconds=8)
    assert len([x for x in selected if x['start_time']<60])<=3
    assert all(a['start_time'] < b['start_time'] for a,b in zip(selected,selected[1:]))
    assert any(x['start_time']==25. for x in selected)
    assert any(x['start_time']==40. for x in selected)
    assert timeline==baseline
    assert all('focus_v08' in x['scene_context'] for x in selected)


def test_focus_rejects_zero_capacity():
    from ai.narrative.focus_filter_v08 import select_focused_descriptions
    with pytest.raises(ValueError):
        select_focused_descriptions([{'start_time':0.,'description':'teste'}],max_per_minute=0)
