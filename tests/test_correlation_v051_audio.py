"""Testes do modulo de transcricao -> identidade, sem Whisper/GPU."""
from ai.spectra.Correlation.speech_identity import SpeechIdentityResolver, detect_mentions
from ai.spectra.Correlation.requesters.transcription import TranscriptionRequester
from ai.spectra.Correlation import CorrelationEngine, CorrelationConfig


class SyntheticEncoder:
    name = 'synthetic'
    embedding_dimension = 2
    def encode_batch(self, paths):
        raise AssertionError('Nao deve rodar encoder com embeddings fornecidos')


def simple_frames():
    return [
        {'frame_id': i, 'timestamp': i * 0.5, 'scene_id': 'scene_0001',
         'people': [{'detection_id': f'p{i}', 'bbox': [10, 10, 200, 400],
                     'embedding': [1.0, 0.0], 'attributes': {'woman': 0.9},
                     'action_predictions': [{'label': 'talking', 'score': 0.90}]}]}
        for i in range(3)
    ]


def whisper(text='Meu nome é Marina.', *, speaker=None, start=0., end=1.):
    row = {'start': start, 'end': end, 'text': text}
    if speaker is not None:
        row['speaker_id'] = speaker
    return {'language': 'pt', 'speech_segments': [row]}


def test_detect_mentions_conservative():
    assert detect_mentions('Meu nome é Marina.')[0]['name'] == 'Marina'
    assert detect_mentions('Esta é a Maria Clara.')[0]['name'] == 'Maria Clara'
    assert any(x['name'] == 'João' and x['kind'] == 'address'
               for x in detect_mentions('Ei, João!'))
    assert not detect_mentions('Oi, tudo bem!')
    assert not detect_mentions('Obrigado por tudo.')


def test_transcription_contract():
    normalized = TranscriptionRequester().request({'audio': {'speech_analysis': whisper()}})
    assert normalized['segments'][0]['segment_id'] == 'speech_0000'
    assert normalized['segments'][0]['speaker_id'] is None
    assert normalized['diarization_available'] is False


def test_no_voice_grounding_does_not_confirm():
    engine = CorrelationEngine(config=CorrelationConfig(enable_reconciliation=False),
                               identity_encoder=SyntheticEncoder())
    payload = {'frames': simple_frames(), 'audio': {'speech_analysis': whisper()}}
    result = engine.process_video(payload)
    speech = result['audio_identity']
    assert speech['available']
    assert speech['name_candidates']['person_0001'][0]['name'] == 'Marina'
    assert speech['name_candidates']['person_0001'][0]['status'] == 'candidate'
    assert speech['confirmed_names'] == {}
    assert result['entity_profiles']['person_0001']['display_name'] == 'Pessoa 1'
    assert speech['segments'][0]['scene_ids'] == ['scene_0001']


def test_verified_speaker_self_introduction_can_confirm():
    engine = CorrelationEngine(config=CorrelationConfig(enable_reconciliation=False),
                               identity_encoder=SyntheticEncoder())
    payload = {'frames': simple_frames(), 'audio': {'speech_analysis': whisper(speaker='VOICE_A')}}
    result = engine.process_video(payload, speaker_entity_map={'VOICE_A': 'person_0001'})
    assert result['audio_identity']['confirmed_names'] == {'person_0001': 'Marina'}
    assert result['entity_profiles']['person_0001']['display_name'] == 'Marina'
    assert result['frames'][0]['people'][0]['entity_id'] == 'person_0001'
    assert result['frames'][0]['people'][0]['display_name'] == 'Marina'
    assert result['presence']['events'][0]['display_name'] == 'Marina'


def test_unverified_map_stays_candidate():
    engine = CorrelationEngine(config=CorrelationConfig(enable_reconciliation=False),
                               identity_encoder=SyntheticEncoder())
    result = engine.process_video({'frames': simple_frames(),
                                   'audio': {'speech_analysis': whisper(speaker='A')}},
                                  speaker_entity_map={'A': {'entity_id': 'person_0001', 'verified': False}})
    assert result['audio_identity']['confirmed_names'] == {}


def test_vocative_is_not_speaker_name():
    engine = CorrelationEngine(config=CorrelationConfig(enable_reconciliation=False),
                               identity_encoder=SyntheticEncoder())
    result = engine.process_video({'frames': simple_frames(),
                                   'audio': {'speech_analysis': whisper('Marina!', speaker='VOICE_A')}},
                                  speaker_entity_map={'VOICE_A': 'person_0001'})
    assert result['audio_identity']['confirmed_names'] == {}
    # O falante nao pode ser automaticamente o alvo da chamada.
    assert result['audio_identity']['unassigned_name_mentions']


def test_explicit_verified_mention_target_can_confirm():
    engine = CorrelationEngine(config=CorrelationConfig(enable_reconciliation=False),
                               identity_encoder=SyntheticEncoder())
    result = engine.process_video({'frames': simple_frames(),
                                   'audio': {'speech_analysis': whisper('Esta é a Marina.')}},
                                  entity_hints=[{'segment_id':'speech_0000', 'name':'Marina',
                                                 'entity_id':'person_0001', 'verified':True,
                                                 'method':'human_review'}])
    assert result['audio_identity']['confirmed_names']['person_0001'] == 'Marina'


def test_no_diarization_and_multiple_people_is_ambiguous():
    resolver = SpeechIdentityResolver()
    frames = [{'frame_id': 0, 'timestamp': 0.2, 'scene_id': 'scene_0001',
               'people': [{'entity_id': 'person_0001'}, {'entity_id': 'person_0002'}]}]
    result = resolver.resolve(whisper('Olá, Marina!'), frames,
                              {'person_0001': {}, 'person_0002': {}})
    assert result['confirmed_names'] == {}
    assert result['unassigned_name_mentions']


def test_scene_alignment_uses_intervals():
    resolver = SpeechIdentityResolver()
    frames = [{'frame_id': 0, 'timestamp': 0., 'scene_id': 'scene_0001', 'people': []},
              {'frame_id': 1, 'timestamp': 5., 'scene_id': 'scene_0002', 'people': []}]
    payload = {'spectra': {'outputs': [
        {'start_time': 0., 'end_time': 3., 'context': {'scene_index': 1}},
        {'start_time': 3., 'end_time': 6., 'context': {'scene_index': 2}},
    ]}}
    result = resolver.resolve(whisper('Meu nome é Marina', start=4., end=4.8), frames,
                              {}, pipeline_payload=payload)
    assert result['segments'][0]['scene_ids'] == ['scene_0002']


def test_duplicate_frames_do_not_inflate_name_evidence():
    resolver = SpeechIdentityResolver()
    frames = [{'frame_id': i, 'timestamp': i / 10., 'scene_id': 's',
               'people': [{'entity_id': 'person_0001'}]} for i in range(10)]
    result = resolver.resolve(whisper('Meu nome é Marina'), frames, {'person_0001': {}})
    assert result['name_candidates']['person_0001'][0]['mention_count'] == 1
