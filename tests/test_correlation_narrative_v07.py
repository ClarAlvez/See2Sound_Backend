"""pytest -q tests/test_correlation_narrative_v07.py"""
import re
from ai.narrative.correlation_narrative import (
    CorrelationNarrativeGenerator, build_correlated_scene_facts,
)


def payload(include_second=False):
    frames = []
    for fid, t in [(0, 0.5), (1, 1.0), (2, 3.5), (3, 4.0)]:
        scene_id = 'scene_0001' if fid < 2 else 'scene_0002'
        girl = {'entity_id': 'person_0001', 'track_id': 'track_0001',
                'attributes': {'woman': .94, 'long_hair': .95, 'black_clothes': .88}}
        people = [girl]
        actions = [{'entity_id': 'person_0001', 'predictions': [{'label': 'walking', 'score': .93}]}]
        relations = [{'entity_id': 'person_0001', 'relation': 'carries',
                      'relation_kind': 'interaction', 'object_label': 'backpack',
                      'object_score': .94, 'evidence_score': .91, 'accepted': True}]
        if include_second and fid >= 2:
            people.append({'entity_id': 'person_0002', 'track_id': 'track_0002',
                           'attributes': {'man': .96}})
            actions.append({'entity_id': 'person_0002', 'predictions': [{'label': 'running', 'score': .97}]})
        frames.append({'frame_id': fid, 'timestamp': t, 'scene_id': scene_id,
                       'people': people, 'actions': actions, 'object_associations': relations})
    return {
        'frames': frames,
        'person_actions': {
            'person_0001': [{'label': 'walking', 'persistent': True}],
            'person_0002': [{'label': 'running', 'persistent': True}],
        },
        'person_object_relations': {'person_0001': [
            {'relation': 'carries', 'object_label': 'backpack', 'persistent': True}]},
        'scenes': {'scene_0001': {'confirmed_attributes': {'street': .91}},
                   'scene_0002': {'confirmed_attributes': {'park': .92}}},
        'atmospheres': {'scene_0001': {'confirmed_attributes': {'day': .96}},
                        'scene_0002': {'confirmed_attributes': {'bright': .93}}},
        'audio_identity': {'confirmed_names': {'person_0001': 'Marina'},
                           'name_candidates': {'person_0002': [{'name': 'Clara'}]}},
    }


def scenes():
    return [
        {'start_time': 0, 'end_time': 2, 'labels': ['woman', 'street'],
         'confidence': {'person.woman': .94, 'scene.street': .91},
         'context': {'scene_index': 1}},
        {'start_time': 3, 'end_time': 5, 'labels': ['woman', 'park'],
         'confidence': {'person.woman': .94, 'scene.park': .92},
         'context': {'scene_index': 2}},
    ]


def test_builder_does_not_mutate_spectra():
    items = scenes()
    updated = build_correlated_scene_facts(items, payload())
    assert 'correlation_narrative_v07' not in items[0]['context']
    assert updated[0]['context']['correlation_narrative_v07']['environment'] == 'street'
    assert updated[1]['context']['correlation_narrative_v07']['environment'] == 'park'


def test_full_narrative_uses_confirmed_name_action_and_relation():
    gen = CorrelationNarrativeGenerator(use_llm=False)
    result = gen.generate_timeline_from_dicts(scenes(), correlation_result=payload())
    assert result and result[0]['start_time'] == 0 and result[0]['end_time'] == 2
    first = result[0]['description']
    assert 'Marina' in first and 'caminha' in first and 'mochila' in first
    assert 'rua' in first


def test_second_speaker_action_is_not_attributed_to_first():
    output = build_correlated_scene_facts(scenes(), payload(include_second=True))
    second_scene = output[1]['context']['correlation_narrative_v07']
    people = {x['entity_id']: x for x in second_scene['people']}
    assert people['person_0001']['action'] == 'walking'
    assert people['person_0002']['action'] == 'running'
    assert people['person_0002']['relation'] is None


def test_name_candidate_never_promoted():
    output = build_correlated_scene_facts(scenes(), payload(include_second=True))
    people = output[1]['context']['correlation_narrative_v07']['people']
    assert next(x for x in people if x['entity_id'] == 'person_0002')['confirmed_name'] is None


def test_low_object_score_is_not_narrated():
    p = payload()
    for f in p['frames']:
        f['object_associations'][0]['object_score'] = .28
    output = build_correlated_scene_facts(scenes(), p)
    assert output[0]['context']['correlation_narrative_v07']['people'][0]['relation'] is None


def test_no_action_from_other_scene():
    p = payload()
    for f in p['frames'][:2]:
        f['actions'] = []
    output = build_correlated_scene_facts(scenes(), p)
    assert output[0]['context']['correlation_narrative_v07']['people'][0]['action'] is None
    assert output[1]['context']['correlation_narrative_v07']['people'][0]['action'] == 'walking'


def test_delta_skips_same_unchanged_character():
    p = payload()
    p['scenes']['scene_0002'] = p['scenes']['scene_0001']
    p['atmospheres']['scene_0002'] = p['atmospheres']['scene_0001']
    enriched = build_correlated_scene_facts(scenes(), p)
    gen = CorrelationNarrativeGenerator(use_llm=False)
    # Garante que a assinatura da identidade permanece estável.
    a = enriched[0]['context']['correlation_narrative_v07']['people'][0]
    b = enriched[1]['context']['correlation_narrative_v07']['people'][0]
    assert a['entity_id'] == b['entity_id']


def test_voice_engine_contract():
    gen = CorrelationNarrativeGenerator(use_llm=False)
    timeline = gen.generate_timeline_from_dicts(scenes(), correlation_result=payload())
    for row in timeline:
        assert all(k in row for k in ('start_time', 'end_time', 'description'))
        assert isinstance(row['description'], str) and row['description'].strip()


def test_missing_correlation_falls_back():
    gen = CorrelationNarrativeGenerator(use_llm=False)
    timeline = gen.generate_timeline_from_dicts(scenes(), correlation_result=None)
    assert isinstance(timeline, list)


def test_name_not_spoiled_before_it_is_spoken():
    p = payload()
    p['audio_identity']['segments'] = [{
        'start': 3.5, 'end': 4.2,
        'name_mentions': [{'name': 'Marina', 'verified': True,
                           'entity_id': 'person_0001'}],
    }]
    enriched = build_correlated_scene_facts(scenes(), p)
    assert enriched[0]['context']['correlation_narrative_v07']['people'][0]['confirmed_name'] is None
    assert enriched[1]['context']['correlation_narrative_v07']['people'][0]['confirmed_name'] is None
    # Uma cena posterior a 4.2 poderá utilizar o nome confirmado.
    extra = {'start_time': 5.0, 'end_time': 7.0, 'labels': ['woman'],
             'context': {'scene_index': 3}}
    p['frames'].append({'frame_id': 5, 'timestamp': 5.5, 'scene_id': 'scene_0003',
                        'people': [p['frames'][0]['people'][0]],
                        'actions': [], 'object_associations': []})
    later = build_correlated_scene_facts(scenes() + [extra], p)[2]
    assert later['context']['correlation_narrative_v07']['people'][0]['confirmed_name'] == 'Marina'
