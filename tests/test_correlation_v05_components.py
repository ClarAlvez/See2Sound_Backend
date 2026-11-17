"""Testes sem pesos de IA, executados depois de copiar o pacote ao repo."""
from ai.spectra.Correlation.feature_vectors import AttributeVectorMemory
from ai.spectra.Correlation.context_memory import ContextMemory
from ai.spectra.Correlation.presence_memory import PresenceMemory
from ai.spectra.Correlation.requesters.object import ObjectRequester
from ai.spectra.Correlation.requesters.scene import SceneRequester
from ai.spectra.Correlation.requesters.atmosphere import AtmosphereRequester


def test_vectors_maintain_mask_and_merge():
    mem = AttributeVectorMemory()
    mem.update('person', 'person_0001', 0, 0.0, {'black_clothes': 0.9})
    mem.update('person', 'person_0001', 1, 0.5, {'black_clothes': 0.8})
    mem.update('person', 'person_0002', 1, 0.5, {'black_clothes': 0.95})
    mem.remap_person_ids({'person_0002': 'person_0001'})
    result = mem.summary('person', 'person_0001')
    idx = result['labels'].index('black_clothes')
    assert result['observation_count'] == 2
    assert result['observed_mask'][idx] == 1
    assert abs(result['values'][idx] - 0.925) < 0.0001
    assert result['frame_ids'] == [0, 1]


def test_requests_object_detector_and_classifier_separate():
    request = ObjectRequester().request({'frame_id': 2, 'raw': {'object': {'crop_results': [
        {'bbox_xyxy': {'x1': 10, 'y1': 20, 'x2': 100, 'y2': 200},
         'detector_label': 'backpack', 'detector_score': 0.60,
         'mapped_labels': ['backpack', 'bag'], 'predictions': [
             {'label': 'backpack', 'score': 0.90}]},
        {'bbox_xyxy': {'x1': 40, 'y1': 40, 'x2': 60, 'y2': 60},
         'detector_label': 'umbrella', 'detector_score': 0.30, 'predictions': []},
    ]}}})
    assert len(request) == 2
    assert request[0]['classification_source'] == 'classifier'
    assert request[0]['label_score'] == 0.90
    assert request[1]['classification_source'] == 'detector_fallback'
    assert request[1]['label_score'] == 0.30


def test_scene_and_atmosphere_from_raw():
    frame = {'raw': {
        'scene': {'predictions': [{'label': 'street', 'score': 0.92}],
                  'derived_predictions': [{'label': 'office', 'score': 0.70}]},
        'atmosphere': {'predictions': [{'label': 'night', 'score': 0.94}]},
    }}
    assert SceneRequester().request(frame)['attributes']['street'] == 0.92
    assert SceneRequester().request(frame)['derived_attributes']['office'] == 0.70
    assert AtmosphereRequester().request(frame)['attributes']['night'] == 0.94


def test_temporal_context_and_presence():
    context = ContextMemory('atmosphere', min_observations=2)
    context.observe('s1', 0, 0.0, {'day': 0.88})
    context.observe('s1', 1, 0.5, {'day': 0.92})
    assert context.to_dict()['s1']['confirmed_attributes']['day'] == 0.9
    presence = PresenceMemory(gap_seconds=1)
    person = [{'entity_id': 'person_0003'}]
    assert presence.observe(0, 0.0, 's1', person)[0]['event'] == 'first_appearance'
    assert not presence.observe(1, 0.5, 's1', person)
    assert presence.observe(4, 2.0, 's1', person)[0]['event'] == 'reappearance'
    assert presence.observe(5, 2.5, 's2', person, True)[0]['event'] == 'visible_after_cut'


def test_real_pipeline_person_raw_crop_results():
    from ai.spectra.Correlation.requesters.person import PersonRequester
    from ai.spectra.Correlation.pipeline_adapter import flatten_spectra_outputs
    scene_outputs = [{
        'start_time': 0.0, 'end_time': 2.0,
        'context': {'scene_index': 3},
        'raw_frame_results': [{
            'timestamp': 0.5, 'frame_path': 'x_frame_000001.jpg',
            'raw': {'person': {'source': 'person_crops', 'raw_crop_results': [{
                'crop': {'detection_bbox': [10, 20, 100, 200], 'crop_path': 'c1.jpg'},
                'result': {'predictions': [{'label': 'woman', 'score': 0.93}]},
            }]}}
        }]
    }]
    flat = flatten_spectra_outputs(scene_outputs)
    assert len(flat) == 1
    assert flat[0]['scene_id'] == 'scene_0003'
    result = PersonRequester().request(flat[0])
    assert result[0]['bbox'] == [10.0, 20.0, 100.0, 200.0]
    assert result[0]['attributes']['woman'] == 0.93
    assert result[0]['crop_path'] == 'c1.jpg'
