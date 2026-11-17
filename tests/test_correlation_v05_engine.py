"""Teste de integracao sem rede/pesos pretreinados (requer modulos legados)."""
from ai.spectra.Correlation import CorrelationEngine, CorrelationConfig


class DummyIdentityEncoder:
    name = 'synthetic'
    embedding_dimension = 2

    def encode_batch(self, paths):
        raise AssertionError('O teste injeta embeddings: nao deve fazer inferencia.')


def test_five_models_in_one_engine():
    config = CorrelationConfig()
    config.enable_reconciliation = False
    engine = CorrelationEngine(config=config, identity_encoder=DummyIdentityEncoder())
    frames = []
    for fid in range(3):
        frames.append({
            'frame_id': fid, 'timestamp': fid * 0.5, 'scene_id': 'scene_A',
            'people': [{
                'detection_id': f'person_{fid}', 'bbox': [100, 100, 300, 500],
                'embedding': [1.0, 0.0],
                'attributes': {'woman': 0.94, 'black_clothes': 0.85},
                'action_predictions': [{'label': 'carrying', 'score': 0.88},
                                       {'label': 'walking', 'score': 0.91}],
            }],
            'raw': {
                'object': {'crop_results': [{
                    'bbox_xyxy': {'x1': 130, 'y1': 200, 'x2': 190, 'y2': 300},
                    'detector_label': 'backpack', 'detector_score': 0.95,
                    'predictions': [{'label': 'backpack', 'score': 0.91}],
                }]},
                'scene': {'predictions': [{'label': 'outdoor', 'score': 0.95},
                                          {'label': 'street', 'score': 0.92}]},
                'atmosphere': {'predictions': [{'label': 'day', 'score': 0.97},
                                               {'label': 'bright', 'score': 0.92}]},
            },
        })
    result = engine.process_video(frames)
    assert result['frame_count'] == 3
    assert len(result['scenes']) == 1
    assert result['atmospheres']['scene_A']['confirmed_attributes']['day'] >= 0.9
    assert len(result['entities']) >= 1
    assert len(result['person_actions']) >= 1
    assert result['feature_vectors']['person']
    assert result['feature_vectors']['object']
    assert result['feature_vectors']['action']
    assert result['feature_vectors']['scene']
    assert result['feature_vectors']['atmosphere']
    assert result['presence']['events'][0]['event'] == 'first_appearance'
    assert any(rel['relation'] == 'carries' and rel['object_label'] == 'backpack'
               for rels in result['person_object_relations'].values() for rel in rels)
