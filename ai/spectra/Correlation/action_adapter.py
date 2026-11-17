"""Converte previsoes de Actions em evidencias por entidade.

Aceita:
- frame['people'][i]['action_predictions'] (mesmos crops de Person),
- frame['actions'] por detection_id,
- frame['raw']['action']['crop_results'] (PersonActionAnalyzer).

Nao atribui as previsoes agregadas de frame a uma pessoa arbitraria.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


def _xyxy(value):
    if isinstance(value, dict):
        value = [value.get(k) for k in ('x1', 'y1', 'x2', 'y2')]
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(v is None for v in value):
        return None
    try:
        a = tuple(float(v) for v in value)
    except (ValueError, TypeError):
        return None
    if a[2] <= a[0] or a[3] <= a[1]:
        return None
    return a


def _iou(a, b):
    if a is None or b is None:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    area_i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - area_i
    return area_i / union if union > 0 else 0.0


def _normal_predictions(value, minimum):
    if isinstance(value, dict):
        value = value.get('predictions', value.get('top_predictions', []))
    best = {}
    for item in value or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get('label') or '').strip().lower()
        try:
            score = max(0.0, min(1.0, float(item.get('score', 0))))
        except (TypeError, ValueError):
            continue
        if label and score >= minimum:
            best[label] = max(score, best.get(label, 0.0))
    return [{'label': label, 'score': round(score, 4)}
            for label, score in sorted(best.items(), key=lambda x: -x[1])]


class CorrelationActionAdapter:
    def __init__(self, min_score: float = 0.30,
                 min_iou: float = 0.20, min_margin: float = 0.07):
        self.min_score = min_score
        self.min_iou = min_iou
        self.min_margin = min_margin

    def process_result(self, frame: Dict[str, Any],
                       correlated_people: List[Dict[str, Any]]):
        """Retorna uma observacao de Actions por pessoa com previsoes.

        correlated_people precisa conter detection_id, entity_id, track_id, bbox.
        """
        by_detection = {p['detection_id']: p for p in correlated_people}
        evidence: Dict[str, Dict[str, float]] = {}
        source: Dict[str, str] = {}

        def add(det_id, predictions, origin):
            if det_id not in by_detection:
                return
            normalized = _normal_predictions(predictions, self.min_score)
            if not normalized:
                return
            record = evidence.setdefault(det_id, {})
            for item in normalized:
                label, score = item['label'], item['score']
                record[label] = max(score, record.get(label, 0.0))
            source.setdefault(det_id, origin)

        # Preferencia absoluta: crops utilizados no proprio PersonCropper.
        for p in frame.get('people', frame.get('persons', [])):
            if 'action_predictions' in p:
                add(p.get('detection_id'), p['action_predictions'], 'person_crop')

        direct = frame.get('actions')
        if isinstance(direct, dict):
            for det_id, predictions in direct.items():
                add(det_id, predictions, 'detection_id')
        elif isinstance(direct, list):
            for item in direct:
                if isinstance(item, dict) and 'detection_id' in item:
                    add(item['detection_id'], item.get('predictions', []), 'detection_id')

        # Segunda opcao: matcher geometrico entre dois PersonCroppers.
        action_raw = (frame.get('raw') or {}).get('action') or {}
        raw_crops = action_raw.get('crop_results', []) if isinstance(action_raw, dict) else []
        used_detection_ids = set()
        for crop in raw_crops:
            if not isinstance(crop, dict):
                continue
            candidates = []
            crop_path = crop.get('crop_path')
            crop_box = _xyxy(crop.get('detection_bbox') or crop.get('bbox'))
            for person in correlated_people:
                det_id = person['detection_id']
                if det_id in used_detection_ids:
                    continue
                # Se o crop é o mesmo, a identidade é conhecida.
                same_path = (crop_path and person.get('crop_path') and
                             Path(str(crop_path)) == Path(str(person['crop_path'])))
                score = 1.0 if same_path else _iou(crop_box, _xyxy(person.get('bbox')))
                candidates.append((score, det_id))
            candidates.sort(reverse=True)
            if not candidates:
                continue
            best_score, det_id = candidates[0]
            second_score = candidates[1][0] if len(candidates) > 1 else 0.0
            if best_score < self.min_iou or (best_score < 1.0 and
                                            best_score - second_score < self.min_margin):
                continue
            add(det_id, crop.get('predictions', []), 'raw_action_crop')
            used_detection_ids.add(det_id)

        observations = []
        for det_id, labels in evidence.items():
            person = by_detection[det_id]
            predictions = [{'label': k, 'score': round(v, 4)}
                           for k, v in sorted(labels.items(), key=lambda x: -x[1])]
            observations.append({
                'detection_id': det_id,
                'entity_id': person['entity_id'],
                'track_id': person['track_id'],
                'source': source[det_id],
                'predictions': predictions,
            })
        return observations
