"""Associacao pessoa <-> objeto (v0.3.1 / v0.4).

Score espacial, score de rotulo e evidencia de Actions sao independentes.
As relacoes carries / holds / wears sao HIPOTESES, nao verificacoes 3D.
"""
from __future__ import annotations

from collections import defaultdict
from math import sqrt


def _clamp(value):
    return max(0.0, min(1.0, float(value)))


def _area(box):
    x1, y1, x2, y2 = (float(v) for v in box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _intersection(a, b):
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    return max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1))


def _containment(object_box, person_box):
    size = _area(object_box)
    return _intersection(object_box, person_box) / size if size else 0.0


def _center_inside(object_box, person_box):
    x1, y1, x2, y2 = (float(v) for v in object_box)
    px1, py1, px2, py2 = (float(v) for v in person_box)
    return px1 <= (x1 + x2) / 2 <= px2 and py1 <= (y1 + y2) / 2 <= py2


def _center_similarity(a, b):
    """Sem escala global: distancia normalizada pelo maior bbox (0 a 1)."""
    acx, acy = (float(a[0]) + float(a[2])) / 2, (float(a[1]) + float(a[3])) / 2
    bcx, bcy = (float(b[0]) + float(b[2])) / 2, (float(b[1]) + float(b[3])) / 2
    w = max(float(a[2])-float(a[0]), float(b[2])-float(b[0]), 1.0)
    h = max(float(a[3])-float(a[1]), float(b[3])-float(b[1]), 1.0)
    from math import hypot
    return _clamp(1.0 - hypot((acx-bcx)/w, (acy-bcy)/h))


# A lista abaixo respeita as labels reais de ai/spectra/Actions/labels.py.
# Suporte contextual nao transforma relacao hipotetica em fato.
ACTION_SUPPORT = {
    ('carries', '*'): {'carrying': 1.0},
    ('holds', 'phone'): {'phone_use': 1.0},
    ('holds', 'book'): {'reading': 0.7},
    ('holds', 'document'): {'reading': 0.65, 'writing': 0.5},
    ('holds', 'paper'): {'reading': 0.55, 'writing': 0.55},
    ('holds', 'cup'): {'drinking': 0.7},
    ('holds', 'bottle'): {'drinking': 0.7},
    ('holds', 'instrument'): {'instrument_playing': 1.0},
}


def _supported_action_score(relation, label, action_scores):
    if not isinstance(action_scores, dict):
        return 0.0
    weights = dict(ACTION_SUPPORT.get((relation, '*'), {}))
    weights.update(ACTION_SUPPORT.get((relation, label), {}))
    return _clamp(max((action_scores.get(act, 0.0) * weight
                       for act, weight in weights.items()), default=0.0))


class PersonObjectAssociator:
    def __init__(self, config):
        self.config = config

    def associate(self, people, objects, action_by_entity=None):
        """Retorna melhores relacoes, incluindo aceitas e rejeitadas (debug).

        Memory deve registrar apenas as de accepted=True.
        """
        action_by_entity = action_by_entity or {}
        associations = []
        for object_data in objects:
            candidates = []
            for person in people:
                match = self._score_pair(
                    person, object_data,
                    action_by_entity.get(person['entity_id'], {}),
                )
                if match:
                    candidates.append(match)
            candidates.sort(key=lambda r: r['evidence_score'], reverse=True)
            if not candidates:
                continue

            best = candidates[0]
            second = candidates[1]['evidence_score'] if len(candidates) > 1 else 0.0
            margin = best['evidence_score'] - second
            min_score = float(getattr(self.config, 'object_association_min_score', 0.45))
            min_margin = float(getattr(self.config, 'object_association_min_margin', 0.05))
            min_object = float(getattr(self.config, 'object_min_label_score', 0.45))
            min_evidence = float(getattr(self.config, 'object_relation_min_evidence', 0.55))

            reasons = []
            if best['spatial_score'] < min_score:
                reasons.append('low_spatial_score')
            if best['object_score'] < min_object:
                reasons.append('low_object_score')
            if best['evidence_score'] < min_evidence:
                reasons.append('low_evidence_score')
            if len(candidates) > 1 and margin < min_margin:
                reasons.append('ambiguous_person')
            best['accepted'] = not reasons
            best['rejection_reasons'] = reasons
            associations.append(best)
        return associations

    def _score_pair(self, person, obj, action_scores):
        person_box = person.get('bbox')
        object_box = obj.get('bbox')
        if not person_box or not object_box:
            return None
        label = str(obj.get('label') or '').lower()
        containment = _containment(object_box, person_box)
        center = _center_similarity(object_box, person_box)
        inside = _center_inside(object_box, person_box)
        spatial = _clamp(containment * 0.70 + center * 0.30)
        if inside:
            spatial = max(spatial, 0.65)

        wearables = set(getattr(self.config, 'object_wearable_labels', ['glasses']))
        carried = set(getattr(self.config, 'object_carried_labels',
                              ['backpack', 'bag', 'handbag', 'suitcase', 'umbrella']))
        held = set(getattr(self.config, 'object_held_labels',
                           ['phone', 'book', 'paper', 'document', 'bottle', 'cup', 'key']))
        if label in wearables and inside:
            relation = 'wears'
        elif label in carried and (inside or containment >= 0.35):
            relation = 'carries'
        elif label in held and (inside or containment >= 0.35):
            relation = 'holds'
        elif inside or containment >= 0.20:
            relation = 'near'
        else:
            return None

        # Scores baixos de objeto nao devem ser compensados por geometrias excelentes.
        object_score = _clamp(obj.get('label_score', 0.0))
        action_score = _supported_action_score(relation, label, action_scores)
        baseline = sqrt(spatial * object_score)
        # Reforco moderado; nao e uma classificacao de objeto por Actions.
        evidence = _clamp(baseline + 0.15 * action_score * (1.0 - baseline))
        return {
            'entity_id': person['entity_id'],
            'person_track_id': person['track_id'],
            'object_id': obj['object_id'],
            'object_track_id': obj['track_id'],
            'object_label': label,
            'relation': relation,
            'relation_kind': 'spatial_context' if relation == 'near' else 'interaction',
            'spatial_score': round(spatial, 4),
            'object_score': round(object_score, 4),
            'action_score': round(action_score, 4),
            'evidence_score': round(evidence, 4),
            'score': round(evidence, 4),  # compatibilidade com consumidores antigos
            'accepted': False,  # definido depois da competicao entre pessoas
            'rejection_reasons': [],
        }


class ObjectAssociationMemory:
    """Agregacao temporal por (pessoa, classe, relacao).

    Contagem e por frame, evitando que duas deteccoes sobrepostas criem
    persistencia artificial. Associacoes rejeitadas nao entram na memoria.
    """
    def __init__(self, config):
        self.config = config
        self._records = {}

    def register(self, association, frame_id, timestamp):
        if not association.get('accepted', True):
            return
        key = (association['entity_id'], association['object_label'], association['relation'])
        record = self._records.setdefault(key, {
            'entity_id': key[0], 'object_label': key[1], 'relation': key[2],
            'relation_kind': association.get('relation_kind', 'interaction'),
            'first_seen': float(timestamp), 'last_seen': float(timestamp),
            'by_frame': {}, 'object_ids': set(),
        })
        record['first_seen'] = min(record['first_seen'], float(timestamp))
        record['last_seen'] = max(record['last_seen'], float(timestamp))
        record['object_ids'].add(association['object_id'])
        fid = int(frame_id)
        old = record['by_frame'].get(fid)
        if old is None or association['evidence_score'] > old['evidence_score']:
            record['by_frame'][fid] = {
                k: association[k] for k in
                ('spatial_score', 'object_score', 'action_score', 'evidence_score')
            }

    def remap_entities(self, entity_remap):
        if not entity_remap:
            return
        old_records = list(self._records.values())
        self._records = {}
        for record in old_records:
            eid = record['entity_id']
            visited = set()
            while eid in entity_remap and eid not in visited:
                visited.add(eid)
                eid = entity_remap[eid]
            key = (eid, record['object_label'], record['relation'])
            target = self._records.setdefault(key, {
                'entity_id': eid, 'object_label': record['object_label'],
                'relation': record['relation'], 'relation_kind': record['relation_kind'],
                'first_seen': record['first_seen'], 'last_seen': record['last_seen'],
                'by_frame': {}, 'object_ids': set(),
            })
            target['first_seen'] = min(target['first_seen'], record['first_seen'])
            target['last_seen'] = max(target['last_seen'], record['last_seen'])
            target['object_ids'].update(record['object_ids'])
            for fid, observation in record['by_frame'].items():
                previous = target['by_frame'].get(fid)
                if previous is None or observation['evidence_score'] > previous['evidence_score']:
                    target['by_frame'][fid] = dict(observation)

    def to_dict(self):
        grouped = defaultdict(list)
        min_obs = int(getattr(self.config, 'object_relation_min_observations', 2))
        min_mean = float(getattr(self.config, 'object_relation_persistent_min_evidence', 0.60))
        for key, record in sorted(self._records.items()):
            frames = sorted(record['by_frame'])
            vals = [record['by_frame'][f] for f in frames]
            count = len(frames)
            avg = sum(v['evidence_score'] for v in vals)/count if count else 0.0
            best = max((v['evidence_score'] for v in vals), default=0.0)
            item = {
                'entity_id': key[0], 'object_label': key[1], 'relation': key[2],
                'relation_kind': record['relation_kind'],
                'observation_count': count,
                'best_score': round(best, 4),  # agora = evidence_score
                'average_evidence_score': round(avg, 4),
                'best_spatial_score': round(max((v['spatial_score'] for v in vals), default=0), 4),
                'best_object_score': round(max((v['object_score'] for v in vals), default=0), 4),
                'best_action_score': round(max((v['action_score'] for v in vals), default=0), 4),
                'first_seen': round(record['first_seen'], 4),
                'last_seen': round(record['last_seen'], 4),
                'frame_ids': frames,
                'object_ids': sorted(record['object_ids']),
                'persistent': count >= min_obs and avg >= min_mean,
            }
            grouped[key[0]].append(item)
        return dict(grouped)
