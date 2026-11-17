"""Memoria temporal de acoes associadas a EntityIDs persistentes."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict


class ActionMemory:
    def __init__(self, config=None):
        self.config = config
        self._records: Dict[tuple, Dict[str, Any]] = {}

    def register(self, action_observation, frame_id: int, timestamp: float):
        eid = action_observation['entity_id']
        for item in action_observation.get('predictions', []):
            label = str(item['label']).lower()
            score = float(item['score'])
            key = (eid, label)
            record = self._records.setdefault(key, {
                'entity_id': eid, 'label': label, 'first_seen': float(timestamp),
                'last_seen': float(timestamp), 'scores_by_frame': {},
                'track_ids': set(),
            })
            # Duas deteccoes no mesmo frame NAO contam como duas observacoes.
            previous = record['scores_by_frame'].get(int(frame_id), 0)
            record['scores_by_frame'][int(frame_id)] = max(previous, score)
            record['last_seen'] = max(record['last_seen'], float(timestamp))
            if action_observation.get('track_id'):
                record['track_ids'].add(action_observation['track_id'])

    def remap_entities(self, entity_remap):
        new_records = {}
        for (eid, label), record in self._records.items():
            seen = set()
            while eid in entity_remap and eid not in seen:
                seen.add(eid)
                eid = entity_remap[eid]
            key = (eid, label)
            target = new_records.setdefault(key, {
                'entity_id': eid, 'label': label,
                'first_seen': record['first_seen'], 'last_seen': record['last_seen'],
                'scores_by_frame': {}, 'track_ids': set(),
            })
            target['first_seen'] = min(target['first_seen'], record['first_seen'])
            target['last_seen'] = max(target['last_seen'], record['last_seen'])
            target['track_ids'].update(record['track_ids'])
            for fid, score in record['scores_by_frame'].items():
                target['scores_by_frame'][fid] = max(score, target['scores_by_frame'].get(fid, 0))
        self._records = new_records

    def to_dict(self):
        grouped = defaultdict(list)
        min_obs = int(getattr(self.config, 'action_min_observations', 2))
        min_mean = float(getattr(self.config, 'action_min_mean_score', 0.50))
        for (eid, _), r in sorted(self._records.items()):
            scores = list(r['scores_by_frame'].values())
            count = len(scores)
            average = sum(scores) / count if count else 0.0
            grouped[eid].append({
                'label': r['label'], 'observation_count': count,
                'average_score': round(average, 4),
                'max_score': round(max(scores), 4) if scores else 0.0,
                'first_seen': round(r['first_seen'], 4),
                'last_seen': round(r['last_seen'], 4),
                'frame_ids': sorted(r['scores_by_frame']),
                'track_ids': sorted(r['track_ids']),
                'persistent': count >= min_obs and average >= min_mean,
            })
        return dict(grouped)
