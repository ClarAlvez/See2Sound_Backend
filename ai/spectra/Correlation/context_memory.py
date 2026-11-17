"""Scene/Atmosphere: memoria temporal por scene_id sem inventar identidade visual."""
from __future__ import annotations
from collections import defaultdict

SCENE_EXCLUSIVE = [('indoor', 'outdoor')]
ATMOSPHERE_EXCLUSIVE = [('day', 'night', 'dawn_dusk'), ('bright', 'low_light')]


class ContextMemory:
    def __init__(self, domain: str, min_observations: int = 2, min_mean_score: float = 0.50):
        if domain not in ('scene', 'atmosphere'):
            raise ValueError(domain)
        self.domain = domain
        self.min_observations = max(1, int(min_observations))
        self.min_mean_score = float(min_mean_score)
        self._records = defaultdict(lambda: {'first_seen': None, 'last_seen': None, 'frame_vectors': {}})

    def observe(self, scene_id: str, frame_id: int, timestamp: float, attributes: dict):
        if not attributes:
            return
        record = self._records[str(scene_id)]
        record['first_seen'] = timestamp if record['first_seen'] is None else min(record['first_seen'], timestamp)
        record['last_seen'] = timestamp if record['last_seen'] is None else max(record['last_seen'], timestamp)
        record['frame_vectors'][int(frame_id)] = dict(attributes)

    def to_dict(self):
        output = {}
        for scene_id, record in sorted(self._records.items()):
            sums, counts = defaultdict(float), defaultdict(int)
            for vector in record['frame_vectors'].values():
                for label, score in vector.items():
                    sums[label] += float(score)
                    counts[label] += 1
            # Media entre observacoes efetivas da label, com coverage explicito.
            means = {k: sums[k] / counts[k] for k in sums}
            confirmed = {
                k: round(v, 4) for k, v in means.items()
                if counts[k] >= self.min_observations and v >= self.min_mean_score
            }
            exclusive_groups = SCENE_EXCLUSIVE if self.domain == 'scene' else ATMOSPHERE_EXCLUSIVE
            for group in exclusive_groups:
                active = [k for k in group if k in confirmed]
                if len(active) > 1:
                    winner = max(active, key=lambda k: confirmed[k])
                    for k in active:
                        if k != winner:
                            confirmed.pop(k, None)
            output[scene_id] = {
                'scene_id': scene_id,
                'domain': self.domain,
                'first_seen': record['first_seen'],
                'last_seen': record['last_seen'],
                'observation_count': len(record['frame_vectors']),
                'frame_ids': sorted(record['frame_vectors']),
                'attributes': {k: round(v, 4) for k, v in sorted(means.items())},
                'confirmed_attributes': confirmed,
                'attribute_coverage': {k: round(counts[k]/len(record['frame_vectors']), 4)
                                       for k in counts},
            }
        return output
