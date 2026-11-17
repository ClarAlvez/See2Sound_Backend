"""Memoria de vetores de atributos por submodelo, por identidade e por cena.

Cada espaco e separado (person, object, action, scene, atmosphere e relation).
Nao mistura ResNet512 (identidade) com probabilidades de labels; a galeria de
Re-ID continua sob responsabilidade de EntityMemory.

Scores ausentes NAO sao negativos: o vetor fornece observed_mask e coverage.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict
import math


@dataclass
class _VectorRecord:
    by_frame: Dict[int, dict] = field(default_factory=dict)
    timestamps: Dict[int, float] = field(default_factory=dict)


class AttributeVectorMemory:
    DOMAINS = ('person', 'object', 'action', 'scene', 'atmosphere', 'relation')

    def __init__(self, alpha=0.35, max_frame_records=10000):
        self.alpha = max(0.0, min(1.0, float(alpha)))
        self.max_frame_records = max(1, int(max_frame_records))
        self._records = {}  # (domain, entity/scene id) => _VectorRecord
        self._schemas: dict[str, set[str]] = {domain: set() for domain in self.DOMAINS}
        self._bootstrap_schemas()

    def _bootstrap_schemas(self):
        # Importacoes opcionais: nao forca carregar pesos nem instanciar IAs.
        imports = {
            'person': ('ai.spectra.Person.labels', 'LABELS'),
            'action': ('ai.spectra.Actions.labels', 'LABELS'),
            'scene': ('ai.spectra.Scene.labels', 'LABELS'),
            'atmosphere': ('ai.spectra.Atmosphere.labels', 'LABELS'),
        }
        from importlib import import_module
        for domain, (module, attr) in imports.items():
            try:
                labels = getattr(import_module(module), attr)
                self._schemas[domain].update(map(str, labels))
            except (ImportError, AttributeError):
                pass

    def update(self, domain: str, owner_id: str, frame_id: int, timestamp: float, attributes: dict):
        if domain not in self._schemas:
            raise ValueError(f'Dominio de atributos desconhecido: {domain}')
        if not owner_id or not isinstance(attributes, dict) or not attributes:
            return
        clean = {}
        for label, value in attributes.items():
            if not label:
                continue
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score):
                continue
            clean[str(label)] = max(0.0, min(1.0, score))
        if not clean:
            return
        self._schemas[domain].update(clean)
        key = (domain, str(owner_id))
        record = self._records.setdefault(key, _VectorRecord())
        fid = int(frame_id)
        if fid not in record.by_frame and len(record.by_frame) >= self.max_frame_records:
            oldest = min(record.by_frame)
            record.by_frame.pop(oldest, None)
            record.timestamps.pop(oldest, None)
        # Redeteccoes no mesmo frame contam uma unica vez; score maior e preservado.
        previous = record.by_frame.setdefault(fid, {})
        for label, score in clean.items():
            previous[label] = max(previous.get(label, 0.0), score)
        record.timestamps[fid] = float(timestamp)

    def remap_person_ids(self, remap: dict):
        if not remap:
            return
        result = {}
        for (domain, owner_id), record in self._records.items():
            if domain not in ('person', 'action', 'relation'):
                result[(domain, owner_id)] = record
                continue
            new_owner = owner_id
            visited = set()
            while new_owner in remap and new_owner not in visited:
                visited.add(new_owner)
                new_owner = str(remap[new_owner])
            target = result.setdefault((domain, new_owner), _VectorRecord())
            for fid, vector in record.by_frame.items():
                old = target.by_frame.setdefault(fid, {})
                for label, score in vector.items():
                    old[label] = max(score, old.get(label, 0.0))
                target.timestamps[fid] = max(target.timestamps.get(fid, 0.0), record.timestamps.get(fid, 0.0))
        self._records = result

    def _summary(self, domain: str, owner_id: str, record: _VectorRecord):
        labels = sorted(self._schemas[domain])
        samples = [record.by_frame[k] for k in sorted(record.by_frame)]
        observation_count = len(samples)
        totals = defaultdict(float)
        counts = defaultdict(int)
        maximum = defaultdict(float)
        ema = {}
        for attrs in samples:
            for label, score in attrs.items():
                totals[label] += score
                counts[label] += 1
                maximum[label] = max(score, maximum[label])
                ema[label] = score if label not in ema else (self.alpha * score + (1.0-self.alpha) * ema[label])
        return {
            'owner_id': owner_id,
            'domain': domain,
            'labels': labels,
            'values': [round(totals[label]/counts[label], 4) if counts[label] else 0.0 for label in labels],
            'observed_mask': [int(counts[label] > 0) for label in labels],
            'coverage': [round(counts[label]/observation_count, 4) if observation_count else 0.0 for label in labels],
            'ema_values': [round(ema.get(label, 0.0), 4) for label in labels],
            'max_values': [round(maximum.get(label, 0.0), 4) for label in labels],
            'observation_count': observation_count,
            'first_seen': min(record.timestamps.values()) if record.timestamps else None,
            'last_seen': max(record.timestamps.values()) if record.timestamps else None,
            'frame_ids': sorted(record.by_frame),
        }

    def summary(self, domain: str, owner_id: str):
        record = self._records.get((domain, str(owner_id)))
        return self._summary(domain, str(owner_id), record) if record else None

    def to_dict(self):
        by_domain = {domain: {} for domain in self.DOMAINS}
        for (domain, owner), record in sorted(self._records.items()):
            by_domain[domain][owner] = self._summary(domain, owner, record)
        return by_domain
