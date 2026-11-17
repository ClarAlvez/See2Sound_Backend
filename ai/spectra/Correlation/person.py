"""Entrada ObjectAnalyzer: distingue detector YOLO e classificador ObjectNet."""
from __future__ import annotations
from .common import xyxy, predictions, clamp, raw


class ObjectRequester:
    def __init__(self, prefer_specific=True):
        self.prefer_specific = prefer_specific
        self.generic_labels = {'bag', 'food', 'fruit', 'animal', 'screen', 'toy'}

    def request(self, frame: dict) -> list[dict]:
        source = frame.get('objects')
        if source is None:
            source = raw(frame, 'object').get('crop_results', [])
        observations = []
        for index, item in enumerate(source or []):
            if not isinstance(item, dict):
                continue
            box = xyxy(item.get('detection_bbox') or item.get('bbox') or item.get('bbox_xyxy'))
            if box is None:
                continue
            cls = predictions(item.get('predictions') or item.get('top_predictions'))
            detector_label = str(item.get('detector_label') or '').lower()
            detector_score = clamp(item.get('detector_confidence', item.get('detector_score', 0.0)))
            mapped_labels = [str(s).lower() for s in item.get('mapped_labels', [])]
            label = str(item.get('label') or '').strip().lower()
            if not label:
                candidates = cls
                if self.prefer_specific:
                    specific = {k: v for k, v in cls.items() if k not in self.generic_labels}
                    candidates = specific or cls
                if candidates:
                    label = max(candidates, key=candidates.get)
                elif mapped_labels:
                    label = mapped_labels[0]
                else:
                    label = detector_label
            if not label:
                continue
            source_name = 'classifier' if label in cls else 'detector_fallback'
            label_score = clamp(item.get('label_score', cls.get(label, detector_score)))
            # Classificador e detector sao duas evidencias; nao inventa confianca conjunta.
            out = dict(item)
            out.update({
                'detection_id': str(item.get('detection_id') or f"frame_{int(frame.get('frame_id', 0)):06d}_object_{index:02d}"),
                'bbox': list(box), 'label': label, 'label_score': label_score,
                'detector_label': detector_label, 'detector_confidence': detector_score,
                'classification_source': source_name,
                'object_attributes': cls or {label: label_score},
            })
            observations.append(out)
        return observations
