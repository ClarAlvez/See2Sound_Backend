"""Recebe as probabilidades por frame de SceneNet (incluindo derivadas)."""
from __future__ import annotations
from .common import raw, predictions


class SceneRequester:
    def request(self, frame: dict) -> dict:
        source = frame.get('scene_predictions')
        if source is None:
            source = frame.get('scene')
        if source is None:
            source = raw(frame, 'scene')
        direct = predictions(source)
        derived = predictions(source.get('derived_predictions', [])) if isinstance(source, dict) else {}
        return {'attributes': direct, 'derived_attributes': derived, 'available': bool(direct or derived)}
