"""Recebe as probabilidades por frame de AtmosphereNet."""
from __future__ import annotations
from .common import raw, predictions


class AtmosphereRequester:
    def request(self, frame: dict) -> dict:
        source = frame.get('atmosphere_predictions')
        if source is None:
            source = frame.get('atmosphere')
        if source is None:
            source = raw(frame, 'atmosphere')
        attrs = predictions(source)
        return {'attributes': attrs, 'available': bool(attrs)}
