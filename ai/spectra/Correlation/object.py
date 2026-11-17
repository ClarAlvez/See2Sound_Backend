"""Utilidades comuns aos requisitores de saida dos submodelos Spectra.

Os requisitores NAO executam inferencia: consomem resultados ja calculados.
"""
from __future__ import annotations

import math
from typing import Any


def clamp(value: Any, default=0.0) -> float:
    try:
        result = float(value)
        if not math.isfinite(result):
            return default
        return max(0.0, min(1.0, result))
    except (TypeError, ValueError):
        return default


def xyxy(value: Any):
    if isinstance(value, dict):
        value = [value.get(k) for k in ('x1', 'y1', 'x2', 'y2')]
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        return None
    try:
        box = tuple(float(x) for x in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(x) for x in box) or box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def predictions(value: Any, minimum=0.0) -> dict[str, float]:
    """Normaliza predictions/top_predictions/label->score, maior score vence."""
    if isinstance(value, dict):
        if 'predictions' in value or 'top_predictions' in value:
            value = value.get('predictions', value.get('top_predictions', []))
        elif all(isinstance(k, str) for k in value):
            value = [{'label': label, 'score': score} for label, score in value.items()]
    result = {}
    if not isinstance(value, (list, tuple)):
        return result
    for item in value:
        if not isinstance(item, dict):
            continue
        label = str(item.get('label') or '').strip().lower()
        if not label:
            continue
        score = clamp(item.get('score'))
        if score >= minimum:
            result[label] = max(score, result.get(label, 0.0))
    return result


def as_predictions(vector: dict[str, float]) -> list[dict]:
    return [{'label': label, 'score': round(score, 4)}
            for label, score in sorted(vector.items(), key=lambda kv: -kv[1])]


def raw(frame: dict, task: str) -> dict:
    value = (frame.get('raw') or {}).get(task)
    return value if isinstance(value, dict) else {}


def iou(a, b) -> float:
    a, b = xyxy(a), xyxy(b)
    if a is None or b is None:
        return 0.0
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right-left) * max(0, bottom-top)
    a_area = (a[2]-a[0]) * (a[3]-a[1])
    b_area = (b[2]-b[0]) * (b[3]-b[1])
    union = a_area+b_area-intersection
    return intersection/union if union > 0 else 0.0
