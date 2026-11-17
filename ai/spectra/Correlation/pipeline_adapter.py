"""Converte o formato real de pipeline/orchestration/process_video.py em frames.

Entrada: build_spectra_outputs_for_scenes(...) -> scene_outputs[] contendo
raw_frame_results[] com outputs das cinco IAs.
"""
from __future__ import annotations
from typing import Any


def flatten_spectra_outputs(scene_outputs: list[dict[str, Any]]) -> list[dict]:
    candidates = []
    for ordinal, scene in enumerate(scene_outputs or []):
        interval = scene.get('context') or {}
        scene_index = interval.get('scene_index', scene.get('scene_index', ordinal))
        scene_id = f'scene_{int(scene_index):04d}' if str(scene_index).isdigit() else str(scene_index)
        start = float(scene.get('start_time', 0.0))
        end = float(scene.get('end_time', float('inf')))
        for data in scene.get('raw_frame_results') or []:
            timestamp = float(data.get('timestamp', start))
            frame_path = str(data.get('frame_path', ''))
            # Evita atribuir um frame duplicado a outra cena se nao pertence ao intervalo.
            belongs = start <= timestamp <= end
            candidates.append({
                'timestamp': timestamp, 'frame_path': frame_path,
                'scene_id': scene_id, 'raw': data.get('raw') or {},
                'belongs_to_interval': belongs,
            })
    candidates.sort(key=lambda x: (x['timestamp'], not x['belongs_to_interval'], x['scene_id']))
    unique = {}
    for frame in candidates:
        key = (frame['frame_path'], frame['timestamp'])
        unique.setdefault(key, frame)
    frames = sorted(unique.values(), key=lambda x: (x['timestamp'], x['frame_path']))
    for index, frame in enumerate(frames):
        frame['frame_id'] = index
        frame.pop('belongs_to_interval', None)
    return frames


def from_pipeline_output(value: dict) -> list[dict]:
    """Aceita JSON final: {'spectra': {'outputs': scene_outputs}}."""
    if not isinstance(value, dict):
        return []
    if 'spectra' in value:
        return flatten_spectra_outputs((value.get('spectra') or {}).get('outputs', []))
    if 'raw_frame_results' in value:
        return flatten_spectra_outputs([value])
    if 'frames' in value and isinstance(value['frames'], list):
        return value['frames']
    return []
