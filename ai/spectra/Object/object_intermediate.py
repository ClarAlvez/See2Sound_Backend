"""Object Intermediate v2 - propostas, supressão narrativa e fusão auditável.

Não interpreta score CLIP como probabilidade. Negação em CLIP é fraca;
remoções automáticas ficam desabilitadas por padrão.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import median
from typing import Any

MODEL_NAME = 'openai/clip-vit-base-patch32'

PROMPTS = {
    'animal': ('a photo of a visible animal', 'a photo of a scene with no animals'),
    'dog': ('a photo of a dog', 'a photo of a scene without dogs'),
    'cat': ('a photo of a cat', 'a photo of a scene without cats'),
    'car': ('a photo of a car', 'a photo of a scene without cars'),
    'truck': ('a photo of a truck', 'a photo of a scene without trucks'),
    'bus': ('a photo of a bus', 'a photo of a scene without buses'),
    'bicycle': ('a photo of a bicycle', 'a photo of a scene without bicycles'),
    'motorcycle': ('a photo of a motorcycle', 'a photo of a scene without motorcycles'),
    'computer': ('a photo of a computer', 'a photo of a scene without computers'),
    'screen': ('a photo of an electronic screen', 'a photo without electronic screens'),
    'knife': ('a photo of a kitchen knife', 'a photo without knives'),
    'bowl': ('a photo of a bowl', 'a photo without bowls'),
    'food': ('a photo of prepared food', 'a photo without prepared food'),
    'bag': ('a photo of a bag', 'a photo without bags'),
    'table': ('a photo of a table', 'a photo without tables'),
    'chair': ('a photo of a chair', 'a photo without chairs'),
    'bottle': ('a photo of a bottle', 'a photo without bottles'),
    'cup': ('a photo of a cup', 'a photo without cups'),
    'book': ('a photo of a book', 'a photo without books'),
    'phone': ('a photo of a cellphone', 'a photo without cellphones'),
}

# Valores experimentais, ajustar com vídeos positivos e negativos.
YOLO_MIN_SCORE = 0.50
YOLO_MIN_FRAMES = 2
MIN_CLIP_FRAMES = 4
ADD_MIN_POSITIVE_RATIO = 0.60
ADD_MIN_MEDIAN_MARGIN = 0.015
SUPPRESS_MAX_SPECTRA_SCORE = 0.60
SUPPRESS_MAX_MEDIAN_MARGIN = -0.025
SUPPRESS_MIN_NEGATIVE_RATIO = 0.60
REMOVE_MAX_SPECTRA_SCORE = 0.45
REMOVE_MAX_MEDIAN_MARGIN = -0.12
REMOVE_MIN_NEGATIVE_RATIO = 0.85


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


class ObjectCLIPVerifier:
    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.model_name = model_name

    def compare(self, image_path: str, label: str) -> dict:
        from PIL import Image
        if label not in PROMPTS:
            raise KeyError(label)
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as im:
            inputs = self.processor(images=im.convert('RGB'), text=list(PROMPTS[label]),
                                    return_tensors='pt', padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits_per_image[0].float().cpu().tolist()
        return {
            'positive_logit': round(float(logits[0]), 5),
            'negative_logit': round(float(logits[1]), 5),
            'margin': round((float(logits[0]) - float(logits[1])) / 100.0, 5),
        }


def _collect_yolo(scene: dict) -> dict:
    matches = defaultdict(list)
    for frame in scene.get('raw_frame_results', []) or []:
        obj = (frame.get('raw') or {}).get('object') or {}
        for crop in obj.get('crop_results', []) or []:
            score = _num(crop.get('detector_score'))
            if score < YOLO_MIN_SCORE:
                continue
            for label in crop.get('mapped_labels', []) or []:
                if label in PROMPTS:
                    matches[label].append({
                        'timestamp': frame.get('timestamp'),
                        'detector_label': crop.get('detector_label'),
                        'score': round(score, 4),
                    })
    return dict(matches)


def _distinct_frame_count(records: list) -> int:
    return len({str(item.get('timestamp')) for item in records})


def _frames(scene: dict, max_frames: int) -> list:
    frames = [f for f in scene.get('raw_frame_results', []) or []
              if f.get('frame_path') and Path(f['frame_path']).is_file()]
    if len(frames) > max_frames:
        if max_frames <= 1:
            return frames[:1]
        ids = sorted({round(i * (len(frames) - 1) / (max_frames - 1))
                      for i in range(max_frames)})
        frames = [frames[i] for i in ids]
    return frames


def analyze_object_scene(scene: dict, verifier: ObjectCLIPVerifier | None = None,
                         max_frames: int = 6, include_suggestions: bool = True) -> dict:
    """Inspeciona somente Object. Não modifica a cena."""
    original_labels = set(scene.get('labels') or [])
    confidence = scene.get('confidence') or {}
    existing = {label for label in PROMPTS if label in original_labels
                and f'object.{label}' in confidence}
    yolo = _collect_yolo(scene)
    # Verificar classes novas somente quando houve detecção física YOLO.
    suggestions = ({label for label, hits in yolo.items()
                    if _distinct_frame_count(hits) >= YOLO_MIN_FRAMES}
                   if include_suggestions else set())
    candidate_labels = sorted(existing | suggestions)
    frames = _frames(scene, max_frames)
    decisions = []

    for label in candidate_labels:
        present = label in existing
        spectra_score = (_num(confidence.get(f'object.{label}')) if present else None)
        yolo_hits = yolo.get(label, [])
        yolo_frames = _distinct_frame_count(yolo_hits)
        votes = []
        if verifier is not None:
            for frame in frames:
                try:
                    vote = verifier.compare(str(frame['frame_path']), label)
                    vote['timestamp'] = frame.get('timestamp')
                    votes.append(vote)
                except (OSError, ValueError) as exc:
                    votes.append({'timestamp': frame.get('timestamp'), 'error': str(exc)})
        valid = [v for v in votes if 'margin' in v]
        margins = [v['margin'] for v in valid]
        med = median(margins) if margins else 0.0
        positive_ratio = (sum(m >= ADD_MIN_MEDIAN_MARGIN for m in margins) / len(margins)
                          if margins else 0.0)
        negative_ratio = (sum(m <= SUPPRESS_MAX_MEDIAN_MARGIN for m in margins) / len(margins)
                          if margins else 0.0)
        clip_ready = len(valid) >= MIN_CLIP_FRAMES
        yolo_positive = yolo_frames >= YOLO_MIN_FRAMES
        clip_positive = (clip_ready and positive_ratio >= ADD_MIN_POSITIVE_RATIO
                         and med >= ADD_MIN_MEDIAN_MARGIN)
        if not present:
            if yolo_positive and clip_positive:
                decision, reason = 'add', 'multi_frame_yolo_plus_clip'
            else:
                decision, reason = 'uncertain', 'new_label_not_corroborated'
        elif yolo_positive or clip_positive:
            decision, reason = 'keep', 'independent_positive_evidence'
        elif (clip_ready and not yolo_hits and spectra_score <= REMOVE_MAX_SPECTRA_SCORE
              and med <= REMOVE_MAX_MEDIAN_MARGIN and negative_ratio >= REMOVE_MIN_NEGATIVE_RATIO):
            decision, reason = 'remove', 'strong_experimental_negative_evidence'
        elif (clip_ready and not yolo_hits and spectra_score <= SUPPRESS_MAX_SPECTRA_SCORE
              and med <= SUPPRESS_MAX_MEDIAN_MARGIN
              and negative_ratio >= SUPPRESS_MIN_NEGATIVE_RATIO):
            decision, reason = 'suppress_narrative', 'weak_unconfirmed_object'
        else:
            decision, reason = 'uncertain', 'insufficient_independent_evidence'
        decisions.append({
            'label': label,
            'spectra_score': round(spectra_score, 4) if present else None,
            'decision': decision,
            'reason': reason,
            'yolo_detections': yolo_hits,
            'yolo_frames': yolo_frames,
            'clip': {
                'frames': valid, 'valid_count': len(valid),
                'positive_ratio': round(positive_ratio, 4),
                'negative_ratio': round(negative_ratio, 4),
                'median_margin': round(med, 5),
            },
        })
    return {
        'status': 'analyzed' if verifier is not None else 'offline_yolo_only',
        'model': verifier.model_name if verifier is not None else None,
        'frames_examined': len(frames),
        'decisions': decisions,
        'unsupported_labels': sorted({label for label in original_labels
                                       if f'object.{label}' in confidence and label not in PROMPTS}),
        'raw_preserved': True,
        'warning': 'Scores CLIP não são calibrados; prompts negativos não comprovam ausência.',
    }


def apply_object_intermediate(scene: dict, verifier: ObjectCLIPVerifier | None = None,
                              apply_removals: bool = False, apply_additions: bool = False,
                              suppress_narrative: bool = True) -> dict:
    """Preserva original, aplica somente objetos, mantém auditoria."""
    result = deepcopy(scene)
    report = analyze_object_scene(scene, verifier)
    proposed_additions = [x['label'] for x in report['decisions'] if x['decision'] == 'add']
    proposed_removals = [x['label'] for x in report['decisions'] if x['decision'] == 'remove']
    proposed_suppression = [x['label'] for x in report['decisions']
                            if x['decision'] in {'remove', 'suppress_narrative'}]
    applied_additions = proposed_additions if apply_additions else []
    applied_removals = proposed_removals if apply_removals else []
    applied_suppression = proposed_suppression if suppress_narrative else []

    report.update({
        'mode': 'fusion_v2',
        'proposed_additions': proposed_additions,
        'proposed_removals': proposed_removals,
        'proposed_suppression': proposed_suppression,
        'applied_additions': applied_additions,
        'applied_removals': applied_removals,
        'narrative_suppression': applied_suppression,
    })
    result['object_original'] = {
        'labels': deepcopy(scene.get('labels', [])),
        'confidence': deepcopy(scene.get('confidence', {})),
        'spectra_tasks_object': deepcopy((scene.get('context') or {}).get('spectra_tasks', {}).get('object', [])),
    }
    result['object_intermediate'] = report
    context = result.setdefault('context', {})
    context['object_intermediate'] = report
    context['suppressed_object_labels'] = applied_suppression

    conf = result.setdefault('confidence', {})
    for label in applied_removals:
        conf.pop(f'object.{label}', None)
    # Never represent an invented CLIP score as a calibrated Spectra confidence.
    for label in applied_additions:
        # Score comes from YOLO detections, not from CLIP probabilities.
        choice = next(x for x in report['decisions'] if x['label'] == label)
        detector_score = max(x['score'] for x in choice['yolo_detections'])
        conf[f'object.{label}'] = detector_score

    # Preserve labels referenced by other tasks (homonyms).
    task_labels = {k.split('.', 1)[1] for k in conf if '.' in k}
    result['labels'] = [x for x in result.get('labels', [])
                        if x not in applied_removals or x in task_labels]
    for label in applied_additions:
        if label not in result['labels']:
            result['labels'].append(label)

    tasks = context.get('spectra_tasks') or {}
    if 'object' in tasks and isinstance(tasks['object'], list):
        tasks['object'] = [x for x in tasks['object'] if x.get('label') not in applied_removals]
        for label in applied_additions:
            if not any(x.get('label') == label for x in tasks['object']):
                tasks['object'].append({'label': label, 'score': conf[f'object.{label}'],
                                        'source': 'yolo_clip_intermediate', 'calibrated': False})
    merged = context.get('merged_label_scores')
    if isinstance(merged, dict):
        for label in applied_removals:
            if label not in task_labels:
                merged.pop(label, None)
        for label in applied_additions:
            merged[label] = conf[f'object.{label}']
    return result


def scene_for_narrative(scene: dict) -> dict:
    """Cópia sem objetos suprimidos, apenas para construir a narração."""
    result = deepcopy(scene)
    hidden = set((result.get('context') or {}).get('suppressed_object_labels', []))
    if not hidden:
        return result
    confidence = result.get('confidence') or {}
    for label in hidden:
        confidence.pop(f'object.{label}', None)
    other_labels = {k.split('.', 1)[1] for k in confidence if '.' in k}
    result['labels'] = [x for x in result.get('labels', [])
                        if x not in hidden or x in other_labels]
    context = result.get('context') or {}
    tasks = context.get('spectra_tasks') or {}
    if isinstance(tasks.get('object'), list):
        tasks['object'] = [x for x in tasks['object'] if x.get('label') not in hidden]
    return result
