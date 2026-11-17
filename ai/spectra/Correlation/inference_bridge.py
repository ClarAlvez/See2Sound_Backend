"""Ponte OPCIONAL de inferencia: executa apenas as IAs ainda ausentes no frame.

O caminho recomendado continua sendo a pipeline produzir 'raw' uma unica vez.
A ponte nao carrega modelos ate serem explicitamente configurados.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional


class SpectraInferenceBridge:
    def __init__(self, *, person_model: Optional[str] = None,
                 action_model: Optional[str] = None, object_model: Optional[str] = None,
                 scene_model: Optional[str] = None, atmosphere_model: Optional[str] = None,
                 yolo_model: str = 'yolov8n.pt', max_people=5, max_objects=20,
                 device: Optional[str] = None, output_dir='data/output/correlation_inference'):
        self.models = dict(person=person_model, action=action_model, object=object_model,
                           scene=scene_model, atmosphere=atmosphere_model)
        self.yolo_model = yolo_model
        self.max_people = max_people
        self.max_objects = max_objects
        self.device = device
        self.output_dir = Path(output_dir)
        self._loaded = {}

    def _get(self, name):
        path = self.models.get(name)
        if not path:
            return None
        if name not in self._loaded:
            if name == 'action':
                from ai.spectra.Actions.inference import ActionPredictor
                model = ActionPredictor(model_path=path, threshold=0.30, top_k=None, device=self.device)
            elif name == 'object':
                from ai.spectra.Object.object_analyzer import ObjectAnalyzer
                model = ObjectAnalyzer(object_model_path=path, max_objects=self.max_objects,
                                       use_full_frame=False)
            else:
                from ai.spectra.predictor import SpectraPredictor
                model = SpectraPredictor(model_path=path, task_name=name,
                                         threshold=0.0, top_k=None, device=self.device)
            self._loaded[name] = model
        return self._loaded[name]

    def _cropper(self):
        if '_cropper' not in self._loaded:
            from ai.spectra.Person.person_cropper import PersonCropper
            self._loaded['_cropper'] = PersonCropper(model_name=self.yolo_model)
        return self._loaded['_cropper']

    def enrich(self, frame: dict) -> dict:
        """Retorna copia do frame com outputs ausentes; nao modifica input."""
        out = dict(frame)
        if isinstance(out.get('people'), list):
            out['people'] = [dict(p) for p in out['people']]
        if isinstance(out.get('persons'), list):
            out['persons'] = [dict(p) for p in out['persons']]
        raw = dict(out.get('raw') or {})
        out['raw'] = raw
        image = out.get('frame_path') or out.get('image_path')
        if not image:
            return out
        image = str(image)
        fid = int(out.get('frame_id', 0))
        # Se a pipeline ja produziu crops de Person, reutiliza-os antes de YOLO.
        existing_person_raw = raw.get('person') or {}
        if (out.get('people') is None and out.get('persons') is None
                and isinstance(existing_person_raw, dict)
                and (existing_person_raw.get('raw_crop_results') or existing_person_raw.get('crop_results'))):
            from ai.spectra.Correlation.requesters.person import PersonRequester
            out['people'] = PersonRequester().request(out)
        # Person e Actions compartilham os mesmos crops YOLO.
        if out.get('people') is None and out.get('persons') is None and (self.models.get('person') or self.models.get('action')):
            crops_dir = self.output_dir / 'people' / f'frame_{fid:06d}'
            crops = self._cropper().crop_people(image_path=image, output_dir=str(crops_dir), max_people=self.max_people)
            out['people'] = [
                {'detection_id': f'frame_{fid:06d}_person_{i:02d}',
                 'bbox': crop.get('detection_bbox', crop.get('bbox')),
                 'crop_path': crop['crop_path'],
                 'detector_confidence': crop.get('detector_confidence')}
                for i, crop in enumerate(crops)
            ]
        people = out.get('people') or out.get('persons') or []
        if self.models.get('person'):
            predictor = self._get('person')
            for item in people:
                if item.get('attributes') or not item.get('crop_path'):
                    continue
                item['attributes'] = {p['label']: p['score'] for p in predictor.predict_frame(
                    image_path=str(item['crop_path']), threshold=0.0, top_k=None).get('predictions', [])}
        if self.models.get('action'):
            predictor = self._get('action')
            for item in people:
                if 'action_predictions' in item or not item.get('crop_path'):
                    continue
                item['action_predictions'] = predictor.predict_frame(
                    image_path=str(item['crop_path']), threshold=0.30, top_k=None).get('predictions', [])
        for name in ('scene', 'atmosphere'):
            if name not in raw and name not in out and f'{name}_predictions' not in out:
                predictor = self._get(name)
                if predictor is not None:
                    raw[name] = predictor.predict_frame(image_path=image, threshold=0.0, top_k=None)
        if 'object' not in raw and out.get('objects') is None:
            analyzer = self._get('object')
            if analyzer is not None:
                folder = self.output_dir / 'objects' / f'frame_{fid:06d}'
                raw['object'] = analyzer.analyze_frame(image_path=image, crops_output_dir=str(folder))
        return out

    def enrich_video(self, frames):
        return [self.enrich(frame) for frame in frames]
