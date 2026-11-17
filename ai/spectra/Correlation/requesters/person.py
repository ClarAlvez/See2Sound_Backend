"""PersonNet/PersonCropper: aceita crops diretos e o formato raw_crop_results da pipeline."""
from __future__ import annotations
from .common import xyxy, predictions, raw


class PersonRequester:
    def request(self, frame: dict) -> list[dict]:
        supplied = frame.get('people')
        if supplied is None:
            supplied = frame.get('persons')
        if supplied is None:
            result = raw(frame, 'person')
            supplied = (result.get('raw_crop_results')
                        or result.get('crop_results')
                        or (frame.get('raw') or {}).get('person_crops')
                        or [])
        people = []
        for index, original in enumerate(supplied or []):
            if not isinstance(original, dict):
                continue
            # Saida da pipeline: {'crop': {bbox, crop_path}, 'result': {predictions}}.
            crop = original.get('crop', original)
            if not isinstance(crop, dict):
                continue
            model_result = original.get('result') if isinstance(original.get('result'), dict) else {}
            bbox = xyxy(crop.get('detection_bbox') or crop.get('bbox') or crop.get('bbox_xyxy'))
            if bbox is None:
                continue
            attrs = original.get('attributes') or crop.get('attributes')
            if not isinstance(attrs, dict):
                attrs = predictions(model_result or original.get('predictions') or original.get('top_predictions'))
            out = dict(original)
            out.update({
                'detection_id': str(original.get('detection_id') or
                       f"frame_{int(frame.get('frame_id', 0)):06d}_person_{index:02d}"),
                'bbox': list(bbox),
                'crop_path': crop.get('crop_path'),
                'detector_confidence': crop.get('detector_confidence', crop.get('confidence')),
                'attributes': predictions(attrs),
            })
            out['attributes'].setdefault('person', 1.0)
            people.append(out)
        return people
