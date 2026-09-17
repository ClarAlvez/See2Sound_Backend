from pathlib import Path
from typing import Any, Dict, List, Optional

from ai.spectra.Object.inference import ObjectPredictor
from ai.spectra.Object.object_cropper import ObjectCropper


def merge_object_predictions(
    prediction_groups: List[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    best_by_label = {}

    for predictions in prediction_groups:
        for prediction in predictions:
            label = prediction.get("label")

            if not label:
                continue

            score = float(prediction.get("score", 0.0))
            current = best_by_label.get(label)

            if current is None or score > float(current.get("score", 0.0)):
                best_by_label[label] = {
                    "label": label,
                    "score": round(score, 4),
                }

    merged = list(best_by_label.values())
    merged.sort(key=lambda item: item["score"], reverse=True)

    return merged


class ObjectAnalyzer:
    def __init__(
        self,
        object_model_path: str,
        object_threshold: float = 0.35,
        object_top_k: Optional[int] = 20,
        cropper_model_name: str = "yolov8n.pt",
        cropper_confidence_threshold: float = 0.25,
        max_objects: int = 20,
        use_full_frame: bool = True,
        device: Optional[str] = None,
    ):
        self.object_model_path = object_model_path
        self.object_threshold = object_threshold
        self.object_top_k = object_top_k
        self.use_full_frame = use_full_frame

        self.predictor = ObjectPredictor(
            model_path=object_model_path,
            threshold=object_threshold,
            top_k=object_top_k,
            device=device,
        )

        self.cropper = ObjectCropper(
            model_name=cropper_model_name,
            confidence_threshold=cropper_confidence_threshold,
            max_objects=max_objects,
            device=device,
        )

    def analyze_frame(
        self,
        image_path: str,
        crops_output_dir: str,
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        image_path = Path(image_path)
        crops_output_dir = Path(crops_output_dir)

        cutoff = self.object_threshold if threshold is None else threshold
        limit = self.object_top_k if top_k is None else top_k

        prediction_groups = []
        crop_results = []

        if self.use_full_frame:
            full_frame_result = self.predictor.predict_frame(
                image_path=str(image_path),
                threshold=cutoff,
                top_k=limit,
                group_by_category=False,
            )
            prediction_groups.append(full_frame_result.get("predictions", []))

        crop_results = self.cropper.crop_objects(
            image_path=str(image_path),
            output_dir=str(crops_output_dir),
        )

        enriched_crop_results = []

        for crop_result in crop_results:
            crop_path = crop_result["crop_path"]

            crop_prediction_result = self.predictor.predict_frame(
                image_path=crop_path,
                threshold=cutoff,
                top_k=limit,
                group_by_category=False,
            )

            crop_predictions = crop_prediction_result.get("predictions", [])

            # Evidência do detector também entra como pista fraca/auxiliar.
            detector_predictions = [
                {
                    "label": label,
                    "score": float(crop_result.get("detector_score", 0.0)),
                }
                for label in crop_result.get("mapped_labels", [])
            ]

            prediction_groups.append(crop_predictions)
            prediction_groups.append(detector_predictions)

            enriched = dict(crop_result)
            enriched["predictions"] = crop_predictions
            enriched_crop_results.append(enriched)

        merged_predictions = merge_object_predictions(prediction_groups)

        if limit is not None:
            merged_predictions = merged_predictions[:limit]

        return {
            "frame_path": str(image_path),
            "task_name": "object",
            "source": "object_crops",
            "threshold": cutoff,
            "crops_detected": len(crop_results),
            "predictions": merged_predictions,
            "crop_results": enriched_crop_results,
            "grouped_predictions": {
                "scene": [],
                "person": [],
                "object": merged_predictions,
                "action": [],
            },
        }