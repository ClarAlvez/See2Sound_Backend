from pathlib import Path
from typing import Any, Dict, List, Optional

from ai.spectra.Actions.inference import ActionPredictor
from ai.spectra.Person.person_cropper import PersonCropper


def normalize_crop_result(crop_result: Any) -> List[Dict[str, Any]]:
    """
    Normaliza diferentes formatos possíveis retornados pelo PersonCropper.

    Formatos aceitos:
    - lista de paths
    - lista de dicts com crop_path/image_path/path
    - dict com chave crops/person_crops
    """

    if crop_result is None:
        return []

    if isinstance(crop_result, dict):
        if "crops" in crop_result:
            crop_result = crop_result["crops"]
        elif "person_crops" in crop_result:
            crop_result = crop_result["person_crops"]
        else:
            crop_result = [crop_result]

    normalized = []

    for item in crop_result:
        if isinstance(item, (str, Path)):
            normalized.append(
                {
                    "crop_path": str(item),
                    "bbox": None,
                    "detector_confidence": None,
                }
            )
            continue

        if isinstance(item, dict):
            crop_path = (
                item.get("crop_path")
                or item.get("image_path")
                or item.get("path")
                or item.get("file_path")
            )

            if crop_path is None:
                continue

            normalized.append(
                {
                    "crop_path": str(crop_path),
                    "bbox": item.get("bbox"),
                    "detector_confidence": item.get("confidence")
                    or item.get("detector_confidence"),
                }
            )

    return normalized


def merge_action_crop_predictions(
    crop_results: List[Dict[str, Any]],
    threshold: float = 0.3,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Junta previsões de vários crops.

    Se duas pessoas gerarem a mesma ação, mantém o maior score.
    """

    best_by_label = {}

    for crop_result in crop_results:
        person_index = crop_result.get("person_index")
        crop_path = crop_result.get("crop_path")

        for prediction in crop_result.get("predictions", []):
            label = prediction["label"]
            score = float(prediction["score"])

            if score < threshold:
                continue

            current = best_by_label.get(label)

            if current is None or score > current["score"]:
                best_by_label[label] = {
                    "label": label,
                    "score": round(score, 4),
                    "source": "person_crop",
                    "person_index": person_index,
                    "crop_path": crop_path,
                }

    merged = list(best_by_label.values())

    merged.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    if top_k is not None:
        merged = merged[:top_k]

    return merged


class PersonActionAnalyzer:
    def __init__(
        self,
        action_model_path: str,
        action_threshold: float = 0.3,
        action_top_k: Optional[int] = 10,
        person_cropper_model_name: str = "yolov8n.pt",
        person_cropper_confidence_threshold: float = 0.35,
        max_people: int = 5,
        device: Optional[str] = None,
    ):
        self.action_predictor = ActionPredictor(
            model_path=action_model_path,
            threshold=action_threshold,
            top_k=action_top_k,
            device=device,
        )

        self.person_cropper = PersonCropper(
            model_name=person_cropper_model_name,
            confidence_threshold=person_cropper_confidence_threshold,
        )

        self.action_threshold = action_threshold
        self.action_top_k = action_top_k
        self.max_people = max_people

    def analyze_frame(
        self,
        image_path: str,
        crops_output_dir: str,
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        image_path = Path(image_path)
        crops_output_dir = Path(crops_output_dir)
        crops_output_dir.mkdir(parents=True, exist_ok=True)

        cutoff = self.action_threshold if threshold is None else threshold
        limit = self.action_top_k if top_k is None else top_k

        raw_crops = self.person_cropper.crop_people(
            image_path=str(image_path),
            output_dir=str(crops_output_dir),
            max_people=self.max_people,
        )

        crops = normalize_crop_result(raw_crops)

        crop_results = []

        for person_index, crop in enumerate(crops):
            crop_path = crop["crop_path"]

            if not Path(crop_path).exists():
                continue

            action_result = self.action_predictor.predict_frame(
                image_path=crop_path,
                threshold=cutoff,
                top_k=limit,
                group_by_category=False,
            )

            crop_results.append(
                {
                    "person_index": person_index,
                    "crop_path": crop_path,
                    "bbox": crop.get("bbox"),
                    "detector_confidence": crop.get("detector_confidence"),
                    "predictions": action_result.get("predictions", []),
                }
            )

        merged_predictions = merge_action_crop_predictions(
            crop_results=crop_results,
            threshold=cutoff,
            top_k=limit,
        )

        return {
            "frame_path": str(image_path),
            "task_name": "action",
            "source": "person_crops",
            "threshold": cutoff,
            "people_detected": len(crops),
            "crops_analyzed": len(crop_results),
            "predictions": merged_predictions,
            "crop_results": crop_results,
            "grouped_predictions": {
                "scene": [],
                "person": [],
                "object": [],
                "action": merged_predictions,
            },
        }