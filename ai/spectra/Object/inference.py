from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from ai.spectra.Object.labels import LABELS
from ai.spectra.Object.model import SpectraObjectNet
from ai.spectra.data.transforms import get_test_transforms


class ObjectPredictor:
    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        top_k: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.top_k = top_k
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modelo de objetos não encontrado: {self.model_path}"
            )

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
        )

        self.labels = checkpoint.get("labels", LABELS)
        self.config = checkpoint.get("config", {})

        self.image_size = self.config.get("image_size", 224)
        self.dropout_rate = self.config.get("dropout_rate", 0.3)
        self.backbone_name = self.config.get("backbone_name", "resnet18")
        self.freeze_backbone = self.config.get("freeze_backbone", False)

        self.transform = get_test_transforms(self.image_size)

        self.model = SpectraObjectNet(
            output_size=len(self.labels),
            image_size=self.image_size,
            dropout_rate=self.dropout_rate,
            backbone_name=self.backbone_name,
            pretrained=False,
            freeze_backbone=self.freeze_backbone,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict_frame(
        self,
        image_path: str,
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
        group_by_category: bool = False,
    ) -> Dict[str, Any]:
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

        cutoff = self.threshold if threshold is None else threshold
        limit = self.top_k if top_k is None else top_k

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.sigmoid(logits).squeeze(0).cpu()

        predictions = []

        for label, score in zip(self.labels, probabilities):
            score = float(score)

            if score >= cutoff:
                predictions.append(
                    {
                        "label": label,
                        "score": round(score, 4),
                    }
                )

        predictions.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        predictions = self._clean_object_predictions(predictions)

        if limit is not None:
            predictions = predictions[:limit]

        result = {
            "frame_path": str(image_path),
            "task_name": "object",
            "threshold": cutoff,
            "predictions": predictions,
        }

        if group_by_category:
            result["grouped_predictions"] = {
                "scene": [],
                "person": [],
                "object": predictions,
                "action": [],
            }

        return result

    def predict_top_labels(
        self,
        image_path: str,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        result = self.predict_frame(
            image_path=image_path,
            threshold=0.0,
            top_k=top_k,
            group_by_category=False,
        )

        return {
            "frame_path": result["frame_path"],
            "task_name": "object",
            "top_predictions": result["predictions"],
        }

    def _clean_object_predictions(
        self,
        predictions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Limpeza leve para reduzir redundância sem esconder objetos úteis.
        """

        best_by_label: Dict[str, Dict[str, Any]] = {}

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

        cleaned = list(best_by_label.values())

        by_label = {
            item["label"]: item
            for item in cleaned
        }

        # Quando labels específicas aparecem, a label ampla pode atrapalhar a narrativa.
        if "dog" in by_label or "cat" in by_label:
            by_label.pop("animal", None)

        # Evita duplicação semântica comum em telas.
        if "television" in by_label or "computer" in by_label:
            screen_score = float(by_label.get("screen", {}).get("score", 0.0))
            if screen_score < 0.65:
                by_label.pop("screen", None)

        cleaned = list(by_label.values())
        cleaned.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return cleaned
