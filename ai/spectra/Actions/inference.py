from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from ai.spectra.Actions.labels import LABELS
from ai.spectra.Actions.model import SpectraActionNet
from ai.spectra.data.transforms import get_test_transforms


class ActionPredictor:
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
                f"Modelo de ações não encontrado: {self.model_path}"
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

        self.model = SpectraActionNet(
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

        predictions = self._apply_action_consistency_rules(predictions)
        predictions = self._clean_action_predictions(predictions)

        if limit is not None:
            predictions = predictions[:limit]

        result = {
            "frame_path": str(image_path),
            "task_name": "action",
            "threshold": cutoff,
            "predictions": predictions,
        }

        if group_by_category:
            result["grouped_predictions"] = {
                "scene": [],
                "person": [],
                "object": [],
                "action": predictions,
            }

        return result

    def predict_top_labels(
        self,
        image_path: str,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        return self.predict_frame(
            image_path=image_path,
            threshold=0.0,
            top_k=top_k,
            group_by_category=False,
        )

    def _clean_action_predictions(self, predictions):
        """
        Remove falsos positivos comuns e adiciona labels derivadas simples.
        """

        if not predictions:
            return []

        by_label = {
            item["label"]: float(item["score"])
            for item in predictions
        }

        cleaned = []

        for item in predictions:
            label = item["label"]
            score = float(item["score"])

            # arms_raised confunde muito com corrida, dança, exercício e salto.
            # Só mantém se estiver bem confiante.
            if label == "arms_raised" and score < 0.65:
                continue

            # jumping também pode aparecer em frames de corrida.
            # Mantém apenas se estiver mais confiante.
            if label == "jumping" and score < 0.75:
                continue

            if label == "standing" and (
                "moving" in by_label
                or "fast_motion" in by_label
                or "jumping" in by_label
            ):
                if score < 0.50:
                    continue

            cleaned.append(
                {
                    "label": label,
                    "score": round(score, 4),
                }
            )

        cleaned_labels = {item["label"] for item in cleaned}

        moving_score = by_label.get("moving", 0.0)
        exercise_score = by_label.get("exercising", 0.0)
        fast_motion_score = by_label.get("fast_motion", 0.0)

        # Regra derivada para corrida.
        # Não força running só por "moving", precisa ter exercício ou movimento rápido.
        if "running" not in cleaned_labels:
            if moving_score >= 0.65 and exercise_score >= 0.30:
                cleaned.append(
                    {
                        "label": "running",
                        "score": round(min(moving_score, max(exercise_score, 0.30)), 4),
                    }
                )
            elif moving_score >= 0.70 and fast_motion_score >= 0.30:
                cleaned.append(
                    {
                        "label": "running",
                        "score": round(min(moving_score, max(fast_motion_score, 0.30)), 4),
                    }
                )

        cleaned.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return cleaned

    def _apply_action_consistency_rules(
        self,
        predictions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        prediction_by_label = {
            prediction["label"]: prediction
            for prediction in predictions
        }

        exclusive_groups = [
            ["standing", "sitting", "lying_down", "crouching"],
            ["still", "moving", "fast_motion"],
        ]

        labels_to_remove = set()

        for group in exclusive_groups:
            active_labels = [
                label
                for label in group
                if label in prediction_by_label
            ]

            if len(active_labels) <= 1:
                continue

            best_label = max(
                active_labels,
                key=lambda label: prediction_by_label[label]["score"],
            )

            for label in active_labels:
                if label != best_label:
                    labels_to_remove.add(label)

        return [
            prediction
            for prediction in predictions
            if prediction["label"] not in labels_to_remove
        ]