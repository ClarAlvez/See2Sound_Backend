from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from ai.spectra.Person.labels import LABELS
from ai.spectra.Person.model import SpectraPersonNet
from ai.spectra.data.transforms import get_test_transforms

LABEL_MIN_THRESHOLDS = {
    "person": 0.50,

    "man": 0.70,
    "woman": 0.70,

    "child": 0.60,
    "adult": 0.45,
    "elderly": 0.60,

    "short_hair": 0.50,
    "long_hair": 0.60,
    "bald_hair": 0.60,

    "black_hair": 0.60,
    "blonde_hair": 0.60,
    "brown_hair": 0.60,
    "gray_hair": 0.60,

    "straight_hair": 0.65,
    "wavy_hair": 0.65,
    "bangs_hair": 0.65,
    "receding_hairline": 0.65,

    "glasses": 0.60,
    "hat": 0.60,

    "bag": 0.70,
    "backpack": 0.70,
}

class PersonPredictor:
    """
    Predictor específico da SpectraPersonNet.

    Carrega checkpoints salvos pelo treino de pessoa e retorna labels
    multilabel para atributos visuais de pessoa.
    """

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
                f"Modelo de pessoa não encontrado: {self.model_path}"
            )

        self.checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
        )

        self.labels = self.checkpoint.get("labels", LABELS)
        self.config = self.checkpoint.get("config", {})

        self.image_size = self.config.get("image_size", 224)
        self.dropout_rate = self.config.get("dropout_rate", 0.3)
        self.backbone_name = self.config.get("backbone_name", "resnet18")
        self.freeze_backbone = self.config.get("freeze_backbone", False)

        self.transform = get_test_transforms(
            image_size=self.image_size,
        )

        self.model = SpectraPersonNet(
            output_size=len(self.labels),
            image_size=self.image_size,
            dropout_rate=self.dropout_rate,
            backbone_name=self.backbone_name,
            pretrained=False,
            freeze_backbone=self.freeze_backbone,
        ).to(self.device)

        self.model.load_state_dict(
            self.checkpoint["model_state_dict"]
        )

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
            raise FileNotFoundError(
                f"Imagem não encontrada: {image_path}"
            )

        threshold = self.threshold if threshold is None else threshold
        top_k = self.top_k if top_k is None else top_k

        image_tensor = self._load_image_as_tensor(image_path)

        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.sigmoid(logits).squeeze(0).cpu()

        predictions = self._build_predictions(
            probabilities=probabilities,
            threshold=threshold,
            top_k=top_k,
        )

        result = {
            "frame_path": str(image_path),
            "task_name": "person",
            "threshold": threshold,
            "predictions": predictions,
        }

        if group_by_category:
            result["grouped_predictions"] = {
                "scene": [],
                "person": predictions,
                "object": [],
            }

        return result

    def predict_top_labels(
        self,
        image_path: str,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Retorna as labels mais prováveis, ignorando threshold.

        Útil para debug visual.
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Imagem não encontrada: {image_path}"
            )

        image_tensor = self._load_image_as_tensor(image_path)

        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.sigmoid(logits).squeeze(0).cpu()

        all_predictions = []

        for label, probability in zip(self.labels, probabilities):
            all_predictions.append(
                {
                    "label": label,
                    "score": round(float(probability), 4),
                }
            )

        all_predictions.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        top_predictions = all_predictions[:top_k]
        top_predictions = self._apply_person_consistency_rules(top_predictions)

        return {
            "frame_path": str(image_path),
            "task_name": "person",
            "top_predictions": top_predictions,
        }

    def _load_image_as_tensor(self, image_path: Path) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)

        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        return image_tensor

    def _build_predictions(
        self,
        probabilities,
        threshold: float,
        top_k: Optional[int],
    ) -> List[Dict[str, Any]]:
        predictions = []

        for label, probability in zip(self.labels, probabilities):
            score = float(probability)

            label_threshold = LABEL_MIN_THRESHOLDS.get(label, threshold)
            final_threshold = max(threshold, label_threshold)

            if score >= final_threshold:
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

        predictions = self._apply_person_consistency_rules(predictions)

        if top_k is not None:
            predictions = predictions[:top_k]

        return predictions

    def _apply_person_consistency_rules(
        self,
        predictions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Remove contradições comuns em atributos de pessoa.

        Exemplos:
        - man e woman ao mesmo tempo;
        - short_hair, long_hair e bald_hair juntos;
        - várias cores de cabelo simultâneas;
        - straight_hair e wavy_hair juntos.
        """
        prediction_by_label = {
            prediction["label"]: prediction
            for prediction in predictions
        }

        exclusive_groups = [
            ["man", "woman"],
            ["child", "adult", "elderly"],
            ["short_hair", "long_hair", "bald_hair"],
            ["black_hair", "blonde_hair", "brown_hair", "gray_hair"],
            ["straight_hair", "wavy_hair"],
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

        filtered_predictions = [
            prediction
            for prediction in predictions
            if prediction["label"] not in labels_to_remove
        ]

        return filtered_predictions