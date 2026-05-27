from pathlib import Path

import torch
from PIL import Image

from ai.spectra.data.transforms import get_test_transforms
from ai.spectra.labels.label_sets import (
    SPECTRA_LABELS,
    SPECTRA_SCENE_LABELS,
    SPECTRA_PERSON_LABELS,
    SPECTRA_OBJECT_LABELS,
    split_predictions_by_group,
)
from ai.spectra.models.spectra_vision_net import SpectraVisionNet
from ai.spectra.models.spectra_scene_net import SpectraSceneNet
from ai.spectra.models.spectra_person_net import SpectraPersonNet
from ai.spectra.models.spectra_object_net import SpectraObjectNet


class SpectraPredictor:
    """
    Predictor genérico da Spectra.

    Ele consegue carregar modelos diferentes:
    - scene
    - person
    - object
    - all/vision

    O tipo do modelo é lido do checkpoint salvo no treino.
    """

    def __init__(
        self,
        model_path="data/models/spectra_scene/scene_net_best.pt",
        threshold=0.5,
        top_k=None,
        task_name=None,
        device=None,
    ):
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.top_k = top_k
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Modelo da Spectra não encontrado: {}".format(self.model_path)
            )

        self.checkpoint = torch.load(
            self.model_path,
            map_location=self.device
        )

        self.task_name = task_name or self.checkpoint.get("task_name", "scene")
        self.labels = self.checkpoint.get("labels", self._get_default_labels(self.task_name))

        self.config = self.checkpoint.get("config", {})

        self.image_size = self.config.get("image_size", 224)
        self.dropout_rate = self.config.get("dropout_rate", 0.3)

        self.transform = get_test_transforms(
            image_size=self.image_size
        )

        self.model = self._create_model().to(self.device)

        self.model.load_state_dict(
            self.checkpoint["model_state_dict"]
        )

        self.model.eval()

    def _get_default_labels(self, task_name):
        if task_name == "scene":
            return SPECTRA_SCENE_LABELS

        if task_name == "person":
            return SPECTRA_PERSON_LABELS

        if task_name == "object":
            return SPECTRA_OBJECT_LABELS

        return SPECTRA_LABELS

    def _create_model(self):
        output_size = len(self.labels)

        if self.task_name == "scene":
            return SpectraSceneNet(
                output_size=output_size,
                image_size=self.image_size,
                dropout_rate=self.dropout_rate,
                backbone_name=self.config.get("backbone_name", "resnet18"),
                pretrained=False,
                freeze_backbone=self.config.get("freeze_backbone", False),
            )

        if self.task_name == "person":
            return SpectraPersonNet(
                output_size=output_size,
                image_size=self.image_size,
                dropout_rate=self.dropout_rate,
                backbone_name=self.config.get("backbone_name", "resnet18"),
                pretrained=False,
                freeze_backbone=self.config.get("freeze_backbone", False),
            )

        if self.task_name == "object":
            return SpectraObjectNet(
                output_size=output_size,
                image_size=self.image_size,
                dropout_rate=self.dropout_rate,
                backbone_name=self.config.get("backbone_name", "resnet18"),
                pretrained=False,
                freeze_backbone=self.config.get("freeze_backbone", False),
            )

        return SpectraVisionNet(
            output_size=output_size,
            image_size=self.image_size,
            dropout_rate=self.dropout_rate,
        )
        
    def _apply_scene_consistency_rules(self, predictions):
        """
        Aplica regras de consistência para modelos de cena.

        Remove contradições como:
        - indoor e outdoor juntos
        - day e night juntos
        - close_up, medium_shot e wide_shot juntos
        - one_person, two_people, group_of_people, crowded_scene e empty_scene juntos
        """
        if self.task_name != "scene":
            return predictions

        prediction_by_label = {
            prediction["label"]: prediction
            for prediction in predictions
        }

        exclusive_groups = [
            ["indoor", "outdoor"],
            ["day", "night"],
            ["dark_place", "bright_place"],
            ["close_up", "medium_shot", "wide_shot"],
            ["empty_scene", "one_person", "two_people", "group_of_people", "crowded_scene"],
            ["calm_scene", "action_scene", "conversation_scene", "movement_scene"],
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
                key=lambda label: prediction_by_label[label]["score"]
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

    def predict_frame(self, image_path, threshold=None, top_k=None, group_by_category=False):
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                "Imagem não encontrada: {}".format(image_path)
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
            "task_name": self.task_name,
            "threshold": threshold,
            "predictions": predictions,
        }

        if group_by_category:
            result["grouped_predictions"] = split_predictions_by_group(
                predictions
            )

        return result

    def predict_top_labels(self, image_path, top_k=10):
        """
        Retorna as labels mais prováveis, ignorando threshold.

        Ideal para debug.
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                "Imagem não encontrada: {}".format(image_path)
            )

        image_tensor = self._load_image_as_tensor(image_path)

        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.sigmoid(logits).squeeze(0).cpu()

        all_predictions = []

        for label, probability in zip(self.labels, probabilities):
            all_predictions.append({
                "label": label,
                "score": round(float(probability), 4),
            })

        all_predictions.sort(
            key=lambda item: item["score"],
            reverse=True,
        )
        
        top_predictions = all_predictions[:top_k]

        if self.task_name == "scene":
            top_predictions = self._apply_scene_consistency_rules(top_predictions)

        return {
            "frame_path": str(image_path),
            "task_name": self.task_name,
            "top_predictions": top_predictions,
        }

    def _load_image_as_tensor(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)

        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        return image_tensor

    def _build_predictions(self, probabilities, threshold, top_k):
        predictions = []

        for label, probability in zip(self.labels, probabilities):
            score = float(probability)

            if score >= threshold:
                predictions.append({
                    "label": label,
                    "score": round(score, 4),
                })

        predictions.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        predictions = self._apply_scene_consistency_rules(predictions)

        if top_k is not None:
            predictions = predictions[:top_k]

        return predictions
