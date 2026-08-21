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

    def _clean_action_predictions(
        self,
        predictions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Remove falsos positivos comuns depois da v3 e adiciona labels derivadas simples.

        Observação importante:
        - sports e exercising podem ser semanticamente possíveis em vídeos de corrida.
        - Para audiodescrição, porém, eles são secundários e podem atrapalhar a frase.
        - Por isso só mantemos sports quando há evidência esportiva específica e só mantemos
          exercising quando está forte o bastante para ser útil na descrição.
        """

        if not predictions:
            return []

        by_label = {
            item["label"]: float(item["score"])
            for item in predictions
        }

        def has_any_label(labels: List[str], min_score: float = 0.0) -> bool:
            return any(by_label.get(label, 0.0) >= min_score for label in labels)

        moving_score = by_label.get("moving", 0.0)
        fast_motion_score = by_label.get("fast_motion", 0.0)
        running_score = by_label.get("running", 0.0)
        exercising_score = by_label.get("exercising", 0.0)
        sports_score = by_label.get("sports", 0.0)
        jumping_score = by_label.get("jumping", 0.0)
        arms_raised_score = by_label.get("arms_raised", 0.0)

        has_running_context = (
            running_score >= 0.30
            or (moving_score >= 0.65 and fast_motion_score >= 0.30)
            or (moving_score >= 0.65 and exercising_score >= 0.30)
        )

        has_specific_sport_context = has_any_label(
            [
                "ball_sport",
                "racket_sport",
                "water_activity",
                "martial_activity",
                "swimming",
                "cycling",
            ],
            min_score=0.30,
        )

        cleaned = []

        for item in predictions:
            label = item["label"]
            score = float(item["score"])

            # arms_raised confunde muito com corrida, dança, exercício e salto.
            if label == "arms_raised" and score < 0.85:
                continue

            # jumping aparece bastante como ruído em passadas de corrida.
            if label == "jumping" and score < 0.90:
                continue

            # falling em frames comuns de movimento geralmente é ruído.
            if label == "falling" and score < 0.70:
                continue

            # sports é amplo demais: só mantém se houver contexto esportivo específico
            # ou se o próprio modelo estiver quase absoluto.
            if label == "sports":
                strong_jump_sport = jumping_score >= 0.90

                if not has_specific_sport_context and not strong_jump_sport and score < 0.98:
                    continue

            # exercising pode ser verdade em corrida, mas é secundário para narrativa.
            # Mantém apenas se estiver forte ou se não houver contexto claro de corrida.
            if label == "exercising":
                if has_running_context and score < 0.80:
                    continue

            # throwing ficou sensível demais após mapear esportes.
            # Exige evidência forte de braço/situação esportiva específica.
            if label == "throwing":
                has_throw_context = (
                    arms_raised_score >= 0.85
                    or by_label.get("ball_sport", 0.0) >= 0.50
                    or by_label.get("racket_sport", 0.0) >= 0.50
                )

                if not has_throw_context or score < 0.80:
                    continue

            # standing baixo junto com movimento geralmente é ruído.
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

        # Regra derivada para corrida.
        # Não força running só por moving; precisa haver exercício ou movimento rápido.
        if "running" not in cleaned_labels:
            if moving_score >= 0.65 and exercising_score >= 0.30:
                cleaned.append(
                    {
                        "label": "running",
                        "score": round(min(moving_score, max(exercising_score, 0.30)), 4),
                    }
                )
            elif moving_score >= 0.70 and fast_motion_score >= 0.30:
                cleaned.append(
                    {
                        "label": "running",
                        "score": round(min(moving_score, max(fast_motion_score, 0.30)), 4),
                    }
                )

        cleaned = self._deduplicate_predictions(cleaned)

        cleaned.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return cleaned

    def _deduplicate_predictions(
        self,
        predictions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
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

        return list(best_by_label.values())

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
