from pathlib import Path

import torch
from PIL import Image

from ai.spectra.data.transforms import get_test_transforms
from ai.spectra.labels.label_sets import (
    SPECTRA_LABELS,
    split_predictions_by_group,
)
from ai.spectra.models.spectra_vision_net import SpectraVisionNet


class SpectraPredictor:
    """
    Classe responsável por usar uma SpectraVisionNet já treinada.

    Ela carrega o modelo salvo em .pt e permite prever labels visuais
    para frames ou imagens novas.
    """

    def __init__(
        self,
        model_path="data/models/spectra/spectra_vision_net_best.pt",
        threshold=0.5,
        top_k=None,
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

        self.labels = self.checkpoint.get("labels", SPECTRA_LABELS)

        config = self.checkpoint.get("config", {})

        self.image_size = config.get("image_size", 224)
        self.dropout_rate = config.get("dropout_rate", 0.3)

        self.transform = get_test_transforms(
            image_size=self.image_size
        )

        self.model = SpectraVisionNet(
            output_size=len(self.labels),
            image_size=self.image_size,
            dropout_rate=self.dropout_rate,
        ).to(self.device)

        self.model.load_state_dict(
            self.checkpoint["model_state_dict"]
        )

        self.model.eval()

    def predict_frame(self, image_path, threshold=None, top_k=None, group_by_category=False):
        """
        Prediz labels para uma imagem/frame.

        Parâmetros:
            image_path:
                caminho da imagem.

            threshold:
                limite mínimo de probabilidade.
                Se None, usa self.threshold.

            top_k:
                quantidade máxima de labels retornadas.
                Se None, usa self.top_k.

            group_by_category:
                se True, retorna as predições separadas por grupo.

        Retorno:
            dict com frame_path, predictions e, opcionalmente, grouped_predictions.
        """
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
            "threshold": threshold,
            "predictions": predictions,
        }

        if group_by_category:
            result["grouped_predictions"] = split_predictions_by_group(
                predictions
            )

        return result

    def predict_frames(self, image_paths, threshold=None, top_k=None, group_by_category=False):
        """
        Prediz labels para várias imagens.
        """
        results = []

        for image_path in image_paths:
            result = self.predict_frame(
                image_path=image_path,
                threshold=threshold,
                top_k=top_k,
                group_by_category=group_by_category,
            )

            results.append(result)

        return results

    def predict_top_labels(self, image_path, top_k=10):
        """
        Retorna somente as labels mais prováveis, ignorando threshold.

        Útil para depuração, quando você quer ver o que a rede está pensando.
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

        return {
            "frame_path": str(image_path),
            "top_predictions": all_predictions[:top_k],
        }

    def _load_image_as_tensor(self, image_path):
        """
        Carrega imagem, aplica transformações e adiciona dimensão de batch.
        """
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

        if top_k is not None:
            predictions = predictions[:top_k]

        return predictions