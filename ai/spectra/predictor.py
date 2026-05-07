from pathlib import Path

import torch
from typing import Union, Optional

from ai.spectra.feature_extractor import SpectraFeatureExtractor
from ai.spectra.network import SpectraNet


class SpectraPredictor:
    """
    Usa a SpectraNet treinada para prever labels visuais em novos frames.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "data/models/spectra_net.pt",
        threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modelo da Spectra não encontrado: {self.model_path}"
            )

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device
        )

        self.labels = checkpoint["labels"]
        self.input_size = checkpoint["input_size"]
        self.output_size = checkpoint["output_size"]

        self.feature_extractor = SpectraFeatureExtractor(device=self.device)

        self.model = SpectraNet(
            input_size=self.input_size,
            output_size=self.output_size
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict_frame(self, frame_path, top_k=None) -> dict:
        features = self.feature_extractor.extract_image_features(frame_path)

        features = features.unsqueeze(0).to(self.device)

        logits = self.model(features)

        probabilities = torch.sigmoid(logits).squeeze(0).cpu()

        predictions = []

        for label, probability in zip(self.labels, probabilities):
            score = float(probability)

            if score >= self.threshold:
                predictions.append({
                    "label": label,
                    "score": round(score, 4)
                })

        predictions.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        if top_k is not None:
            predictions = predictions[:top_k]

        return {
            "frame_path": str(frame_path),
            "predictions": predictions
        }