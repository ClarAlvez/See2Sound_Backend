from pathlib import Path

import torch

from ai.spectra.Correlation.labels import LABELS
from ai.spectra.Correlation.model import SpectraCorrelationNet


class CorrelationPredictor:
    def __init__(self, model_path, threshold=0.5, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(Path(model_path), map_location=self.device)
        config = checkpoint.get("config", {})
        self.labels = checkpoint.get("labels", LABELS)
        self.threshold = threshold
        self.model = SpectraCorrelationNet(
            input_size=config["input_size"], hidden_size=config.get("hidden_size", 128),
            output_size=len(self.labels), num_layers=config.get("num_layers", 1),
            dropout_rate=config.get("dropout_rate", 0.3),
            bidirectional=config.get("bidirectional", True),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict_sequence(self, sequence):
        tensor = torch.as_tensor(sequence, dtype=torch.float32, device=self.device)
        if tensor.ndim == 2: tensor = tensor.unsqueeze(0)
        with torch.no_grad(): probabilities = torch.sigmoid(self.model(tensor)).squeeze(0).cpu()
        return [{"label": label, "score": round(float(score), 4)} for label, score in zip(self.labels, probabilities) if score >= self.threshold]
