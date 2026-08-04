from pathlib import Path

import torch
from PIL import Image

from ai.spectra.Person.labels import LABELS
from ai.spectra.Person.model import SpectraPersonNet
from ai.spectra.data.transforms import get_test_transforms


class PersonPredictor:
    def __init__(self, model_path, threshold=0.5, top_k=None, device=None):
        self.model_path = Path(model_path)
        self.threshold, self.top_k = threshold, top_k
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo de pessoa não encontrado: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.labels = checkpoint.get("labels", LABELS)
        config = checkpoint.get("config", {})
        self.transform = get_test_transforms(config.get("image_size", 224))
        self.model = SpectraPersonNet(len(self.labels), config.get("image_size", 224), config.get("dropout_rate", 0.3), config.get("backbone_name", "resnet18"), pretrained=False, freeze_backbone=config.get("freeze_backbone", False)).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict_frame(self, image_path, threshold=None, top_k=None, group_by_category=False):
        image_path = Path(image_path)
        image = self.transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad(): probabilities = torch.sigmoid(self.model(image)).squeeze(0).cpu()
        limit = self.top_k if top_k is None else top_k
        cutoff = self.threshold if threshold is None else threshold
        predictions = [{"label": label, "score": round(float(score), 4)} for label, score in zip(self.labels, probabilities) if score >= cutoff]
        predictions.sort(key=lambda item: item["score"], reverse=True)
        if limit is not None: predictions = predictions[:limit]
        result = {"frame_path": str(image_path), "task_name": "person", "threshold": cutoff, "predictions": predictions}
        if group_by_category: result["grouped_predictions"] = {"scene": [], "person": predictions, "object": []}
        return result

    def predict_top_labels(self, image_path, top_k=10):
        result = self.predict_frame(image_path, threshold=0.0, top_k=top_k)
        return {"frame_path": result["frame_path"], "task_name": "person", "top_predictions": result["predictions"]}
