from pathlib import Path

import torch
from PIL import Image

from ai.spectra.Scene.labels import LABELS
from ai.spectra.Scene.model import SpectraSceneNet
from ai.spectra.data.transforms import get_test_transforms


class ScenePredictor:
    def __init__(self, model_path, threshold=0.5, top_k=None, device=None):
        self.model_path = Path(model_path)
        self.threshold, self.top_k = threshold, top_k
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo de cena não encontrado: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.labels = checkpoint.get("labels", LABELS)
        config = checkpoint.get("config", {})
        self.transform = get_test_transforms(config.get("image_size", 224))
        self.model = SpectraSceneNet(
            len(self.labels), config.get("image_size", 224), config.get("dropout_rate", 0.3),
            config.get("backbone_name", "resnet18"), pretrained=False,
            freeze_backbone=config.get("freeze_backbone", False),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict_frame(self, image_path, threshold=None, top_k=None, group_by_category=False):
        image_path = Path(image_path)
        image = self.transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probabilities = torch.sigmoid(self.model(image)).squeeze(0).cpu()
        predictions = self._select(probabilities, threshold, top_k)
        result = {"frame_path": str(image_path), "task_name": "scene", "threshold": self.threshold if threshold is None else threshold, "predictions": predictions}
        if group_by_category:
            result["grouped_predictions"] = {"scene": predictions, "person": [], "object": []}
        return result

    def predict_top_labels(self, image_path, top_k=10):
        result = self.predict_frame(image_path, threshold=0.0, top_k=top_k)
        return {"frame_path": result["frame_path"], "task_name": "scene", "top_predictions": result["predictions"]}

    def _select(self, probabilities, threshold, top_k):
        threshold = self.threshold if threshold is None else threshold
        top_k = self.top_k if top_k is None else top_k
        values = [{"label": label, "score": round(float(score), 4)} for label, score in zip(self.labels, probabilities) if score >= threshold]
        values.sort(key=lambda item: item["score"], reverse=True)
        exclusive = [["indoor", "outdoor"], ["day", "night"], ["dark_place", "bright_place"], ["close_up", "medium_shot", "wide_shot"], ["empty_scene", "one_person", "two_people", "group_of_people", "crowded_scene"]]
        by_label = {item["label"]: item for item in values}
        remove = {label for group in exclusive for label in group if label in by_label and label != max((x for x in group if x in by_label), key=lambda x: by_label[x]["score"], default=label)}
        values = [item for item in values if item["label"] not in remove]
        return values if top_k is None else values[:top_k]
