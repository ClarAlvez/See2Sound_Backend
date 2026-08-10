from pathlib import Path

import torch

from ai.spectra.Object.inference import ObjectPredictor
from ai.spectra.Person.inference import PersonPredictor
from ai.spectra.Scene.inference import ScenePredictor


class SpectraPredictor:
    """Fachada compatível que encaminha a inferência ao módulo do modelo."""

    PREDICTORS = {
        "scene": ScenePredictor,
        "person": PersonPredictor,
        "object": ObjectPredictor,
    }

    def __init__(self, model_path="data/models/spectra_scene/scene_net_best.pt", threshold=0.5, top_k=None, task_name=None, device=None):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo da Spectra não encontrado: {model_path}")
        if task_name is None:
            checkpoint = torch.load(model_path, map_location="cpu")
            task_name = checkpoint.get("task_name", "scene")
        if task_name not in self.PREDICTORS:
            raise ValueError(f"Task visual não suportada: {task_name}")
        self.task_name = task_name
        self._predictor = self.PREDICTORS[task_name](model_path, threshold, top_k, device)

    def predict_frame(self, *args, **kwargs):
        return self._predictor.predict_frame(*args, **kwargs)

    def predict_top_labels(self, *args, **kwargs):
        return self._predictor.predict_top_labels(*args, **kwargs)
