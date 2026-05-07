from pathlib import Path
from typing import Any, Union, Optional

from ai.spectra.predictor import SpectraPredictor


class Spectra:
    """
    Interface principal da Spectra Vision.

    Essa classe será usada pelo pipeline principal para analisar frames
    e, futuramente, cenas completas.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "data/models/spectra_net.pt",
        threshold: float = 0.5,
    ):
        self.predictor = SpectraPredictor(
            model_path=model_path,
            threshold=threshold
        )

    def analyze_frame(self, frame_path: Union[str, Path], timestamp: Optional[float] = None) -> dict[str, Any]:
        result = self.predictor.predict_frame(frame_path)

        return {
            "frame_path": result["frame_path"],
            "timestamp": timestamp,
            "predictions": result["predictions"]
        }

    def analyze_frames(self, frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results = []

        for frame in frames:
            frame_path = frame["frame_path"]
            timestamp = frame.get("timestamp")

            analysis = self.analyze_frame(
                frame_path=frame_path,
                timestamp=timestamp
            )

            results.append(analysis)

        return results