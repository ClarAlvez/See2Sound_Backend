from pathlib import Path

from typing import Optional, Union
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class SpectraFeatureExtractor:
    """
    Extrai características visuais de uma imagem usando CLIP.

    A imagem é transformada em um vetor numérico, que depois será usado
    como entrada para a rede neural própria da Spectra.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)

        self.model.eval()

    @torch.no_grad()
    def extract_image_features(self, image_path: Union[str, Path]) -> torch.Tensor:
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        image_features = self.model.get_image_features(**inputs)

        image_features = image_features / image_features.norm(
            dim=-1,
            keepdim=True
        )

        return image_features.squeeze(0).cpu()