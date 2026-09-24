from pathlib import Path
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    resnet18,
    resnet34,
    resnet50,
)

from ai.spectra.Correlation.encoders.base import (
    IdentityEncoder,
)


class ResNetIdentityEncoder(
    IdentityEncoder
):
    SUPPORTED_BACKBONES = {
        "resnet18",
        "resnet34",
        "resnet50",
    }

    EMBEDDING_DIMENSIONS = {
        "resnet18": 512,
        "resnet34": 512,
        "resnet50": 2048,
    }

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        device: Optional[str] = None,
    ):
        backbone_name = (
            backbone_name
            .strip()
            .lower()
        )

        if (
            backbone_name
            not in self.SUPPORTED_BACKBONES
        ):
            raise ValueError(
                "Backbone não suportado: "
                f"{backbone_name}. "
                "Opções: "
                f"{sorted(self.SUPPORTED_BACKBONES)}"
            )

        self.backbone_name = (
            backbone_name
        )

        self.pretrained = pretrained

        self.device = (
            device
            or (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        (
            self.model,
            self.transform,
        ) = self._create_model()

        self.model = (
            self.model
            .to(self.device)
        )

        self.model.eval()

    @property
    def name(self):
        return (
            f"resnet:{self.backbone_name}"
        )

    @property
    def embedding_dimension(self):
        return (
            self.EMBEDDING_DIMENSIONS[
                self.backbone_name
            ]
        )

    def _create_model(self):
        if (
            self.backbone_name
            == "resnet18"
        ):
            weights = (
                ResNet18_Weights.DEFAULT
                if self.pretrained
                else None
            )

            model = resnet18(
                weights=weights
            )

        elif (
            self.backbone_name
            == "resnet34"
        ):
            weights = (
                ResNet34_Weights.DEFAULT
                if self.pretrained
                else None
            )

            model = resnet34(
                weights=weights
            )

        elif (
            self.backbone_name
            == "resnet50"
        ):
            weights = (
                ResNet50_Weights.DEFAULT
                if self.pretrained
                else None
            )

            model = resnet50(
                weights=weights
            )

        else:
            raise ValueError(
                "Backbone inválido: "
                f"{self.backbone_name}"
            )

        # Remove o classificador final.
        #
        # A saída passa a ser o vetor de features.
        model.fc = nn.Identity()

        if weights is not None:
            transform = (
                weights.transforms()
            )

        else:
            from torchvision import transforms

            transform = transforms.Compose([
                transforms.Resize(
                    (224, 224)
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406,
                    ],

                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ],
                ),
            ])

        return (
            model,
            transform,
        )

    def _load_image(
        self,
        image_path,
    ):
        image_path = Path(
            image_path
        )

        if not image_path.exists():
            raise FileNotFoundError(
                "Imagem não encontrada: "
                f"{image_path}"
            )

        return (
            Image.open(
                image_path
            )
            .convert("RGB")
        )

    @torch.no_grad()
    def encode(
        self,
        image_path: str,
    ):
        image = self._load_image(
            image_path
        )

        tensor = (
            self.transform(
                image
            )
            .unsqueeze(0)
            .to(self.device)
        )

        embedding = self.model(
            tensor
        )

        embedding = F.normalize(
            embedding,
            p=2,
            dim=1,
        )

        return (
            embedding
            .squeeze(0)
            .detach()
            .cpu()
        )

    @torch.no_grad()
    def encode_batch(
        self,
        image_paths: Iterable[str],
    ) -> List[torch.Tensor]:
        image_paths = list(
            image_paths
        )

        if not image_paths:
            return []

        tensors = []

        for image_path in image_paths:
            image = self._load_image(
                image_path
            )

            tensor = self.transform(
                image
            )

            tensors.append(
                tensor
            )

        batch = (
            torch.stack(
                tensors
            )
            .to(self.device)
        )

        embeddings = self.model(
            batch
        )

        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=1,
        )

        embeddings = (
            embeddings
            .detach()
            .cpu()
        )

        return [
            embedding
            for embedding
            in embeddings
        ]