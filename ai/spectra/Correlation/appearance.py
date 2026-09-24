from pathlib import Path
from typing import Iterable, List

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


class AppearanceEncoder:
    def __init__(
        self,
        backbone_name="resnet18",
        pretrained=True,
        device=None,
    ):
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.backbone_name = backbone_name

        (
            self.model,
            self.transform,
            self.embedding_size,
        ) = self.create_model(
            backbone_name=backbone_name,
            pretrained=pretrained,
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

    def create_model(
        self,
        backbone_name,
        pretrained,
    ):
        if backbone_name != "resnet18":
            raise ValueError(
                "Correlation v0.1 suporta "
                "apenas resnet18 como "
                "appearance encoder."
            )

        weights = (
            models.ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        model = models.resnet18(
            weights=weights
        )

        embedding_size = (
            model.fc.in_features
        )

        model.fc = nn.Identity()

        if weights is not None:
            transform = (
                weights.transforms()
            )

        else:
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
            embedding_size,
        )

    def encode(
        self,
        image_path,
    ):
        embeddings = self.encode_batch(
            [image_path]
        )

        return embeddings[0]

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
            path = Path(
                image_path
            )

            if not path.exists():
                raise FileNotFoundError(
                    f"Crop não encontrado: {path}"
                )

            image = Image.open(
                path
            ).convert(
                "RGB"
            )

            tensor = self.transform(
                image
            )

            tensors.append(
                tensor
            )

        batch = torch.stack(
            tensors
        ).to(
            self.device
        )

        with torch.no_grad():
            embeddings = self.model(
                batch
            )

            embeddings = torch.nn.functional.normalize(
                embeddings,
                p=2,
                dim=1,
            )

        return [
            embedding.detach().cpu()
            for embedding in embeddings
        ]