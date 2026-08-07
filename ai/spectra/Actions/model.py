from typing import Optional

import torch
from torch import nn
from torchvision import models


class SpectraActionNet(nn.Module):
    def __init__(
        self,
        output_size: int,
        image_size: int = 224,
        dropout_rate: float = 0.3,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.output_size = output_size
        self.image_size = image_size
        self.dropout_rate = dropout_rate
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone

        self.backbone, feature_size = self._create_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
        )

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_size, output_size),
        )

    def _create_backbone(
        self,
        backbone_name: str,
        pretrained: bool,
    ):
        backbone_name = backbone_name.lower()

        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
            feature_size = model.fc.in_features
            model.fc = nn.Identity()
            return model, feature_size

        if backbone_name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            model = models.resnet34(weights=weights)
            feature_size = model.fc.in_features
            model.fc = nn.Identity()
            return model, feature_size

        if backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
            feature_size = model.fc.in_features
            model.fc = nn.Identity()
            return model, feature_size

        raise ValueError(f"Backbone não suportado para Actions: {backbone_name}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        logits = self.classifier(features)
        return logits