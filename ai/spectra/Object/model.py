import torch
from torch import nn
from torchvision import models


class SpectraObjectNet(nn.Module):
    """
    Modelo multilabel de objetos da Spectra.

    Entrada:
    - frame inteiro
    - crop/região candidata de objeto

    Saída:
    - logits multilabel, treinados com BCEWithLogitsLoss.
    """

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

        self.backbone, in_features = self._create_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
        )

        if freeze_backbone:
            self._freeze_backbone()

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, output_size),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def predict_probabilities(self, x):
        return torch.sigmoid(self.forward(x))

    def _create_backbone(self, backbone_name: str, pretrained: bool):
        backbone_name = backbone_name.lower().strip()

        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)

        elif backbone_name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            model = models.resnet34(weights=weights)

        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)

        else:
            raise ValueError(
                f"Backbone não suportado: {backbone_name}. "
                "Use resnet18, resnet34 ou resnet50."
            )

        in_features = model.fc.in_features
        model.fc = nn.Identity()

        return model, in_features

    def _freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
