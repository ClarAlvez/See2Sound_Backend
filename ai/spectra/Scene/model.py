import torch
from torch import nn
from torchvision import models


class SpectraSceneNet(nn.Module):
    """
    Modelo de cena da Spectra usando ResNet como backbone.

    A ResNet funciona como extrator visual pré-treinado.
    A cabeça final é própria da Spectra e faz classificação multilabel
    para labels de cena.

    Entrada:
        [batch_size, 3, 224, 224]

    Saída:
        logits [batch_size, output_size]

    Use:
        BCEWithLogitsLoss no treino
        sigmoid na inferência
    """

    def __init__(
        self,
        output_size,
        image_size=224,
        dropout_rate=0.3,
        backbone_name="resnet18",
        pretrained=True,
        freeze_backbone=False,
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
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, output_size),
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)

        return logits

    def predict_probabilities(self, x):
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def _create_backbone(self, backbone_name, pretrained):
        if backbone_name == "resnet18":
            if pretrained:
                weights = models.ResNet18_Weights.DEFAULT
            else:
                weights = None

            model = models.resnet18(weights=weights)

        elif backbone_name == "resnet34":
            if pretrained:
                weights = models.ResNet34_Weights.DEFAULT
            else:
                weights = None

            model = models.resnet34(weights=weights)

        elif backbone_name == "resnet50":
            if pretrained:
                weights = models.ResNet50_Weights.DEFAULT
            else:
                weights = None

            model = models.resnet50(weights=weights)

        else:
            raise ValueError(
                "Backbone não suportado: {}. Use resnet18, resnet34 ou resnet50.".format(
                    backbone_name
                )
            )

        in_features = model.fc.in_features

        # Remove a camada final original da ImageNet.
        # Agora a ResNet retorna apenas o vetor de características.
        model.fc = nn.Identity()

        return model, in_features

    def _freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
