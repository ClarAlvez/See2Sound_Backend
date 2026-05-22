import torch
from torch import nn


class SpectraPersonNet(nn.Module):
    """
    Modelo de pessoa da Spectra.

    Deve receber crops/recortes de pessoas, não o frame inteiro.

    Detecta:
    - cabelo
    - roupas
    - acessórios
    - aparência visual
    - atributos úteis para diferenciação de personagens
    """

    def __init__(
        self,
        output_size,
        image_size=224,
        dropout_rate=0.3,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        flattened_size = self._calculate_flattened_size(image_size)

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(256, output_size),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def _calculate_flattened_size(self, image_size):
        with torch.no_grad():
            fake_input = torch.zeros(1, 3, image_size, image_size)
            fake_output = self.features(fake_input)

        return fake_output.view(1, -1).shape[1]