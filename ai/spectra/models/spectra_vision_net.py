import torch
from torch import nn


class SpectraVisionNet(nn.Module):
    """
    Rede neural visual da Spectra.

    Responsável por analisar frames/imagens e realizar classificação multilabel,
    identificando elementos como:
    - pessoas
    - objetos
    - ações
    - cenários
    - composição visual da cena

    Entrada esperada:
        Tensor no formato [batch_size, 3, 224, 224]

    Saída:
        Logits no formato [batch_size, output_size]

    Observação:
        A rede retorna logits crus.
        Durante o treino, use BCEWithLogitsLoss.
        Durante a inferência, aplique torch.sigmoid(logits).
    """

    def __init__(
        self,
        output_size,
        image_size=224,
        dropout_rate=0.3,
    ):
        super().__init__()

        self.output_size = output_size
        self.image_size = image_size
        self.dropout_rate = dropout_rate

        self.features = nn.Sequential(
            # Bloco 1: 224x224 -> 112x112
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout_rate),

            # Bloco 2: 112x112 -> 56x56
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout_rate),

            # Bloco 3: 56x56 -> 28x28
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout_rate),

            # Bloco 4: 28x28 -> 14x14
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
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
        x = self.features(x)
        x = self.classifier(x)

        return x

    def predict_probabilities(self, x):
        """
        Retorna probabilidades entre 0 e 1.

        Use apenas para inferência/testes rápidos.
        No treino, prefira forward() + BCEWithLogitsLoss.
        """
        logits = self.forward(x)
        probabilities = torch.sigmoid(logits)

        return probabilities

    def _calculate_flattened_size(self, image_size):
        """
        Calcula automaticamente o tamanho do vetor após as camadas convolucionais.

        Isso evita erro caso o image_size mude no futuro.
        """
        with torch.no_grad():
            fake_input = torch.zeros(1, 3, image_size, image_size)
            fake_output = self.features(fake_input)

        return fake_output.view(1, -1).shape[1]