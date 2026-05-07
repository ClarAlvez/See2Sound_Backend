import torch
from torch import nn


class SpectraNet(nn.Module):
    """
    Rede neural autoral da Spectra.

    Recebe embeddings visuais extraídos pelo CLIP e prevê múltiplas labels,
    como pessoa, carro, rua, noite, correndo, falando etc.
    """

    def __init__(self, input_size: int, output_size: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)