from pathlib import Path

from typing import Union
import pandas as pd
import torch
from torch.utils.data import Dataset

from ai.spectra.feature_extractor import SpectraFeatureExtractor
from ai.spectra.label_sets import SPECTRA_LABELS


class SpectraDataset(Dataset):
    """
    Dataset da Spectra.

    Lê um CSV com frames rotulados e transforma cada imagem em:
    - features visuais extraídas pelo CLIP
    - labels corretas para treino da SpectraNet
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        feature_extractor: SpectraFeatureExtractor,
    ):
        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV não encontrado: {self.csv_path}")

        self.dataframe = pd.read_csv(self.csv_path)
        self.feature_extractor = feature_extractor

        self._validate_columns()

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]

        frame_path = row["frame_path"]

        features = self.feature_extractor.extract_image_features(frame_path)

        labels = torch.tensor(
            [float(row[label]) for label in SPECTRA_LABELS],
            dtype=torch.float32
        )

        return features, labels

    def _validate_columns(self):
        required_columns = ["frame_path"] + SPECTRA_LABELS

        missing_columns = [
            column for column in required_columns
            if column not in self.dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Colunas ausentes no CSV da Spectra: {missing_columns}"
            )