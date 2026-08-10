from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class SpectraImageDataset(Dataset):
    """
    Dataset genérico da Spectra.

    Recebe os rótulos explicitamente do módulo Person, Object ou Scene.

    O CSV precisa ter:
    - frame_path
    - uma coluna para cada label usada na task escolhida
    """

    def __init__(
        self,
        csv_path,
        transform=None,
        image_root_dir=None,
        label_columns=None,
    ):
        self.csv_path = Path(csv_path)
        self.transform = transform
        if label_columns is None:
            raise ValueError("label_columns deve ser fornecido pelo módulo do modelo")
        self.label_columns = label_columns

        if image_root_dir is not None:
            self.image_root_dir = Path(image_root_dir)
        else:
            self.image_root_dir = None

        if not self.csv_path.exists():
            raise FileNotFoundError(
                "CSV do dataset não encontrado: {}".format(self.csv_path)
            )

        self.dataframe = pd.read_csv(self.csv_path)

        self._validate_columns()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_path = self._resolve_image_path(row["frame_path"])

        if not image_path.exists():
            raise FileNotFoundError(
                "Imagem não encontrada: {}".format(image_path)
            )

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        labels = torch.tensor(
            [float(row[label]) for label in self.label_columns],
            dtype=torch.float32
        )

        return image, labels

    def _resolve_image_path(self, frame_path):
        image_path = Path(frame_path)

        if image_path.is_absolute():
            return image_path

        if self.image_root_dir is not None:
            return self.image_root_dir / image_path

        return image_path

    def _validate_columns(self):
        required_columns = ["frame_path"] + self.label_columns

        missing_columns = [
            column
            for column in required_columns
            if column not in self.dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                "Colunas ausentes no CSV da Spectra: {}".format(missing_columns)
            )

    def get_labels_count(self):
        counts = {}

        for label in self.label_columns:
            counts[label] = int(self.dataframe[label].sum())

        return counts

    def get_label_columns(self):
        return self.label_columns
