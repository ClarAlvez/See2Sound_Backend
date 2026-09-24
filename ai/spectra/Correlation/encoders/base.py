from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

import torch


class IdentityEncoder(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def embedding_dimension(
        self,
    ) -> Optional[int]:
        pass

    @abstractmethod
    def encode(
        self,
        image_path: str,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def encode_batch(
        self,
        image_paths: Iterable[str],
    ) -> List[torch.Tensor]:
        pass