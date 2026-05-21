from dataclasses import dataclass


@dataclass
class SpectraTrainingConfig:
    """
    Configuração de hiperparâmetros do treinamento da Spectra.
    """

    image_size: int = 224
    batch_size: int = 16
    epochs: int = 20

    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    dropout_rate: float = 0.3

    threshold: float = 0.5

    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15

    seed: int = 42

    optimizer_name: str = "adam"

    num_workers: int = 0