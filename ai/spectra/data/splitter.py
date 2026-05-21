from torch.utils.data import random_split


def split_dataset(
    dataset,
    train_ratio=0.7,
    validation_ratio=0.15,
    test_ratio=0.15,
    seed=42,
):
    """
    Divide um dataset em treino, validação e teste.

    train_ratio + validation_ratio + test_ratio precisa ser igual a 1.0.

    Retorna:
    - train_dataset
    - validation_dataset
    - test_dataset
    """
    total_ratio = train_ratio + validation_ratio + test_ratio

    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            "A soma das proporções precisa ser 1.0. Soma atual: {}".format(total_ratio)
        )

    dataset_size = len(dataset)

    train_size = int(dataset_size * train_ratio)
    validation_size = int(dataset_size * validation_ratio)

    test_size = dataset_size - train_size - validation_size

    generator = _create_generator(seed)

    return random_split(
        dataset,
        [train_size, validation_size, test_size],
        generator=generator,
    )


def split_dataset_train_validation(
    dataset,
    train_ratio=0.8,
    validation_ratio=0.2,
    seed=42,
):
    """
    Divide um dataset apenas em treino e validação.

    Útil para treino simples ou grid search.
    """
    total_ratio = train_ratio + validation_ratio

    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            "A soma das proporções precisa ser 1.0. Soma atual: {}".format(total_ratio)
        )

    dataset_size = len(dataset)

    train_size = int(dataset_size * train_ratio)
    validation_size = dataset_size - train_size

    generator = _create_generator(seed)

    return random_split(
        dataset,
        [train_size, validation_size],
        generator=generator,
    )


def _create_generator(seed):
    """
    Cria um gerador com seed fixa para garantir divisão reproduzível.
    """
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)

    return generator