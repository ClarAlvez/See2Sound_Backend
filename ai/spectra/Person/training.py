import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from ai.spectra.Person.labels import LABELS
from ai.spectra.Person.model import SpectraPersonNet
from ai.spectra.data.dataset import SpectraImageDataset
from ai.spectra.data.transforms import (
    get_train_transforms,
    get_validation_transforms,
    get_test_transforms,
)


@dataclass
class PersonTrainingConfig:
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 20

    learning_rate: float = 0.0003
    weight_decay: float = 0.0001
    dropout_rate: float = 0.3

    threshold: float = 0.5

    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15

    seed: int = 42

    optimizer_name: str = "adamw"

    num_workers: int = 0

    backbone_name: str = "resnet18"
    pretrained: bool = True
    freeze_backbone: bool = False


class PersonTrainingHistory:
    def __init__(self):
        self.history = {
            "train_loss": [],
            "validation_loss": [],

            "train_precision": [],
            "validation_precision": [],

            "train_recall": [],
            "validation_recall": [],

            "train_f1_score": [],
            "validation_f1_score": [],

            "train_hamming_accuracy": [],
            "validation_hamming_accuracy": [],

            "train_exact_match_accuracy": [],
            "validation_exact_match_accuracy": [],
        }

    def add_epoch(
        self,
        train_result: Dict[str, float],
        validation_result: Dict[str, float],
    ) -> None:
        self.history["train_loss"].append(train_result["loss"])
        self.history["validation_loss"].append(validation_result["loss"])

        self.history["train_precision"].append(train_result["precision"])
        self.history["validation_precision"].append(validation_result["precision"])

        self.history["train_recall"].append(train_result["recall"])
        self.history["validation_recall"].append(validation_result["recall"])

        self.history["train_f1_score"].append(train_result["f1_score"])
        self.history["validation_f1_score"].append(validation_result["f1_score"])

        self.history["train_hamming_accuracy"].append(train_result["hamming_accuracy"])
        self.history["validation_hamming_accuracy"].append(validation_result["hamming_accuracy"])

        self.history["train_exact_match_accuracy"].append(train_result["exact_match_accuracy"])
        self.history["validation_exact_match_accuracy"].append(validation_result["exact_match_accuracy"])

    def to_dict(self) -> Dict[str, List[float]]:
        return self.history

    def save_json(self, output_path: Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(self.history, file, indent=4, ensure_ascii=False)


def calculate_multilabel_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()

    labels = labels.float()

    true_positive = (predictions * labels).sum().item()
    false_positive = (predictions * (1 - labels)).sum().item()
    false_negative = ((1 - predictions) * labels).sum().item()

    precision = true_positive / (true_positive + false_positive + 1e-8)
    recall = true_positive / (true_positive + false_negative + 1e-8)

    f1_score = (
        2 * precision * recall / (precision + recall + 1e-8)
    )

    hamming_accuracy = (
        (predictions == labels).float().mean().item()
    )

    exact_match_accuracy = (
        (predictions == labels).all(dim=1).float().mean().item()
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "hamming_accuracy": hamming_accuracy,
        "exact_match_accuracy": exact_match_accuracy,
    }


def calculate_loss_average(
    total_loss: float,
    data_loader: DataLoader,
) -> float:
    if len(data_loader) == 0:
        return 0.0

    return total_loss / len(data_loader)


def create_data_splits(
    dataset_size: int,
    config: PersonTrainingConfig,
) -> Tuple[List[int], List[int], List[int]]:
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    indices = torch.randperm(dataset_size, generator=generator).tolist()

    train_size = int(dataset_size * config.train_ratio)
    validation_size = int(dataset_size * config.validation_ratio)

    train_indices = indices[:train_size]
    validation_indices = indices[train_size:train_size + validation_size]
    test_indices = indices[train_size + validation_size:]

    return train_indices, validation_indices, test_indices


def create_dataloaders(
    dataset_path: str,
    config: PersonTrainingConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_base_dataset = SpectraImageDataset(
        csv_path=dataset_path,
        transform=get_train_transforms(config.image_size),
        label_columns=LABELS,
    )

    validation_base_dataset = SpectraImageDataset(
        csv_path=dataset_path,
        transform=get_validation_transforms(config.image_size),
        label_columns=LABELS,
    )

    test_base_dataset = SpectraImageDataset(
        csv_path=dataset_path,
        transform=get_test_transforms(config.image_size),
        label_columns=LABELS,
    )

    dataset_size = len(train_base_dataset)

    if dataset_size <= 2:
        raise ValueError(
            f"Dataset muito pequeno para treino/validação/teste: {dataset_size} amostras."
        )

    train_indices, validation_indices, test_indices = create_data_splits(
        dataset_size=dataset_size,
        config=config,
    )

    train_dataset = Subset(train_base_dataset, train_indices)
    validation_dataset = Subset(validation_base_dataset, validation_indices)
    test_dataset = Subset(test_base_dataset, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return train_loader, validation_loader, test_loader


def create_person_model(
    config: PersonTrainingConfig,
) -> SpectraPersonNet:
    return SpectraPersonNet(
        output_size=len(LABELS),
        image_size=config.image_size,
        dropout_rate=config.dropout_rate,
        backbone_name=config.backbone_name,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
    )


def create_optimizer(
    model: SpectraPersonNet,
    config: PersonTrainingConfig,
):
    optimizer_name = config.optimizer_name.lower()

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=0.9,
        )

    raise ValueError(
        f"Otimizador não suportado: {config.optimizer_name}"
    )


def run_epoch(
    model: SpectraPersonNet,
    data_loader: DataLoader,
    criterion,
    optimizer,
    device: str,
    threshold: float,
    training: bool,
) -> Dict[str, float]:
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    all_logits = []
    all_labels = []

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()

        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    if not all_logits:
        return {
            "loss": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "hamming_accuracy": 0.0,
            "exact_match_accuracy": 0.0,
        }

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = calculate_multilabel_metrics(
        logits=all_logits,
        labels=all_labels,
        threshold=threshold,
    )

    metrics["loss"] = calculate_loss_average(
        total_loss=total_loss,
        data_loader=data_loader,
    )

    return metrics


def save_model(
    model: SpectraPersonNet,
    output_path: Path,
    config: PersonTrainingConfig,
    validation_result: Dict[str, float],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "labels": LABELS,
            "task_name": "person",
            "config": asdict(config),
            "validation_result": validation_result,
        },
        output_path,
    )


def save_test_result(
    test_result: Dict[str, float],
    output_path: Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame([test_result])
    dataframe.to_csv(output_path, index=False)


def plot_training_history(
    history: Dict[str, List[float]],
    output_dir: Path,
) -> None:
    """
    Import local para evitar obrigar matplotlib quando não for treinar.
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_pairs = [
        ("train_loss", "validation_loss", "Training and Validation Loss", "Loss", "loss.png"),
        ("train_f1_score", "validation_f1_score", "Training and Validation F1-score", "F1-score", "f1_score.png"),
        ("train_precision", "validation_precision", "Training and Validation Precision", "Precision", "precision.png"),
        ("train_recall", "validation_recall", "Training and Validation Recall", "Recall", "recall.png"),
        ("train_hamming_accuracy", "validation_hamming_accuracy", "Training and Validation Hamming Accuracy", "Hamming Accuracy", "hamming_accuracy.png"),
        ("train_exact_match_accuracy", "validation_exact_match_accuracy", "Training and Validation Exact Match Accuracy", "Exact Match Accuracy", "exact_match_accuracy.png"),
    ]

    for train_key, validation_key, title, ylabel, file_name in metric_pairs:
        epochs = range(1, len(history[train_key]) + 1)

        plt.figure()
        plt.plot(epochs, history[train_key], label="Training")
        plt.plot(epochs, history[validation_key], label="Validation")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / file_name)
        plt.close()


def train_person_model(
    dataset_path: str,
    output_dir: str,
    config: Optional[PersonTrainingConfig] = None,
) -> Dict[str, Any]:
    """
    Treina a SpectraPersonNet.

    Salva:
    - person_net_best.pt
    - training_history.json
    - test_result.csv
    - plots/*.png
    """
    config = config or PersonTrainingConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, validation_loader, test_loader = create_dataloaders(
        dataset_path=dataset_path,
        config=config,
    )

    model = create_person_model(
        config=config,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = create_optimizer(
        model=model,
        config=config,
    )

    history = PersonTrainingHistory()

    best_validation_f1 = -1.0
    best_model_path = output_dir / "person_net_best.pt"

    print("\nIniciando treino da SpectraPersonNet")
    print("Dataset:", dataset_path)
    print("Output:", output_dir)
    print("Device:", device)
    print("Backbone:", config.backbone_name)
    print("Pretrained:", config.pretrained)
    print("Freeze backbone:", config.freeze_backbone)
    print("Epochs:", config.epochs)
    print("Batch size:", config.batch_size)
    print("Learning rate:", config.learning_rate)
    print("Threshold:", config.threshold)
    print()

    for epoch in range(config.epochs):
        train_result = run_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            threshold=config.threshold,
            training=True,
        )

        validation_result = run_epoch(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            threshold=config.threshold,
            training=False,
        )

        history.add_epoch(
            train_result=train_result,
            validation_result=validation_result,
        )

        print(
            "Epoch {}/{} | "
            "Train Loss: {:.4f} | "
            "Val Loss: {:.4f} | "
            "Val F1: {:.4f} | "
            "Val Precision: {:.4f} | "
            "Val Recall: {:.4f}".format(
                epoch + 1,
                config.epochs,
                train_result["loss"],
                validation_result["loss"],
                validation_result["f1_score"],
                validation_result["precision"],
                validation_result["recall"],
            )
        )

        if validation_result["f1_score"] > best_validation_f1:
            best_validation_f1 = validation_result["f1_score"]

            save_model(
                model=model,
                output_path=best_model_path,
                config=config,
                validation_result=validation_result,
            )

    test_result = run_epoch(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        optimizer=None,
        device=device,
        threshold=config.threshold,
        training=False,
    )

    history_path = output_dir / "training_history.json"
    history.save_json(history_path)

    plot_training_history(
        history=history.to_dict(),
        output_dir=output_dir / "plots",
    )

    save_test_result(
        test_result=test_result,
        output_path=output_dir / "test_result.csv",
    )

    print("\nTreinamento finalizado.")
    print("Melhor modelo salvo em:", best_model_path)
    print("Melhor F1 de validação:", best_validation_f1)
    print("Resultado de teste:", test_result)

    return {
        "best_model_path": str(best_model_path),
        "best_validation_f1": best_validation_f1,
        "test_result": test_result,
        "history": history.to_dict(),
    }


def build_training_config_from_args(args) -> PersonTrainingConfig:
    return PersonTrainingConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,

        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout_rate,

        threshold=args.threshold,

        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,

        seed=args.seed,

        optimizer_name=args.optimizer,

        num_workers=args.num_workers,

        backbone_name=args.backbone,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Treina a SpectraPersonNet."
    )

    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Caminho do CSV de treino.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Pasta onde o modelo treinado será salvo.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0003,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
    )

    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--optimizer",
        choices=["adam", "adamw", "sgd"],
        default="adamw",
    )

    parser.add_argument(
        "--backbone",
        choices=["resnet18", "resnet34", "resnet50"],
        default="resnet18",
    )

    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
    )

    parser.add_argument(
        "--no-pretrained",
        action="store_true",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    config = build_training_config_from_args(args)

    train_person_model(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        config=config,
    )


if __name__ == "__main__":
    main()