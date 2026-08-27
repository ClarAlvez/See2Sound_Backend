import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from ai.spectra.Object.labels import LABELS
from ai.spectra.Object.model import SpectraObjectNet
from ai.spectra.data.dataset import SpectraImageDataset
from ai.spectra.data.transforms import get_train_transforms, get_validation_transforms


@dataclass
class ObjectTrainingConfig:
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    dropout_rate: float = 0.3
    threshold: float = 0.5
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    num_workers: int = 0
    backbone_name: str = "resnet18"
    pretrained: bool = True
    freeze_backbone: bool = False
    optimizer_name: str = "adamw"


def create_data_splits(dataset_size: int, config: ObjectTrainingConfig):
    if dataset_size < 3:
        raise ValueError("Dataset precisa ter pelo menos 3 imagens para treino/val/test.")

    total_ratio = config.train_ratio + config.validation_ratio + config.test_ratio

    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + validation_ratio + test_ratio precisa somar 1.0")

    generator = torch.Generator().manual_seed(config.seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()

    train_size = int(dataset_size * config.train_ratio)
    validation_size = int(dataset_size * config.validation_ratio)

    train_indices = indices[:train_size]
    validation_indices = indices[train_size:train_size + validation_size]
    test_indices = indices[train_size + validation_size:]

    if not train_indices or not validation_indices or not test_indices:
        raise ValueError(
            "Divisão gerou split vazio. Aumente o dataset ou ajuste as proporções."
        )

    return train_indices, validation_indices, test_indices


def create_optimizer(model, config: ObjectTrainingConfig):
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    name = config.optimizer_name.lower().strip()

    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    if name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )

    raise ValueError(f"Otimizador não suportado: {config.optimizer_name}")


def calculate_multilabel_metrics(logits, labels, threshold: float = 0.5) -> Dict[str, float]:
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()
    labels = labels.float()

    true_positive = (predictions * labels).sum().item()
    false_positive = (predictions * (1 - labels)).sum().item()
    false_negative = ((1 - predictions) * labels).sum().item()

    precision = true_positive / (true_positive + false_positive + 1e-8)
    recall = true_positive / (true_positive + false_negative + 1e-8)
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)

    hamming_accuracy = (predictions == labels).float().mean().item()
    exact_match_accuracy = (predictions == labels).all(dim=1).float().mean().item()

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "hamming_accuracy": hamming_accuracy,
        "exact_match_accuracy": exact_match_accuracy,
    }


def run_epoch(
    model,
    data_loader,
    criterion,
    device,
    threshold: float,
    optimizer=None,
) -> Dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_batches = 0

    metric_sums = {
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "hamming_accuracy": 0.0,
        "exact_match_accuracy": 0.0,
    }

    with torch.set_grad_enabled(is_training):
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels.float())

            if is_training:
                loss.backward()
                optimizer.step()

            metrics = calculate_multilabel_metrics(
                logits=logits.detach(),
                labels=labels.detach(),
                threshold=threshold,
            )

            total_loss += loss.item()
            total_batches += 1

            for key, value in metrics.items():
                metric_sums[key] += value

    result = {
        key: value / max(1, total_batches)
        for key, value in metric_sums.items()
    }

    result["loss"] = total_loss / max(1, total_batches)

    return result


def train_object_model(
    dataset_path: str,
    output_dir: str,
    config: Optional[ObjectTrainingConfig] = None,
):
    config = config or ObjectTrainingConfig()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nIniciando treino da SpectraObjectNet")
    print("Dataset:", dataset_path)
    print("Output:", output_dir)
    print("Device:", device)
    print("Backbone:", config.backbone_name)
    print("Pretrained:", config.pretrained)
    print("Freeze backbone:", config.freeze_backbone)
    print("Epochs:", config.epochs)
    print("Batch size:", config.batch_size)
    print("Learning rate:", config.learning_rate)

    train_data = SpectraImageDataset(
        dataset_path,
        get_train_transforms(config.image_size),
        label_columns=LABELS,
    )

    eval_data = SpectraImageDataset(
        dataset_path,
        get_validation_transforms(config.image_size),
        label_columns=LABELS,
    )

    train_indices, validation_indices, test_indices = create_data_splits(
        dataset_size=len(train_data),
        config=config,
    )

    train_loader = DataLoader(
        Subset(train_data, train_indices),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    validation_loader = DataLoader(
        Subset(eval_data, validation_indices),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    test_loader = DataLoader(
        Subset(eval_data, test_indices),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = SpectraObjectNet(
        output_size=len(LABELS),
        image_size=config.image_size,
        dropout_rate=config.dropout_rate,
        backbone_name=config.backbone_name,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
    ).to(device)

    optimizer = create_optimizer(model, config)
    criterion = nn.BCEWithLogitsLoss()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "object_net_best.pt"
    history_path = output_dir / "training_history.json"
    test_result_path = output_dir / "test_result.csv"

    best_f1 = -1.0
    best_validation_result = None
    history = []

    for epoch in range(1, config.epochs + 1):
        train_result = run_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            device=device,
            threshold=config.threshold,
            optimizer=optimizer,
        )

        validation_result = run_epoch(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
            threshold=config.threshold,
            optimizer=None,
        )

        history.append(
            {
                "epoch": epoch,
                "train": train_result,
                "validation": validation_result,
            }
        )

        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"Train Loss: {train_result['loss']:.4f} | "
            f"Val Loss: {validation_result['loss']:.4f} | "
            f"Val F1: {validation_result['f1_score']:.4f} | "
            f"Val Precision: {validation_result['precision']:.4f} | "
            f"Val Recall: {validation_result['recall']:.4f}"
        )

        if validation_result["f1_score"] > best_f1:
            best_f1 = validation_result["f1_score"]
            best_validation_result = validation_result

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "labels": LABELS,
                    "task_name": "object",
                    "config": asdict(config),
                    "best_validation_result": best_validation_result,
                },
                checkpoint_path,
            )

    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_result = run_epoch(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        threshold=config.threshold,
        optimizer=None,
    )

    pd.DataFrame([test_result]).to_csv(test_result_path, index=False)

    print("\nTreinamento finalizado.")
    print("Melhor modelo salvo em:", checkpoint_path)
    print("Melhor F1 de validação:", best_f1)
    print("Resultado de teste:", test_result)

    return {
        "best_model_path": str(checkpoint_path),
        "best_validation_f1": best_f1,
        "best_validation_result": best_validation_result,
        "test_result": test_result,
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Treinamento do modelo Object da Spectra",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout-rate", type=float, default=0.3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "adam", "sgd"])
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    config = ObjectTrainingConfig(
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
        num_workers=args.num_workers,
        backbone_name=args.backbone,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
        optimizer_name=args.optimizer,
    )

    train_object_model(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        config=config,
    )


if __name__ == "__main__":
    main()
