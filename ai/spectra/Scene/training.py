from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import csv
import json

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split

from ai.spectra.Scene.labels import LABELS
from ai.spectra.Scene.model import SpectraSceneNet
from ai.spectra.data.dataset import SpectraImageDataset
from ai.spectra.data.transforms import get_train_transforms, get_validation_transforms


@dataclass
class SceneTrainingConfig:
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    dropout_rate: float = 0.3
    validation_ratio: float = 0.15
    seed: int = 42
    num_workers: int = 0
    backbone_name: str = "resnet18"
    pretrained: bool = True
    freeze_backbone: bool = False
    threshold: float = 0.4
    save_plots: bool = True
    cross_validation: bool = False
    folds: int = 5


def train_scene_model(dataset_path, output_dir, config=None):
    config = config or SceneTrainingConfig()

    if config.cross_validation:
        return _train_cross_validation(
            dataset_path=dataset_path,
            output_dir=output_dir,
            config=config,
        )

    return _train(
        dataset_path=dataset_path,
        output_dir=output_dir,
        config=config,
    )


def _train(dataset_path, output_dir, config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "scene_net_best.pt"
    history_json_path = output_dir / "training_history.json"
    history_csv_path = output_dir / "training_history.csv"
    loss_plot_path = output_dir / "training_loss.png"
    metrics_plot_path = output_dir / "training_metrics.png"

    train_data = SpectraImageDataset(
        csv_path=dataset_path,
        transform=get_train_transforms(config.image_size),
        label_columns=LABELS,
    )

    validation_data = SpectraImageDataset(
        csv_path=dataset_path,
        transform=get_validation_transforms(config.image_size),
        label_columns=LABELS,
    )

    train_subset, validation_subset = create_train_validation_split(
        dataset_size=len(train_data),
        validation_ratio=config.validation_ratio,
        seed=config.seed,
    )

    train_loader = DataLoader(
        dataset=Subset(train_data, train_subset.indices),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    validation_loader = DataLoader(
        dataset=Subset(validation_data, validation_subset.indices),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = SpectraSceneNet(
        output_size=len(LABELS),
        image_size=config.image_size,
        dropout_rate=config.dropout_rate,
        backbone_name=config.backbone_name,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    criterion = nn.BCEWithLogitsLoss()

    best_validation_loss = float("inf")

    history = {
        "train_loss": [],
        "validation_loss": [],
        "validation_precision": [],
        "validation_recall": [],
        "validation_f1": [],
    }

    print_training_header(
        dataset_path=dataset_path,
        output_dir=output_dir,
        config=config,
        device=device,
        train_size=len(train_subset.indices),
        validation_size=len(validation_subset.indices),
    )

    for epoch in range(config.epochs):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        validation_loss, validation_metrics = evaluate(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
            threshold=config.threshold,
        )

        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["validation_precision"].append(validation_metrics["precision"])
        history["validation_recall"].append(validation_metrics["recall"])
        history["validation_f1"].append(validation_metrics["f1"])

        print_epoch_result(
            epoch=epoch,
            total_epochs=config.epochs,
            train_loss=train_loss,
            validation_loss=validation_loss,
            validation_metrics=validation_metrics,
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                config=config,
                best_validation_loss=best_validation_loss,
            )

            print("Novo melhor modelo salvo:", checkpoint_path)

        save_training_history_json(
            history=history,
            output_path=history_json_path,
        )

        save_training_history_csv(
            history=history,
            output_path=history_csv_path,
        )

    if config.save_plots:
        save_training_plots(
            history=history,
            loss_plot_path=loss_plot_path,
            metrics_plot_path=metrics_plot_path,
        )

    print("\nTreinamento finalizado.")
    print("Melhor modelo salvo em:", checkpoint_path)
    print("Melhor loss de validação:", best_validation_loss)
    print("Histórico JSON:", history_json_path)
    print("Histórico CSV:", history_csv_path)

    if config.save_plots:
        print("Gráfico de loss:", loss_plot_path)
        print("Gráfico de métricas:", metrics_plot_path)

    return {
        "best_model_path": str(checkpoint_path),
        "best_validation_loss": best_validation_loss,
        "history_json_path": str(history_json_path),
        "history_csv_path": str(history_csv_path),
        "loss_plot_path": str(loss_plot_path) if config.save_plots else None,
        "metrics_plot_path": str(metrics_plot_path) if config.save_plots else None,
    }

def create_k_fold_indices(dataset_size, folds, seed):
    generator = torch.Generator().manual_seed(seed)

    shuffled_indices = torch.randperm(
        dataset_size,
        generator=generator,
    ).tolist()

    fold_sizes = []

    base_fold_size = dataset_size // folds
    remainder = dataset_size % folds

    for fold_index in range(folds):
        fold_size = base_fold_size

        if fold_index < remainder:
            fold_size += 1

        fold_sizes.append(fold_size)

    fold_indices = []
    current_index = 0

    for fold_size in fold_sizes:
        fold = shuffled_indices[current_index:current_index + fold_size]
        fold_indices.append(fold)

        current_index += fold_size

    return fold_indices


def build_cross_validation_summary(fold_results, best_global_model_path):
    validation_losses = [
        result["best_validation_loss"]
        for result in fold_results
    ]

    validation_f1_scores = [
        result["best_validation_f1"]
        for result in fold_results
    ]

    mean_validation_loss = sum(validation_losses) / max(1, len(validation_losses))
    mean_validation_f1 = sum(validation_f1_scores) / max(1, len(validation_f1_scores))

    best_fold = max(
        fold_results,
        key=lambda result: result["best_validation_f1"],
    )

    summary = {
        "folds": len(fold_results),
        "mean_validation_loss": mean_validation_loss,
        "mean_validation_f1": mean_validation_f1,
        "best_fold": best_fold["fold"],
        "best_global_model_path": str(best_global_model_path),
        "fold_results": fold_results,
    }

    return summary

def _train_cross_validation(dataset_path, output_dir, config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = SpectraImageDataset(
        csv_path=dataset_path,
        transform=get_train_transforms(config.image_size),
        label_columns=LABELS,
    )

    validation_data = SpectraImageDataset(
        csv_path=dataset_path,
        transform=get_validation_transforms(config.image_size),
        label_columns=LABELS,
    )

    dataset_size = len(train_data)

    if dataset_size < config.folds:
        raise ValueError(
            "O número de folds não pode ser maior que o tamanho do dataset. "
            f"Dataset: {dataset_size}, folds: {config.folds}"
        )

    fold_indices = create_k_fold_indices(
        dataset_size=dataset_size,
        folds=config.folds,
        seed=config.seed,
    )

    fold_results = []
    best_global_f1 = -1.0
    best_global_loss = float("inf")
    best_global_model_path = None

    print("=" * 80)
    print("TREINANDO SPECTRA SCENE NET - CROSS VALIDATION")
    print("=" * 80)
    print("Dataset:", dataset_path)
    print("Output:", output_dir)
    print("Device:", device)
    print("Labels:", len(LABELS))
    print("Folds:", config.folds)
    print("Epochs por fold:", config.epochs)
    print("Backbone:", config.backbone_name)
    print("Freeze backbone:", config.freeze_backbone)
    print("Threshold:", config.threshold)
    print("=" * 80)

    for fold_index in range(config.folds):
        fold_number = fold_index + 1
        fold_output_dir = output_dir / f"fold_{fold_number}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        validation_indices = fold_indices[fold_index]

        train_indices = []

        for index, indices in enumerate(fold_indices):
            if index == fold_index:
                continue

            train_indices.extend(indices)

        train_loader = DataLoader(
            dataset=Subset(train_data, train_indices),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )

        validation_loader = DataLoader(
            dataset=Subset(validation_data, validation_indices),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

        model = SpectraSceneNet(
            output_size=len(LABELS),
            image_size=config.image_size,
            dropout_rate=config.dropout_rate,
            backbone_name=config.backbone_name,
            pretrained=config.pretrained,
            freeze_backbone=config.freeze_backbone,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        criterion = nn.BCEWithLogitsLoss()

        fold_checkpoint_path = fold_output_dir / "scene_net_best.pt"
        fold_history_json_path = fold_output_dir / "training_history.json"
        fold_history_csv_path = fold_output_dir / "training_history.csv"
        fold_loss_plot_path = fold_output_dir / "training_loss.png"
        fold_metrics_plot_path = fold_output_dir / "training_metrics.png"

        best_fold_loss = float("inf")
        best_fold_f1 = -1.0

        history = {
            "train_loss": [],
            "validation_loss": [],
            "validation_precision": [],
            "validation_recall": [],
            "validation_f1": [],
        }

        print("\n" + "=" * 80)
        print(f"FOLD {fold_number}/{config.folds}")
        print("=" * 80)
        print("Train samples:", len(train_indices))
        print("Validation samples:", len(validation_indices))

        for epoch in range(config.epochs):
            train_loss = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
            )

            validation_loss, validation_metrics = evaluate(
                model=model,
                loader=validation_loader,
                criterion=criterion,
                device=device,
                threshold=config.threshold,
            )

            validation_f1 = validation_metrics["f1"]

            history["train_loss"].append(train_loss)
            history["validation_loss"].append(validation_loss)
            history["validation_precision"].append(validation_metrics["precision"])
            history["validation_recall"].append(validation_metrics["recall"])
            history["validation_f1"].append(validation_f1)

            print(
                f"Fold {fold_number}/{config.folds} "
                f"- Epoch {epoch + 1}/{config.epochs} "
                f"- train_loss: {train_loss:.4f} "
                f"- validation_loss: {validation_loss:.4f} "
                f"- precision: {validation_metrics['precision']:.4f} "
                f"- recall: {validation_metrics['recall']:.4f} "
                f"- f1: {validation_f1:.4f}"
            )

            should_save_fold_model = (
                validation_loss < best_fold_loss
                or validation_f1 > best_fold_f1
            )

            if should_save_fold_model:
                best_fold_loss = validation_loss
                best_fold_f1 = validation_f1

                save_checkpoint(
                    checkpoint_path=fold_checkpoint_path,
                    model=model,
                    config=config,
                    best_validation_loss=best_fold_loss,
                )

                print("Novo melhor modelo do fold salvo:", fold_checkpoint_path)

        save_training_history_json(
            history=history,
            output_path=fold_history_json_path,
        )

        save_training_history_csv(
            history=history,
            output_path=fold_history_csv_path,
        )

        if config.save_plots:
            save_training_plots(
                history=history,
                loss_plot_path=fold_loss_plot_path,
                metrics_plot_path=fold_metrics_plot_path,
            )

        fold_result = {
            "fold": fold_number,
            "best_validation_loss": best_fold_loss,
            "best_validation_f1": best_fold_f1,
            "best_model_path": str(fold_checkpoint_path),
            "history_json_path": str(fold_history_json_path),
            "history_csv_path": str(fold_history_csv_path),
            "loss_plot_path": str(fold_loss_plot_path) if config.save_plots else None,
            "metrics_plot_path": str(fold_metrics_plot_path) if config.save_plots else None,
        }

        fold_results.append(fold_result)

        is_best_global = (
            best_fold_f1 > best_global_f1
            or (
                best_fold_f1 == best_global_f1
                and best_fold_loss < best_global_loss
            )
        )

        if is_best_global:
            best_global_f1 = best_fold_f1
            best_global_loss = best_fold_loss
            best_global_model_path = fold_checkpoint_path

    summary = build_cross_validation_summary(
        fold_results=fold_results,
        best_global_model_path=best_global_model_path,
    )

    summary_path = output_dir / "cross_validation_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("CROSS VALIDATION FINALIZADO")
    print("=" * 80)
    print("Folds:", config.folds)
    print("Média validation loss:", f"{summary['mean_validation_loss']:.4f}")
    print("Média validation F1:", f"{summary['mean_validation_f1']:.4f}")
    print("Melhor modelo global:", summary["best_global_model_path"])
    print("Resumo salvo em:", summary_path)

    return summary


def create_train_validation_split(dataset_size, validation_ratio, seed):
    validation_size = max(1, int(dataset_size * validation_ratio))
    train_size = dataset_size - validation_size

    if train_size <= 0:
        raise ValueError(
            "Dataset pequeno demais para criar treino e validação. "
            f"Total: {dataset_size}, validação: {validation_size}"
        )

    generator = torch.Generator().manual_seed(seed)

    train_subset, validation_subset = random_split(
        range(dataset_size),
        [train_size, validation_size],
        generator=generator,
    )

    return train_subset, validation_subset


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    with torch.set_grad_enabled(is_training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

    return total_loss / max(1, len(loader))


def evaluate(model, loader, criterion, device, threshold):
    model.eval()

    total_loss = 0.0

    true_positives = 0.0
    false_positives = 0.0
    false_negatives = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= threshold).float()

            total_loss += loss.item()

            true_positives += ((predictions == 1) & (labels == 1)).sum().item()
            false_positives += ((predictions == 1) & (labels == 0)).sum().item()
            false_negatives += ((predictions == 0) & (labels == 1)).sum().item()

    precision = true_positives / max(1.0, true_positives + false_positives)
    recall = true_positives / max(1.0, true_positives + false_negatives)
    f1 = (2 * precision * recall) / max(1e-8, precision + recall)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    average_loss = total_loss / max(1, len(loader))

    return average_loss, metrics


def save_checkpoint(checkpoint_path, model, config, best_validation_loss):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "labels": LABELS,
            "task_name": "scene",
            "config": asdict(config),
            "best_validation_loss": best_validation_loss,
        },
        checkpoint_path,
    )


def save_training_history_json(history, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4, ensure_ascii=False)


def save_training_history_csv(history, output_path):
    fieldnames = [
        "epoch",
        "train_loss",
        "validation_loss",
        "validation_precision",
        "validation_recall",
        "validation_f1",
    ]

    total_epochs = len(history["train_loss"])

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for index in range(total_epochs):
            writer.writerow(
                {
                    "epoch": index + 1,
                    "train_loss": history["train_loss"][index],
                    "validation_loss": history["validation_loss"][index],
                    "validation_precision": history["validation_precision"][index],
                    "validation_recall": history["validation_recall"][index],
                    "validation_f1": history["validation_f1"][index],
                }
            )


def save_training_plots(history, loss_plot_path, metrics_plot_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "\nAVISO: matplotlib não está instalado. "
            "Instale com: pip install matplotlib"
        )
        return

    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["validation_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SpectraSceneNet - Training Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["validation_precision"], label="Validation precision")
    plt.plot(epochs, history["validation_recall"], label="Validation recall")
    plt.plot(epochs, history["validation_f1"], label="Validation F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("SpectraSceneNet - Validation Metrics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(metrics_plot_path)
    plt.close()


def print_training_header(
    dataset_path,
    output_dir,
    config,
    device,
    train_size,
    validation_size,
):
    print("=" * 80)
    print("TREINANDO SPECTRA SCENE NET")
    print("=" * 80)
    print("Dataset:", dataset_path)
    print("Output:", output_dir)
    print("Device:", device)
    print("Labels:", len(LABELS))
    print("Train samples:", train_size)
    print("Validation samples:", validation_size)
    print("Backbone:", config.backbone_name)
    print("Pretrained:", config.pretrained)
    print("Freeze backbone:", config.freeze_backbone)
    print("Epochs:", config.epochs)
    print("Batch size:", config.batch_size)
    print("Image size:", config.image_size)
    print("Learning rate:", config.learning_rate)
    print("Weight decay:", config.weight_decay)
    print("Dropout:", config.dropout_rate)
    print("Threshold:", config.threshold)
    print("=" * 80)


def print_epoch_result(
    epoch,
    total_epochs,
    train_loss,
    validation_loss,
    validation_metrics,
):
    print(
        f"Epoch {epoch + 1}/{total_epochs} "
        f"- train_loss: {train_loss:.4f} "
        f"- validation_loss: {validation_loss:.4f} "
        f"- precision: {validation_metrics['precision']:.4f} "
        f"- recall: {validation_metrics['recall']:.4f} "
        f"- f1: {validation_metrics['f1']:.4f}"
    )


def build_config_from_args(args):
    return SceneTrainingConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout_rate,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        backbone_name=args.backbone,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
        threshold=args.threshold,
        save_plots=not args.no_plots,
        cross_validation=args.cross_validation,
        folds=args.folds,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Treina a SpectraSceneNet."
    )

    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Caminho para o CSV de treino.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Pasta onde o modelo será salvo.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--validation-ratio",
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
        "--threshold",
        type=float,
        default=0.4,
        help="Threshold usado para calcular precision/recall/F1.",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Desativa geração dos gráficos de treinamento.",
    )

    parser.add_argument(
        "--cross-validation",
        action="store_true",
        help="Ativa treino com K-Fold cross-validation.",
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Número de folds para cross-validation.",
    )

    args = parser.parse_args()

    config = build_config_from_args(args)

    result = train_scene_model(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        config=config,
    )

    print("\nTreinamento finalizado.")
    print(json.dumps(result, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()