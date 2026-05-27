from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import argparse

from ai.spectra.data.dataset import SpectraImageDataset
from ai.spectra.data.transforms import (
    get_train_transforms,
    get_validation_transforms,
    get_test_transforms,
)
from ai.spectra.labels.label_sets import get_labels_for_task
from ai.spectra.models.spectra_scene_net import SpectraSceneNet
from ai.spectra.models.spectra_person_net import SpectraPersonNet
from ai.spectra.models.spectra_object_net import SpectraObjectNet
from ai.spectra.training.config import SpectraTrainingConfig
from ai.spectra.training.history import TrainingHistory
from ai.spectra.training.metrics import (
    calculate_multilabel_metrics,
    calculate_loss_average,
)
from ai.spectra.training.plots import plot_training_history


def create_data_splits(dataset_size, config):
    """
    Cria índices de treino, validação e teste.

    Usa seed fixa para os resultados serem reproduzíveis.
    """
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    indices = torch.randperm(dataset_size, generator=generator).tolist()

    train_size = int(dataset_size * config.train_ratio)
    validation_size = int(dataset_size * config.validation_ratio)

    train_indices = indices[:train_size]
    validation_indices = indices[train_size:train_size + validation_size]
    test_indices = indices[train_size + validation_size:]

    return train_indices, validation_indices, test_indices


def create_dataloaders(dataset_path, config):
    """
    Cria DataLoaders com transforms corretos.

    Importante:
    - treino usa data augmentation
    - validação/teste não usam augmentation
    """
    train_base_dataset = SpectraImageDataset(
        csv_path=dataset_path,
        transform=get_train_transforms(config.image_size),
        task_name=config.task_name,
    )

    validation_base_dataset = SpectraImageDataset(
        csv_path=dataset_path,
        transform=get_validation_transforms(config.image_size),
        task_name=config.task_name,
    )

    test_base_dataset = SpectraImageDataset(
        csv_path=dataset_path,
        transform=get_test_transforms(config.image_size),
        task_name=config.task_name,
    )

    dataset_size = len(train_base_dataset)

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

def create_model_for_task(task_name, output_size, config):
    """
    Cria o modelo correto de acordo com a task.
    """
    if task_name == "scene":
        return SpectraSceneNet(
            output_size=output_size,
            image_size=config.image_size,
            dropout_rate=config.dropout_rate,
            backbone_name=config.backbone_name,
            pretrained=config.pretrained,
            freeze_backbone=config.freeze_backbone,
        )

    if task_name == "person":
        return SpectraPersonNet(
            output_size=output_size,
            image_size=config.image_size,
            dropout_rate=config.dropout_rate,
            backbone_name=config.backbone_name,
            pretrained=config.pretrained,
            freeze_backbone=config.freeze_backbone,
        )

    if task_name == "object":
        return SpectraObjectNet(
            output_size=output_size,
            image_size=config.image_size,
            dropout_rate=config.dropout_rate,
            backbone_name=config.backbone_name,
            pretrained=config.pretrained,
            freeze_backbone=config.freeze_backbone,
        )

    raise ValueError("Task não suportada: {}".format(task_name))

def create_optimizer(model, config):
    """
    Cria o otimizador com base na configuração.
    """
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

    raise ValueError("Otimizador não suportado: {}".format(config.optimizer_name))


def run_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
    threshold,
    training,
):
    """
    Executa uma época de treino ou avaliação.
    """
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


def save_model(model, output_path, config, validation_result, label_columns):
    """
    Salva o modelo treinado.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "labels": label_columns,
            "task_name": config.task_name,
            "config": config.__dict__,
            "validation_result": validation_result,
        },
        output_path,
    )


def save_test_result(test_result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame([test_result])
    dataframe.to_csv(output_path, index=False)


def train_spectra_model(
    dataset_path="data/datasets/spectra_auto_labels.csv",
    output_dir="data/models/spectra",
    config=None,
):
    """
    Treina a SpectraVisionNet.

    Retorna um dicionário com:
    - caminho do melhor modelo
    - melhor F1 de validação
    - resultado de teste
    - histórico
    """
    config = config or SpectraTrainingConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, validation_loader, test_loader = create_dataloaders(
        dataset_path=dataset_path,
        config=config,
    )

    label_columns = get_labels_for_task(config.task_name)

    model = create_model_for_task(
        task_name=config.task_name,
        output_size=len(label_columns),
        config=config,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = create_optimizer(model, config)

    history = TrainingHistory()

    best_validation_f1 = -1.0
    best_model_path = output_dir / "{}_net_best.pt".format(config.task_name)

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
                label_columns=label_columns,
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

DEFAULT_DATASET_PATHS = {
    "scene": "data/datasets/spectra_scene_labels.csv",
    "object": "data/datasets/spectra_object_labels.csv",
    "person": "data/datasets/spectra_person_labels.csv",
}


DEFAULT_OUTPUT_DIRS = {
    "scene": "data/models/spectra_scene",
    "object": "data/models/spectra_object",
    "person": "data/models/spectra_person",
}


def build_training_config_from_args(args):
    """
    Monta a configuração de treino a partir dos argumentos de linha de comando.
    """
    return SpectraTrainingConfig(
        task_name=args.task,

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


def resolve_dataset_path(args):
    """
    Define o dataset usado no treino.

    Se --dataset-path for passado, usa ele.
    Caso contrário, usa o padrão da task.
    """
    if args.dataset_path is not None:
        return args.dataset_path

    return DEFAULT_DATASET_PATHS[args.task]


def resolve_output_dir(args):
    """
    Define a pasta de saída do modelo.

    Se --output-dir for passado, usa ele.
    Caso contrário, usa o padrão da task.
    """
    if args.output_dir is not None:
        return args.output_dir

    return DEFAULT_OUTPUT_DIRS[args.task]


def main():
    parser = argparse.ArgumentParser(
        description="Treina um dos submodelos da Spectra: scene, object ou person."
    )

    parser.add_argument(
        "--task",
        choices=["scene", "object", "person"],
        required=True,
        help="Submodelo da Spectra que será treinado.",
    )

    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Caminho do CSV de treino. Se omitido, usa o padrão da task.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Pasta onde o modelo treinado será salvo. Se omitido, usa o padrão da task.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Quantidade de épocas de treino.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Tamanho do batch.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0003,
        help="Taxa de aprendizado.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
        help="Weight decay do otimizador.",
    )

    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.3,
        help="Taxa de dropout.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold usado nas métricas multilabel.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Tamanho da imagem de entrada.",
    )

    parser.add_argument(
        "--optimizer",
        choices=["adam", "adamw", "sgd"],
        default="adamw",
        help="Otimizador usado no treino.",
    )

    parser.add_argument(
        "--backbone",
        choices=["resnet18", "resnet34", "resnet50"],
        default="resnet18",
        help="Backbone visual usado pelo modelo.",
    )

    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Congela o backbone e treina apenas a cabeça final.",
    )

    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Não usa pesos pré-treinados no backbone.",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Proporção do dataset usada para treino.",
    )

    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.15,
        help="Proporção do dataset usada para validação.",
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Proporção do dataset usada para teste.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed usada para divisão do dataset.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Número de workers do DataLoader.",
    )

    args = parser.parse_args()

    config = build_training_config_from_args(args)

    dataset_path = resolve_dataset_path(args)
    output_dir = resolve_output_dir(args)

    print("\nIniciando treino da Spectra")
    print("Task:", config.task_name)
    print("Dataset:", dataset_path)
    print("Output:", output_dir)
    print("Backbone:", config.backbone_name)
    print("Pretrained:", config.pretrained)
    print("Freeze backbone:", config.freeze_backbone)
    print("Epochs:", config.epochs)
    print("Batch size:", config.batch_size)
    print("Learning rate:", config.learning_rate)
    print()

    train_spectra_model(
        dataset_path=dataset_path,
        output_dir=output_dir,
        config=config,
    )


if __name__ == "__main__":
    main()