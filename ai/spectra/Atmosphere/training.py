from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import csv
import json

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from ai.spectra.Atmosphere.labels import LABELS
from ai.spectra.Atmosphere.model import SpectraAtmosphereNet
from ai.spectra.data.dataset import SpectraImageDataset
from ai.spectra.data.transforms import (
    get_train_transforms,
    get_validation_transforms,
)


@dataclass
class AtmosphereTrainingConfig:
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
    use_pos_weight: bool = True
    max_pos_weight: float = 8.0


def train_atmosphere_model(
    dataset_path,
    output_dir,
    config=None,
):
    config = config or AtmosphereTrainingConfig()

    if config.cross_validation:
        return train_cross_validation(
            dataset_path=dataset_path,
            output_dir=output_dir,
            config=config,
        )

    return train_single_split(
        dataset_path=dataset_path,
        output_dir=output_dir,
        config=config,
    )


def train_single_split(
    dataset_path,
    output_dir,
    config,
):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        output_dir
        / "atmosphere_net_best.pt"
    )

    history_json_path = (
        output_dir
        / "training_history.json"
    )

    history_csv_path = (
        output_dir
        / "training_history.csv"
    )

    loss_plot_path = (
        output_dir
        / "training_loss.png"
    )

    metrics_plot_path = (
        output_dir
        / "training_metrics.png"
    )

    test_metrics_path = (
        output_dir
        / "test_metrics.json"
    )

    train_data = SpectraImageDataset(
        csv_path=dataset_path,
        transform=get_train_transforms(
            config.image_size
        ),
        label_columns=LABELS,
    )

    evaluation_data = SpectraImageDataset(
        csv_path=dataset_path,
        transform=get_validation_transforms(
            config.image_size
        ),
        label_columns=LABELS,
    )

    (
        train_indices,
        validation_indices,
        test_indices,
    ) = create_dataset_split_indices(
        dataset_path=dataset_path,
    )

    validate_dataset_alignment(
        dataset_size=len(train_data),
        dataframe_size=(
            len(train_indices)
            + len(validation_indices)
            + len(test_indices)
        ),
    )

    train_dataset = Subset(
        train_data,
        train_indices,
    )

    validation_dataset = Subset(
        evaluation_data,
        validation_indices,
    )

    test_dataset = Subset(
        evaluation_data,
        test_indices,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    test_loader = None

    if len(test_dataset) > 0:
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

    model = SpectraAtmosphereNet(
        output_size=len(LABELS),
        image_size=config.image_size,
        dropout_rate=config.dropout_rate,
        backbone_name=config.backbone_name,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters(),
        ),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    criterion = build_criterion(
        dataset=train_dataset,
        label_count=len(LABELS),
        device=device,
        config=config,
    )

    best_validation_loss = float("inf")
    best_validation_f1 = -1.0
    best_epoch = 0

    history = create_empty_history()

    print_training_header(
        dataset_path=dataset_path,
        output_dir=output_dir,
        config=config,
        device=device,
        train_size=len(train_dataset),
        validation_size=len(
            validation_dataset
        ),
        test_size=len(test_dataset),
    )

    for epoch in range(config.epochs):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        (
            validation_loss,
            validation_metrics,
        ) = evaluate(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
            threshold=config.threshold,
        )

        update_history(
            history=history,
            train_loss=train_loss,
            validation_loss=validation_loss,
            validation_metrics=(
                validation_metrics
            ),
        )

        print_epoch_result(
            epoch=epoch,
            total_epochs=config.epochs,
            train_loss=train_loss,
            validation_loss=validation_loss,
            validation_metrics=(
                validation_metrics
            ),
        )

        validation_f1 = (
            validation_metrics["f1"]
        )

        should_save = (
            validation_f1
            > best_validation_f1
            or (
                validation_f1
                == best_validation_f1
                and validation_loss
                < best_validation_loss
            )
        )

        if should_save:
            best_validation_f1 = (
                validation_f1
            )

            best_validation_loss = (
                validation_loss
            )

            best_epoch = epoch + 1

            save_checkpoint(
                checkpoint_path=(
                    checkpoint_path
                ),
                model=model,
                config=config,
                best_validation_loss=(
                    best_validation_loss
                ),
                best_validation_f1=(
                    best_validation_f1
                ),
                best_epoch=best_epoch,
            )

            print(
                "Novo melhor modelo salvo:",
                checkpoint_path,
            )

        save_training_history_json(
            history=history,
            output_path=(
                history_json_path
            ),
        )

        save_training_history_csv(
            history=history,
            output_path=(
                history_csv_path
            ),
        )

    if config.save_plots:
        save_training_plots(
            history=history,
            loss_plot_path=(
                loss_plot_path
            ),
            metrics_plot_path=(
                metrics_plot_path
            ),
        )

    print(
        "\nCarregando melhor checkpoint..."
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    (
        final_validation_loss,
        final_validation_metrics,
    ) = evaluate(
        model=model,
        loader=validation_loader,
        criterion=criterion,
        device=device,
        threshold=config.threshold,
    )

    print_per_label_metrics(
        metrics=final_validation_metrics,
        title=(
            "VALIDATION SET "
            "- MÉTRICAS POR LABEL"
        ),
    )

    test_loss = None
    test_metrics = None

    if test_loader is not None:
        (
            test_loss,
            test_metrics,
        ) = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            threshold=config.threshold,
        )

        print(
            "\n"
            + "=" * 80
        )

        print(
            "AVALIAÇÃO FINAL "
            "- TEST SET"
        )

        print(
            "=" * 80
        )

        print(
            "Test loss:",
            f"{test_loss:.4f}",
        )

        print(
            "Precision:",
            f"{test_metrics['precision']:.4f}",
        )

        print(
            "Recall:",
            f"{test_metrics['recall']:.4f}",
        )

        print(
            "F1:",
            f"{test_metrics['f1']:.4f}",
        )

        print_per_label_metrics(
            metrics=test_metrics,
            title=(
                "TEST SET "
                "- MÉTRICAS POR LABEL"
            ),
        )

        save_test_metrics(
            output_path=(
                test_metrics_path
            ),
            test_loss=test_loss,
            test_metrics=test_metrics,
            threshold=config.threshold,
            best_epoch=best_epoch,
            best_validation_loss=(
                best_validation_loss
            ),
            best_validation_f1=(
                best_validation_f1
            ),
        )

    print(
        "\nTreinamento finalizado."
    )

    print(
        "Melhor modelo salvo em:",
        checkpoint_path,
    )

    print(
        "Melhor época:",
        best_epoch,
    )

    print(
        "Melhor loss de validação:",
        best_validation_loss,
    )

    print(
        "Melhor F1 de validação:",
        best_validation_f1,
    )

    print(
        "Histórico JSON:",
        history_json_path,
    )

    print(
        "Histórico CSV:",
        history_csv_path,
    )

    if test_metrics is not None:
        print(
            "Métricas de teste:",
            test_metrics_path,
        )

    if config.save_plots:
        print(
            "Gráfico de loss:",
            loss_plot_path,
        )

        print(
            "Gráfico de métricas:",
            metrics_plot_path,
        )

    return {
        "best_model_path": str(
            checkpoint_path
        ),
        "best_epoch": best_epoch,
        "best_validation_loss": (
            best_validation_loss
        ),
        "best_validation_f1": (
            best_validation_f1
        ),
        "final_validation_loss": (
            final_validation_loss
        ),
        "final_validation_precision": (
            final_validation_metrics[
                "precision"
            ]
        ),
        "final_validation_recall": (
            final_validation_metrics[
                "recall"
            ]
        ),
        "final_validation_f1": (
            final_validation_metrics[
                "f1"
            ]
        ),
        "final_validation_per_label": (
            final_validation_metrics[
                "per_label"
            ]
        ),
        "test_loss": test_loss,
        "test_precision": (
            test_metrics["precision"]
            if test_metrics
            else None
        ),
        "test_recall": (
            test_metrics["recall"]
            if test_metrics
            else None
        ),
        "test_f1": (
            test_metrics["f1"]
            if test_metrics
            else None
        ),
        "test_per_label": (
            test_metrics[
                "per_label"
            ]
            if test_metrics
            else None
        ),
        "history_json_path": str(
            history_json_path
        ),
        "history_csv_path": str(
            history_csv_path
        ),
        "test_metrics_path": (
            str(test_metrics_path)
            if test_metrics
            else None
        ),
        "loss_plot_path": (
            str(loss_plot_path)
            if config.save_plots
            else None
        ),
        "metrics_plot_path": (
            str(metrics_plot_path)
            if config.save_plots
            else None
        ),
    }


def train_cross_validation(
    dataset_path,
    output_dir,
    config,
):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_data = SpectraImageDataset(
        csv_path=dataset_path,
        transform=get_train_transforms(
            config.image_size
        ),
        label_columns=LABELS,
    )

    validation_data = SpectraImageDataset(
        csv_path=dataset_path,
        transform=get_validation_transforms(
            config.image_size
        ),
        label_columns=LABELS,
    )

    (
        official_train_indices,
        official_validation_indices,
        test_indices,
    ) = create_dataset_split_indices(
        dataset_path=dataset_path,
    )

    development_indices = (
        official_train_indices
        + official_validation_indices
    )

    if len(development_indices) < config.folds:
        raise ValueError(
            "O número de folds não pode ser "
            "maior que o conjunto de desenvolvimento. "
            f"Amostras: {len(development_indices)}, "
            f"folds: {config.folds}"
        )

    fold_indices = create_k_fold_indices(
        indices=development_indices,
        folds=config.folds,
        seed=config.seed,
    )

    test_dataset = Subset(
        validation_data,
        test_indices,
    )

    test_loader = None

    if test_indices:
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

    fold_results = []

    best_global_f1 = -1.0
    best_global_loss = float("inf")
    best_global_model_path = None

    print("=" * 80)
    print(
        "TREINANDO SPECTRA ATMOSPHERE NET "
        "- CROSS VALIDATION"
    )
    print("=" * 80)
    print("Dataset:", dataset_path)
    print("Output:", output_dir)
    print("Device:", device)
    print("Labels:", len(LABELS))
    print(
        "Development samples:",
        len(development_indices),
    )
    print(
        "Test samples reservadas:",
        len(test_indices),
    )
    print("Folds:", config.folds)
    print("Epochs por fold:", config.epochs)
    print(
        "Backbone:",
        config.backbone_name,
    )
    print(
        "Freeze backbone:",
        config.freeze_backbone,
    )
    print(
        "Threshold:",
        config.threshold,
    )
    print("=" * 80)

    for fold_index in range(
        config.folds
    ):
        fold_number = fold_index + 1

        fold_output_dir = (
            output_dir
            / f"fold_{fold_number}"
        )

        fold_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        validation_indices = (
            fold_indices[fold_index]
        )

        train_indices = []

        for index, indices in enumerate(
            fold_indices
        ):
            if index == fold_index:
                continue

            train_indices.extend(
                indices
            )

        train_dataset = Subset(
            train_data,
            train_indices,
        )

        validation_dataset = Subset(
            validation_data,
            validation_indices,
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )

        validation_loader = DataLoader(
            dataset=validation_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

        model = SpectraAtmosphereNet(
            output_size=len(LABELS),
            image_size=config.image_size,
            dropout_rate=(
                config.dropout_rate
            ),
            backbone_name=(
                config.backbone_name
            ),
            pretrained=config.pretrained,
            freeze_backbone=(
                config.freeze_backbone
            ),
        ).to(device)

        optimizer = torch.optim.AdamW(
            filter(
                lambda parameter:
                parameter.requires_grad,
                model.parameters(),
            ),
            lr=config.learning_rate,
            weight_decay=(
                config.weight_decay
            ),
        )

        criterion = build_criterion(
            dataset=train_dataset,
            label_count=len(LABELS),
            device=device,
            config=config,
        )

        fold_checkpoint_path = (
            fold_output_dir
            / "atmosphere_net_best.pt"
        )

        fold_history_json_path = (
            fold_output_dir
            / "training_history.json"
        )

        fold_history_csv_path = (
            fold_output_dir
            / "training_history.csv"
        )

        fold_loss_plot_path = (
            fold_output_dir
            / "training_loss.png"
        )

        fold_metrics_plot_path = (
            fold_output_dir
            / "training_metrics.png"
        )

        best_fold_loss = float(
            "inf"
        )

        best_fold_f1 = -1.0
        best_fold_epoch = 0

        history = (
            create_empty_history()
        )

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"FOLD "
            f"{fold_number}/"
            f"{config.folds}"
        )

        print("=" * 80)

        print(
            "Train samples:",
            len(train_dataset),
        )

        print(
            "Validation samples:",
            len(validation_dataset),
        )

        for epoch in range(
            config.epochs
        ):
            train_loss = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
            )

            (
                validation_loss,
                validation_metrics,
            ) = evaluate(
                model=model,
                loader=validation_loader,
                criterion=criterion,
                device=device,
                threshold=(
                    config.threshold
                ),
            )

            validation_f1 = (
                validation_metrics[
                    "f1"
                ]
            )

            update_history(
                history=history,
                train_loss=train_loss,
                validation_loss=(
                    validation_loss
                ),
                validation_metrics=(
                    validation_metrics
                ),
            )

            print(
                f"Fold "
                f"{fold_number}/"
                f"{config.folds} "
                f"- Epoch "
                f"{epoch + 1}/"
                f"{config.epochs} "
                f"- train_loss: "
                f"{train_loss:.4f} "
                f"- validation_loss: "
                f"{validation_loss:.4f} "
                f"- precision: "
                f"{validation_metrics['precision']:.4f} "
                f"- recall: "
                f"{validation_metrics['recall']:.4f} "
                f"- f1: "
                f"{validation_f1:.4f}"
            )

            should_save = (
                validation_f1
                > best_fold_f1
                or (
                    validation_f1
                    == best_fold_f1
                    and validation_loss
                    < best_fold_loss
                )
            )

            if should_save:
                best_fold_loss = (
                    validation_loss
                )

                best_fold_f1 = (
                    validation_f1
                )

                best_fold_epoch = (
                    epoch + 1
                )

                save_checkpoint(
                    checkpoint_path=(
                        fold_checkpoint_path
                    ),
                    model=model,
                    config=config,
                    best_validation_loss=(
                        best_fold_loss
                    ),
                    best_validation_f1=(
                        best_fold_f1
                    ),
                    best_epoch=(
                        best_fold_epoch
                    ),
                )

                print(
                    "Novo melhor modelo "
                    "do fold salvo:",
                    fold_checkpoint_path,
                )

        save_training_history_json(
            history=history,
            output_path=(
                fold_history_json_path
            ),
        )

        save_training_history_csv(
            history=history,
            output_path=(
                fold_history_csv_path
            ),
        )

        if config.save_plots:
            save_training_plots(
                history=history,
                loss_plot_path=(
                    fold_loss_plot_path
                ),
                metrics_plot_path=(
                    fold_metrics_plot_path
                ),
            )

        fold_result = {
            "fold": fold_number,
            "best_epoch": (
                best_fold_epoch
            ),
            "best_validation_loss": (
                best_fold_loss
            ),
            "best_validation_f1": (
                best_fold_f1
            ),
            "best_model_path": str(
                fold_checkpoint_path
            ),
            "history_json_path": str(
                fold_history_json_path
            ),
            "history_csv_path": str(
                fold_history_csv_path
            ),
            "loss_plot_path": (
                str(
                    fold_loss_plot_path
                )
                if config.save_plots
                else None
            ),
            "metrics_plot_path": (
                str(
                    fold_metrics_plot_path
                )
                if config.save_plots
                else None
            ),
        }

        fold_results.append(
            fold_result
        )

        is_best_global = (
            best_fold_f1
            > best_global_f1
            or (
                best_fold_f1
                == best_global_f1
                and best_fold_loss
                < best_global_loss
            )
        )

        if is_best_global:
            best_global_f1 = (
                best_fold_f1
            )

            best_global_loss = (
                best_fold_loss
            )

            best_global_model_path = (
                fold_checkpoint_path
            )

    summary = (
        build_cross_validation_summary(
            fold_results=fold_results,
            best_global_model_path=(
                best_global_model_path
            ),
        )
    )

    if (
        test_loader is not None
        and best_global_model_path
        is not None
    ):
        checkpoint = torch.load(
            best_global_model_path,
            map_location=device,
        )

        model = SpectraAtmosphereNet(
            output_size=len(LABELS),
            image_size=config.image_size,
            dropout_rate=(
                config.dropout_rate
            ),
            backbone_name=(
                config.backbone_name
            ),
            pretrained=False,
            freeze_backbone=(
                config.freeze_backbone
            ),
        ).to(device)

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        test_criterion = (
            nn.BCEWithLogitsLoss()
        )

        (
            test_loss,
            test_metrics,
        ) = evaluate(
            model=model,
            loader=test_loader,
            criterion=test_criterion,
            device=device,
            threshold=config.threshold,
        )

        summary[
            "test_loss"
        ] = test_loss

        summary[
            "test_precision"
        ] = test_metrics[
            "precision"
        ]

        summary[
            "test_recall"
        ] = test_metrics[
            "recall"
        ]

        summary[
            "test_f1"
        ] = test_metrics[
            "f1"
        ]

        summary[
            "test_per_label"
        ] = test_metrics[
            "per_label"
        ]

        print_per_label_metrics(
            metrics=test_metrics,
            title=(
                "CROSS VALIDATION "
                "- TEST SET"
            ),
        )

    summary_path = (
        output_dir
        / "cross_validation_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "CROSS VALIDATION FINALIZADO"
    )

    print("=" * 80)

    print(
        "Folds:",
        config.folds,
    )

    print(
        "Média validation loss:",
        f"{summary['mean_validation_loss']:.4f}",
    )

    print(
        "Média validation F1:",
        f"{summary['mean_validation_f1']:.4f}",
    )

    print(
        "Melhor modelo global:",
        summary[
            "best_global_model_path"
        ],
    )

    print(
        "Resumo salvo em:",
        summary_path,
    )

    return summary


def build_criterion(
    dataset,
    label_count,
    device,
    config,
):
    if not config.use_pos_weight:
        return nn.BCEWithLogitsLoss()

    pos_weight = compute_pos_weight(
        dataset=dataset,
        label_count=label_count,
        device=device,
        max_pos_weight=(
            config.max_pos_weight
        ),
    )

    print_pos_weight(
        pos_weight=pos_weight,
    )

    return nn.BCEWithLogitsLoss(
        pos_weight=pos_weight,
    )


def compute_pos_weight(
    dataset,
    label_count,
    device,
    max_pos_weight=8.0,
):
    positives = torch.zeros(
        label_count
    )

    for _, labels in dataset:
        positives += labels.cpu()

    total = len(dataset)

    negatives = (
        total - positives
    )

    pos_weight = (
        negatives
        / torch.clamp(
            positives,
            min=1.0,
        )
    )

    pos_weight = torch.clamp(
        pos_weight,
        min=1.0,
        max=max_pos_weight,
    )

    return pos_weight.to(
        device
    )


def print_pos_weight(
    pos_weight,
):
    print(
        "\nPOS WEIGHT POR LABEL"
    )

    for label, weight in zip(
        LABELS,
        pos_weight.detach().cpu(),
    ):
        print(
            f"{label:<20}"
            f"{float(weight):.4f}"
        )


def create_dataset_split_indices(
    dataset_path,
):
    dataframe = pd.read_csv(
        dataset_path
    )

    if (
        "generated_split"
        not in dataframe.columns
    ):
        raise ValueError(
            "O dataset não possui a coluna "
            "'generated_split'."
        )

    splits = (
        dataframe[
            "generated_split"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    train_indices = dataframe.index[
        splits == "train"
    ].tolist()

    validation_indices = (
        dataframe.index[
            splits.isin(
                [
                    "validation",
                    "val",
                ]
            )
        ]
        .tolist()
    )

    test_indices = dataframe.index[
        splits == "test"
    ].tolist()

    unknown_indices = dataframe.index[
        ~splits.isin(
            [
                "train",
                "validation",
                "val",
                "test",
            ]
        )
    ].tolist()

    if unknown_indices:
        raise ValueError(
            "Existem amostras com "
            "generated_split inválido ou vazio. "
            f"Quantidade: "
            f"{len(unknown_indices)}"
        )

    if not train_indices:
        raise ValueError(
            "Nenhuma amostra de treino "
            "foi encontrada."
        )

    if not validation_indices:
        raise ValueError(
            "Nenhuma amostra de validação "
            "foi encontrada."
        )

    if not test_indices:
        print(
            "AVISO: nenhuma amostra de "
            "teste foi encontrada."
        )

    return (
        train_indices,
        validation_indices,
        test_indices,
    )


def validate_dataset_alignment(
    dataset_size,
    dataframe_size,
):
    if dataset_size != dataframe_size:
        raise ValueError(
            "O número de linhas do CSV "
            "não corresponde ao número "
            "de amostras carregadas pelo "
            "SpectraImageDataset. "
            f"Dataset: {dataset_size} | "
            f"CSV splits: {dataframe_size}"
        )


def create_k_fold_indices(
    indices,
    folds,
    seed,
):
    generator = (
        torch.Generator()
        .manual_seed(seed)
    )

    permutation = torch.randperm(
        len(indices),
        generator=generator,
    ).tolist()

    shuffled_indices = [
        indices[index]
        for index in permutation
    ]

    fold_sizes = []

    base_fold_size = (
        len(indices)
        // folds
    )

    remainder = (
        len(indices)
        % folds
    )

    for fold_index in range(
        folds
    ):
        fold_size = (
            base_fold_size
        )

        if fold_index < remainder:
            fold_size += 1

        fold_sizes.append(
            fold_size
        )

    fold_indices = []
    current_index = 0

    for fold_size in fold_sizes:
        fold = shuffled_indices[
            current_index:
            current_index
            + fold_size
        ]

        fold_indices.append(
            fold
        )

        current_index += (
            fold_size
        )

    return fold_indices


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
):
    is_training = (
        optimizer is not None
    )

    if is_training:
        model.train()

    else:
        model.eval()

    total_loss = 0.0

    with torch.set_grad_enabled(
        is_training
    ):
        for images, labels in loader:
            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            if is_training:
                optimizer.zero_grad()

            logits = model(
                images
            )

            loss = criterion(
                logits,
                labels,
            )

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += (
                loss.item()
            )

    return (
        total_loss
        / max(
            1,
            len(loader),
        )
    )


def evaluate(
    model,
    loader,
    criterion,
    device,
    threshold,
):
    model.eval()

    total_loss = 0.0

    true_positives = torch.zeros(
        len(LABELS),
        dtype=torch.float64,
    )

    false_positives = torch.zeros(
        len(LABELS),
        dtype=torch.float64,
    )

    false_negatives = torch.zeros(
        len(LABELS),
        dtype=torch.float64,
    )

    true_negatives = torch.zeros(
        len(LABELS),
        dtype=torch.float64,
    )

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            logits = model(
                images
            )

            loss = criterion(
                logits,
                labels,
            )

            probabilities = (
                torch.sigmoid(
                    logits
                )
            )

            predictions = (
                probabilities
                >= threshold
            ).float()

            total_loss += (
                loss.item()
            )

            batch_tp = (
                (
                    (predictions == 1)
                    & (labels == 1)
                )
                .sum(dim=0)
                .cpu()
            )

            batch_fp = (
                (
                    (predictions == 1)
                    & (labels == 0)
                )
                .sum(dim=0)
                .cpu()
            )

            batch_fn = (
                (
                    (predictions == 0)
                    & (labels == 1)
                )
                .sum(dim=0)
                .cpu()
            )

            batch_tn = (
                (
                    (predictions == 0)
                    & (labels == 0)
                )
                .sum(dim=0)
                .cpu()
            )

            true_positives += (
                batch_tp
            )

            false_positives += (
                batch_fp
            )

            false_negatives += (
                batch_fn
            )

            true_negatives += (
                batch_tn
            )

    total_tp = (
        true_positives
        .sum()
        .item()
    )

    total_fp = (
        false_positives
        .sum()
        .item()
    )

    total_fn = (
        false_negatives
        .sum()
        .item()
    )

    precision = (
        total_tp
        / max(
            1.0,
            total_tp
            + total_fp,
        )
    )

    recall = (
        total_tp
        / max(
            1.0,
            total_tp
            + total_fn,
        )
    )

    f1 = (
        2
        * precision
        * recall
        / max(
            1e-8,
            precision
            + recall,
        )
    )

    per_label = {}

    label_f1_scores = []

    for index, label in enumerate(
        LABELS
    ):
        tp = (
            true_positives[
                index
            ].item()
        )

        fp = false_positives[index].item()

        fn = (
            false_negatives[
                index
            ].item()
        )

        tn = (
            true_negatives[
                index
            ].item()
        )

        label_precision = (
            tp
            / max(
                1.0,
                tp + fp,
            )
        )

        label_recall = (
            tp
            / max(
                1.0,
                tp + fn,
            )
        )

        label_f1 = (
            2
            * label_precision
            * label_recall
            / max(
                1e-8,
                label_precision
                + label_recall,
            )
        )

        label_accuracy = (
            (tp + tn)
            / max(
                1.0,
                tp + fp + fn + tn,
            )
        )

        label_f1_scores.append(
            label_f1
        )

        per_label[label] = {
            "precision": (
                label_precision
            ),
            "recall": (
                label_recall
            ),
            "f1": label_f1,
            "accuracy": (
                label_accuracy
            ),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }

    macro_f1 = (
        sum(label_f1_scores)
        / max(
            1,
            len(label_f1_scores),
        )
    )

    average_loss = (
        total_loss
        / max(
            1,
            len(loader),
        )
    )

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "per_label": per_label,
    }

    return (
        average_loss,
        metrics,
    )


def create_empty_history():
    return {
        "train_loss": [],
        "validation_loss": [],
        "validation_precision": [],
        "validation_recall": [],
        "validation_f1": [],
        "validation_macro_f1": [],
    }


def update_history(
    history,
    train_loss,
    validation_loss,
    validation_metrics,
):
    history[
        "train_loss"
    ].append(
        train_loss
    )

    history[
        "validation_loss"
    ].append(
        validation_loss
    )

    history[
        "validation_precision"
    ].append(
        validation_metrics[
            "precision"
        ]
    )

    history[
        "validation_recall"
    ].append(
        validation_metrics[
            "recall"
        ]
    )

    history[
        "validation_f1"
    ].append(
        validation_metrics[
            "f1"
        ]
    )

    history[
        "validation_macro_f1"
    ].append(
        validation_metrics[
            "macro_f1"
        ]
    )


def save_checkpoint(
    checkpoint_path,
    model,
    config,
    best_validation_loss,
    best_validation_f1,
    best_epoch,
):
    torch.save(
        {
            "model_state_dict": (
                model.state_dict()
            ),
            "labels": LABELS,
            "task_name": (
                "atmosphere"
            ),
            "config": asdict(
                config
            ),
            "best_epoch": (
                best_epoch
            ),
            "best_validation_loss": (
                best_validation_loss
            ),
            "best_validation_f1": (
                best_validation_f1
            ),
        },
        checkpoint_path,
    )


def save_test_metrics(
    output_path,
    test_loss,
    test_metrics,
    threshold,
    best_epoch,
    best_validation_loss,
    best_validation_f1,
):
    result = {
        "threshold": threshold,
        "best_epoch": (
            best_epoch
        ),
        "best_validation_loss": (
            best_validation_loss
        ),
        "best_validation_f1": (
            best_validation_f1
        ),
        "test_loss": test_loss,
        "test_precision": (
            test_metrics[
                "precision"
            ]
        ),
        "test_recall": (
            test_metrics[
                "recall"
            ]
        ),
        "test_f1": (
            test_metrics[
                "f1"
            ]
        ),
        "test_macro_f1": (
            test_metrics[
                "macro_f1"
            ]
        ),
        "per_label": (
            test_metrics[
                "per_label"
            ]
        ),
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=4,
            ensure_ascii=False,
        )


def save_training_history_json(
    history,
    output_path,
):
    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history,
            file,
            indent=4,
            ensure_ascii=False,
        )


def save_training_history_csv(
    history,
    output_path,
):
    fieldnames = [
        "epoch",
        "train_loss",
        "validation_loss",
        "validation_precision",
        "validation_recall",
        "validation_f1",
        "validation_macro_f1",
    ]

    total_epochs = len(
        history["train_loss"]
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for index in range(
            total_epochs
        ):
            writer.writerow(
                {
                    "epoch": (
                        index + 1
                    ),
                    "train_loss": (
                        history[
                            "train_loss"
                        ][index]
                    ),
                    "validation_loss": (
                        history[
                            "validation_loss"
                        ][index]
                    ),
                    "validation_precision": (
                        history[
                            "validation_precision"
                        ][index]
                    ),
                    "validation_recall": (
                        history[
                            "validation_recall"
                        ][index]
                    ),
                    "validation_f1": (
                        history[
                            "validation_f1"
                        ][index]
                    ),
                    "validation_macro_f1": (
                        history[
                            "validation_macro_f1"
                        ][index]
                    ),
                }
            )


def save_training_plots(
    history,
    loss_plot_path,
    metrics_plot_path,
):
    try:
        import matplotlib.pyplot as plt

    except ImportError:
        print(
            "\nAVISO: matplotlib "
            "não está instalado. "
            "Instale com: "
            "pip install matplotlib"
        )

        return

    epochs = list(
        range(
            1,
            len(
                history[
                    "train_loss"
                ]
            )
            + 1,
        )
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        epochs,
        history[
            "train_loss"
        ],
        label="Train loss",
    )

    plt.plot(
        epochs,
        history[
            "validation_loss"
        ],
        label="Validation loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        "SpectraAtmosphereNet "
        "- Training Loss"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        loss_plot_path
    )
    plt.close()

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        epochs,
        history[
            "validation_precision"
        ],
        label=(
            "Validation precision"
        ),
    )

    plt.plot(
        epochs,
        history[
            "validation_recall"
        ],
        label=(
            "Validation recall"
        ),
    )

    plt.plot(
        epochs,
        history[
            "validation_f1"
        ],
        label="Validation F1",
    )

    plt.plot(
        epochs,
        history[
            "validation_macro_f1"
        ],
        label=(
            "Validation Macro F1"
        ),
    )

    plt.xlabel("Epoch")
    plt.ylabel("Score")

    plt.title(
        "SpectraAtmosphereNet "
        "- Validation Metrics"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        metrics_plot_path
    )

    plt.close()


def build_cross_validation_summary(
    fold_results,
    best_global_model_path,
):
    validation_losses = [
        result[
            "best_validation_loss"
        ]
        for result in fold_results
    ]

    validation_f1_scores = [
        result[
            "best_validation_f1"
        ]
        for result in fold_results
    ]

    mean_validation_loss = (
        sum(validation_losses)
        / max(
            1,
            len(validation_losses),
        )
    )

    mean_validation_f1 = (
        sum(validation_f1_scores)
        / max(
            1,
            len(validation_f1_scores),
        )
    )

    best_fold = max(
        fold_results,
        key=lambda result:
        result[
            "best_validation_f1"
        ],
    )

    return {
        "folds": len(
            fold_results
        ),
        "mean_validation_loss": (
            mean_validation_loss
        ),
        "mean_validation_f1": (
            mean_validation_f1
        ),
        "best_fold": (
            best_fold["fold"]
        ),
        "best_global_model_path": (
            str(
                best_global_model_path
            )
        ),
        "fold_results": (
            fold_results
        ),
    }


def print_training_header(
    dataset_path,
    output_dir,
    config,
    device,
    train_size,
    validation_size,
    test_size,
):
    print("=" * 80)

    print(
        "TREINANDO "
        "SPECTRA ATMOSPHERE NET"
    )

    print("=" * 80)

    print(
        "Dataset:",
        dataset_path,
    )

    print(
        "Output:",
        output_dir,
    )

    print(
        "Device:",
        device,
    )

    print(
        "Labels:",
        len(LABELS),
    )

    print(
        "\nDataset splits:"
    )

    print(
        "Train samples:",
        train_size,
    )

    print(
        "Validation samples:",
        validation_size,
    )

    print(
        "Test samples:",
        test_size,
    )

    print(
        "\nConfiguração:"
    )

    print(
        "Backbone:",
        config.backbone_name,
    )

    print(
        "Pretrained:",
        config.pretrained,
    )

    print(
        "Freeze backbone:",
        config.freeze_backbone,
    )

    print(
        "Epochs:",
        config.epochs,
    )

    print(
        "Batch size:",
        config.batch_size,
    )

    print(
        "Image size:",
        config.image_size,
    )

    print(
        "Learning rate:",
        config.learning_rate,
    )

    print(
        "Weight decay:",
        config.weight_decay,
    )

    print(
        "Dropout:",
        config.dropout_rate,
    )

    print(
        "Threshold:",
        config.threshold,
    )

    print(
        "Use pos weight:",
        config.use_pos_weight,
    )

    print("=" * 80)


def print_epoch_result(
    epoch,
    total_epochs,
    train_loss,
    validation_loss,
    validation_metrics,
):
    print(
        f"Epoch "
        f"{epoch + 1}/"
        f"{total_epochs} "
        f"- train_loss: "
        f"{train_loss:.4f} "
        f"- validation_loss: "
        f"{validation_loss:.4f} "
        f"- precision: "
        f"{validation_metrics['precision']:.4f} "
        f"- recall: "
        f"{validation_metrics['recall']:.4f} "
        f"- f1: "
        f"{validation_metrics['f1']:.4f} "
        f"- macro_f1: "
        f"{validation_metrics['macro_f1']:.4f}"
    )


def print_per_label_metrics(
    metrics,
    title="MÉTRICAS POR LABEL",
):
    print(
        "\n"
        + "=" * 88
    )

    print(title)

    print("=" * 88)

    print(
        f"{'Label':<20}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'Accuracy':>12}"
    )

    print("-" * 88)

    for label in LABELS:
        values = (
            metrics[
                "per_label"
            ][label]
        )

        print(
            f"{label:<20}"
            f"{values['precision']:>12.4f}"
            f"{values['recall']:>12.4f}"
            f"{values['f1']:>12.4f}"
            f"{values['accuracy']:>12.4f}"
        )

    print("-" * 88)

    print(
        f"{'MICRO':<20}"
        f"{metrics['precision']:>12.4f}"
        f"{metrics['recall']:>12.4f}"
        f"{metrics['f1']:>12.4f}"
    )

    print(
        f"{'MACRO F1':<20}"
        f"{'':>12}"
        f"{'':>12}"
        f"{metrics['macro_f1']:>12.4f}"
    )

    print("=" * 88)


def build_config_from_args(
    args,
):
    return AtmosphereTrainingConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=(
            args.learning_rate
        ),
        weight_decay=(
            args.weight_decay
        ),
        dropout_rate=(
            args.dropout_rate
        ),
        validation_ratio=(
            args.validation_ratio
        ),
        seed=args.seed,
        num_workers=(
            args.num_workers
        ),
        backbone_name=(
            args.backbone
        ),
        pretrained=(
            not args.no_pretrained
        ),
        freeze_backbone=(
            args.freeze_backbone
        ),
        threshold=args.threshold,
        save_plots=(
            not args.no_plots
        ),
        cross_validation=(
            args.cross_validation
        ),
        folds=args.folds,
        use_pos_weight=(
            not args.no_pos_weight
        ),
        max_pos_weight=(
            args.max_pos_weight
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Treina a "
            "SpectraAtmosphereNet."
        )
    )

    parser.add_argument(
        "--dataset-path",
        required=True,
        help=(
            "Caminho para "
            "o CSV de treino."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Pasta onde o modelo "
            "será salvo."
        ),
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
        help=(
            "Mantido por compatibilidade. "
            "Não é usado quando o CSV "
            "possui generated_split."
        ),
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
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
        ],
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
        help=(
            "Threshold usado para "
            "precision/recall/F1."
        ),
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help=(
            "Desativa geração dos "
            "gráficos de treinamento."
        ),
    )

    parser.add_argument(
        "--cross-validation",
        action="store_true",
        help=(
            "Ativa K-Fold apenas sobre "
            "train + validation. "
            "O test continua reservado."
        ),
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--no-pos-weight",
        action="store_true",
        help=(
            "Desativa pos_weight no "
            "BCEWithLogitsLoss."
        ),
    )

    parser.add_argument(
        "--max-pos-weight",
        type=float,
        default=8.0,
    )

    args = parser.parse_args()

    config = (
        build_config_from_args(
            args
        )
    )

    result = (
        train_atmosphere_model(
            dataset_path=(
                args.dataset_path
            ),
            output_dir=(
                args.output_dir
            ),
            config=config,
        )
    )

    print(
        "\nTreinamento finalizado."
    )

    print(
        json.dumps(
            result,
            indent=4,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()