from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, Subset

from ai.spectra.data.dataset import SpectraImageDataset
from ai.spectra.data.transforms import (
    get_train_transforms,
    get_validation_transforms,
)
from ai.spectra.labels.label_sets import SPECTRA_LABELS
from ai.spectra.models.spectra_vision_net import SpectraVisionNet
from ai.spectra.training.config import SpectraTrainingConfig
from ai.spectra.training.train import (
    create_optimizer,
    run_epoch,
)


def run_cross_validation(
    dataset_path="data/datasets/spectra_labels.csv",
    output_dir="data/models/spectra_cross_validation",
    config=None,
    folds=5,
):
    config = config or SpectraTrainingConfig(epochs=10)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_dataset = SpectraImageDataset(
        csv_path=dataset_path,
        transform=get_validation_transforms(config.image_size),
    )

    dataset_size = len(base_dataset)
    indices = list(range(dataset_size))

    if dataset_size < folds:
        raise ValueError(
            "Dataset tem menos amostras ({}) do que folds ({}).".format(
                dataset_size,
                folds,
            )
        )

    kfold = KFold(
        n_splits=folds,
        shuffle=True,
        random_state=config.seed,
    )

    fold_results = []

    for fold_index, (train_indices, validation_indices) in enumerate(kfold.split(indices)):
        print("\nFold {}/{}".format(fold_index + 1, folds))

        train_dataset_with_transforms = SpectraImageDataset(
            csv_path=dataset_path,
            transform=get_train_transforms(config.image_size),
        )

        validation_dataset_with_transforms = SpectraImageDataset(
            csv_path=dataset_path,
            transform=get_validation_transforms(config.image_size),
        )

        train_subset = Subset(
            train_dataset_with_transforms,
            train_indices,
        )

        validation_subset = Subset(
            validation_dataset_with_transforms,
            validation_indices,
        )

        train_loader = DataLoader(
            train_subset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )

        validation_loader = DataLoader(
            validation_subset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

        model = SpectraVisionNet(
            output_size=len(SPECTRA_LABELS),
            image_size=config.image_size,
            dropout_rate=config.dropout_rate,
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = create_optimizer(model, config)

        best_fold_f1 = -1.0
        best_fold_precision = 0.0
        best_fold_recall = 0.0

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

            if validation_result["f1_score"] > best_fold_f1:
                best_fold_f1 = validation_result["f1_score"]
                best_fold_precision = validation_result["precision"]
                best_fold_recall = validation_result["recall"]

            print(
                "Epoch {}/{} | "
                "Train Loss: {:.4f} | "
                "Val Loss: {:.4f} | "
                "Val F1: {:.4f}".format(
                    epoch + 1,
                    config.epochs,
                    train_result["loss"],
                    validation_result["loss"],
                    validation_result["f1_score"],
                )
            )

        fold_results.append({
            "fold": fold_index + 1,
            "best_validation_f1": best_fold_f1,
            "best_validation_precision": best_fold_precision,
            "best_validation_recall": best_fold_recall,
        })

    results_dataframe = pd.DataFrame(fold_results)

    summary = {
        "fold": "mean",
        "best_validation_f1": results_dataframe["best_validation_f1"].mean(),
        "best_validation_precision": results_dataframe["best_validation_precision"].mean(),
        "best_validation_recall": results_dataframe["best_validation_recall"].mean(),
    }

    results_dataframe = pd.concat(
        [results_dataframe, pd.DataFrame([summary])],
        ignore_index=True,
    )

    results_path = output_dir / "cross_validation_results.csv"
    results_dataframe.to_csv(results_path, index=False)

    print("\nCross Validation finalizado.")
    print("Resultados salvos em:", results_path)
    print(results_dataframe)

    return results_dataframe


if __name__ == "__main__":
    run_cross_validation()