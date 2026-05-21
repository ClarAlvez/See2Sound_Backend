from itertools import product
from pathlib import Path

import pandas as pd

from ai.spectra.training.config import SpectraTrainingConfig
from ai.spectra.training.train import train_spectra_model


def run_grid_search(
    dataset_path="data/datasets/spectra_labels.csv",
    output_dir="data/models/spectra_grid_search",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = {
        "learning_rate": [0.001, 0.0005],
        "batch_size": [8, 16],
        "dropout_rate": [0.2, 0.3],
        "weight_decay": [0.0001, 0.00001],
        "optimizer_name": ["adam"],
    }

    keys = list(grid.keys())
    combinations = list(product(*grid.values()))

    results = []

    for index, values in enumerate(combinations):
        params = dict(zip(keys, values))

        print("\nGrid Search {}/{}".format(index + 1, len(combinations)))
        print("Parâmetros:", params)

        config = SpectraTrainingConfig(
            epochs=10,
            learning_rate=params["learning_rate"],
            batch_size=params["batch_size"],
            dropout_rate=params["dropout_rate"],
            weight_decay=params["weight_decay"],
            optimizer_name=params["optimizer_name"],
        )

        run_output_dir = output_dir / "run_{}".format(index + 1)

        result = train_spectra_model(
            dataset_path=dataset_path,
            output_dir=run_output_dir,
            config=config,
        )

        results.append({
            "run": index + 1,
            "learning_rate": params["learning_rate"],
            "batch_size": params["batch_size"],
            "dropout_rate": params["dropout_rate"],
            "weight_decay": params["weight_decay"],
            "optimizer_name": params["optimizer_name"],
            "best_validation_f1": result["best_validation_f1"],
            "test_f1_score": result["test_result"]["f1_score"],
            "best_model_path": result["best_model_path"],
        })

    results_dataframe = pd.DataFrame(results)

    results_dataframe = results_dataframe.sort_values(
        by="best_validation_f1",
        ascending=False,
    )

    results_path = output_dir / "grid_search_results.csv"
    results_dataframe.to_csv(results_path, index=False)

    print("\nGrid Search finalizado.")
    print("Resultados salvos em:", results_path)
    print(results_dataframe.head())

    return results_dataframe


if __name__ == "__main__":
    run_grid_search()