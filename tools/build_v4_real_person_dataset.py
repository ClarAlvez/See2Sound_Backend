import argparse
from pathlib import Path

import pandas as pd

from ai.spectra.Person.labels import SPECTRA_PERSON_LABELS


def normalize_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)

    if "frame_path" not in df.columns:
        raise ValueError(f"{csv_path} não possui frame_path.")

    for label in SPECTRA_PERSON_LABELS:
        if label not in df.columns:
            df[label] = 0

    metadata_cols = [
        column
        for column in df.columns
        if column.startswith("source_")
        or column.startswith("bbox_")
        or column in ["detector_confidence"]
    ]

    keep_cols = ["frame_path"] + metadata_cols + SPECTRA_PERSON_LABELS

    return df[keep_cols]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-csv",
        default="data/datasets/spectra_person_v3_hair_labels.csv",
    )

    parser.add_argument(
        "--real-crops-csv",
        default="data/datasets/manual_review/spectra_person_real_crops_review.csv",
    )

    parser.add_argument(
        "--output-csv",
        default="data/datasets/spectra_person_v4_real_crops_labels.csv",
    )

    parser.add_argument(
        "--real-repeat",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    base_csv = Path(args.base_csv)
    real_crops_csv = Path(args.real_crops_csv)
    output_csv = Path(args.output_csv)

    if not base_csv.exists():
        raise FileNotFoundError(f"Base CSV não encontrado: {base_csv}")

    if not real_crops_csv.exists():
        raise FileNotFoundError(f"CSV de crops reais não encontrado: {real_crops_csv}")

    base_df = normalize_dataset(base_csv)
    real_df = normalize_dataset(real_crops_csv)

    parts = [base_df]

    for _ in range(args.real_repeat):
        parts.append(real_df)

    merged = pd.concat(parts, ignore_index=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    print("\nDataset v4 criado com sucesso.")
    print("Base:", base_csv)
    print("Crops reais:", real_crops_csv)
    print("Repetições dos crops reais:", args.real_repeat)
    print("Saída:", output_csv)
    print("Total de linhas:", len(merged))

    counts = merged[SPECTRA_PERSON_LABELS].sum().sort_values(ascending=False)

    print("\nDistribuição por label:")
    print(counts)

    print("\nLabels zeradas:")
    print(list(counts[counts == 0].index))

    print("\nMédia de labels positivas por imagem:")
    print(merged[SPECTRA_PERSON_LABELS].sum(axis=1).mean())


if __name__ == "__main__":
    main()
