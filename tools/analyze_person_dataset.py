import argparse
from pathlib import Path

import pandas as pd

from ai.spectra.labels.label_sets import SPECTRA_PERSON_LABELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    df = pd.read_csv(csv_path)

    print("\nArquivo:", csv_path)
    print("Total de linhas:", len(df))

    missing_columns = [
        label for label in SPECTRA_PERSON_LABELS
        if label not in df.columns
    ]

    if missing_columns:
        print("\nLabels ausentes no CSV:")
        print(missing_columns)
        return

    print("\nVerificando imagens...")
    missing_paths = []

    for path in df["frame_path"].head(100):
        if not Path(path).exists():
            missing_paths.append(path)

    if missing_paths:
        print("Algumas imagens não existem:")
        for path in missing_paths[:10]:
            print("-", path)
    else:
        print("As primeiras 100 imagens existem.")

    label_cols = SPECTRA_PERSON_LABELS

    counts = df[label_cols].sum().sort_values(ascending=False)

    print("\nDistribuição por label:")
    print(counts)

    zero_labels = list(counts[counts == 0].index)

    print("\nLabels zeradas:")
    print(zero_labels)

    positive_per_image = df[label_cols].sum(axis=1)

    print("\nMédia de labels positivas por imagem:")
    print(positive_per_image.mean())

    print("\nDistribuição de labels positivas por imagem:")
    print(positive_per_image.describe())

    print("\nPrimeiras linhas:")
    print(df.head())


if __name__ == "__main__":
    main()