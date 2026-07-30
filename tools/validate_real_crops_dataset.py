import argparse
from pathlib import Path

import pandas as pd

from ai.spectra.labels.label_sets import SPECTRA_PERSON_LABELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="data/datasets/manual_review/spectra_person_real_crops_review.csv",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    if "frame_path" not in df.columns:
        raise ValueError("CSV precisa ter coluna frame_path.")

    print("Arquivo:", csv_path)
    print("Total de linhas:", len(df))

    missing_images = []

    for path_text in df["frame_path"].head(200):
        path = Path(str(path_text))

        if not path.exists():
            missing_images.append(str(path))

    if missing_images:
        print("\nImagens ausentes nas primeiras 200:")
        for path in missing_images[:20]:
            print("-", path)
    else:
        print("\nAs primeiras 200 imagens existem.")

    for label in SPECTRA_PERSON_LABELS:
        if label not in df.columns:
            df[label] = 0

    counts = df[SPECTRA_PERSON_LABELS].sum().sort_values(ascending=False)

    print("\nDistribuição por label:")
    print(counts)

    print("\nLabels zeradas:")
    print(list(counts[counts == 0].index))

    print("\nMédia de labels positivas por crop:")
    print(df[SPECTRA_PERSON_LABELS].sum(axis=1).mean())


if __name__ == "__main__":
    main()