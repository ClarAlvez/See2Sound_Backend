import argparse
from pathlib import Path

import pandas as pd

from ai.spectra.labels.label_sets import SPECTRA_PERSON_LABELS


def normalize_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "frame_path" not in df.columns:
        raise ValueError(f"{csv_path} não possui coluna frame_path.")

    for label in SPECTRA_PERSON_LABELS:
        if label not in df.columns:
            df[label] = 0

    metadata_cols = [
        column
        for column in df.columns
        if column.startswith("source_")
    ]

    keep_cols = ["frame_path"] + metadata_cols + SPECTRA_PERSON_LABELS

    return df[keep_cols]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frames = []

    for input_path in args.inputs:
        path = Path(input_path)

        if not path.exists():
            raise FileNotFoundError(f"CSV não encontrado: {path}")

        print("Lendo:", path)
        frames.append(normalize_dataset(path))

    merged = pd.concat(frames, ignore_index=True)

    before = len(merged)
    merged = merged.drop_duplicates(subset=["frame_path"])
    after = len(merged)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged.to_csv(output_path, index=False)

    print("\nDataset unido salvo em:", output_path)
    print("Linhas antes:", before)
    print("Linhas depois de remover duplicados:", after)

    counts = merged[SPECTRA_PERSON_LABELS].sum().sort_values(ascending=False)

    print("\nDistribuição por label:")
    print(counts)

    print("\nLabels zeradas:")
    print(list(counts[counts == 0].index))

    print("\nMédia de labels positivas por imagem:")
    print(merged[SPECTRA_PERSON_LABELS].sum(axis=1).mean())


if __name__ == "__main__":
    main()