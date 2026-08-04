import argparse
from pathlib import Path

import pandas as pd

from ai.spectra.Person.labels import SPECTRA_PERSON_LABELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)

    print("Linhas originais:", len(df))

    if "frame_path" not in df.columns:
        raise ValueError("CSV não possui coluna frame_path.")

    # Remove linhas vazias ou quebradas.
    df["frame_path"] = df["frame_path"].astype("string")
    df = df.dropna(subset=["frame_path"])
    df = df[df["frame_path"].str.strip() != ""]
    df = df[df["frame_path"].str.lower() != "nan"]

    print("Linhas após remover frame_path vazio:", len(df))

    # Mantém só imagens que existem.
    exists_mask = df["frame_path"].apply(lambda value: Path(str(value)).exists())

    missing = df[~exists_mask]

    if len(missing) > 0:
        print("\nImagens ausentes removidas:", len(missing))
        print("Exemplos:")
        print(missing["frame_path"].head(20).to_string(index=False))

    df = df[exists_mask].copy()

    # Garante que todas as labels existem e são 0/1.
    for label in SPECTRA_PERSON_LABELS:
        if label not in df.columns:
            df[label] = 0

        df[label] = pd.to_numeric(df[label], errors="coerce").fillna(0).astype(int)
        df[label] = df[label].clip(0, 1)

    # Garante person = 1 quando vier de crop real ou dataset de pessoa.
    df["person"] = 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\nCSV limpo salvo em:")
    print(output_path)

    print("\nLinhas finais:", len(df))

    counts = df[SPECTRA_PERSON_LABELS].sum().sort_values(ascending=False)

    print("\nDistribuição por label:")
    print(counts)

    print("\nLabels zeradas:")
    print(list(counts[counts == 0].index))

    print("\nMédia de labels positivas por imagem:")
    print(df[SPECTRA_PERSON_LABELS].sum(axis=1).mean())


if __name__ == "__main__":
    main()
