import argparse
from pathlib import Path

import pandas as pd

from ai.spectra.Person.labels import SPECTRA_PERSON_LABELS


FOCUS_LABELS = [
    "glasses",
    "hat",
    "dress",
    "bald_hair",
    "orange_clothes",
    "pink_clothes",
    "purple_clothes",
    "brown_clothes",
    "gray_clothes",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--target-total", type=int, default=30000)
    parser.add_argument("--per-focus-label", type=int, default=2500)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {input_path}")

    df = pd.read_csv(input_path)

    selected_parts = []

    print("\nSelecionando exemplos das labels fracas:")

    for label in FOCUS_LABELS:
        if label not in df.columns:
            print(f"- {label}: coluna ausente")
            continue

        positives = df[df[label] == 1]

        if positives.empty:
            print(f"- {label}: 0 exemplos positivos")
            continue

        sample_size = min(args.per_focus_label, len(positives))

        sampled = positives.sample(
            n=sample_size,
            random_state=args.random_state,
            replace=False,
        )

        selected_parts.append(sampled)

        print(f"- {label}: {sample_size} exemplos")

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()

    selected = selected.drop_duplicates(subset=["frame_path"])

    remaining_needed = args.target_total - len(selected)

    if remaining_needed > 0:
        remaining_pool = df[~df["frame_path"].isin(selected["frame_path"])]

        random_part = remaining_pool.sample(
            n=min(remaining_needed, len(remaining_pool)),
            random_state=args.random_state,
            replace=False,
        )

        selected = pd.concat([selected, random_part], ignore_index=True)

    selected = selected.drop_duplicates(subset=["frame_path"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_path, index=False)

    print("\nSubset balanceado salvo em:")
    print(output_path)

    print("\nTotal final:", len(selected))

    counts = selected[SPECTRA_PERSON_LABELS].sum().sort_values(ascending=False)

    print("\nDistribuição por label:")
    print(counts)

    print("\nLabels zeradas:")
    print(list(counts[counts == 0].index))

    print("\nMédia de labels positivas por imagem:")
    print(selected[SPECTRA_PERSON_LABELS].sum(axis=1).mean())


if __name__ == "__main__":
    main()
